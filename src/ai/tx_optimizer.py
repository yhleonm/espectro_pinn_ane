"""
Optimizador de Ubicación de Transmisores FM Comunitarios.

Usa la arquitectura híbrida PINN (FSPL + Eikonal + Shadow Mask Knife-Edge)
como función objetivo para maximizar la cobertura social ponderada por
población en las zonas de déficit de Bogotá.

Estrategia: Grid Search sobre zonas de déficit + evaluación PINN completa.
La Shadow Mask (Knife-Edge) es no-diferenciable → no se usa gradient descent.

Restricciones regulatorias:
- Res. 415/2010 ANE: Clase D (Comunitaria), PRA máx 250W (54 dBm)
- Frecuencia: Banda FM 88-108 MHz
- Altura antena: 15-30m sobre terreno
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from src.ai.pinn_trainer import PINNTrainer
from src.parsers.srtm_reader import SRTMReader, download_srtm_tile
from src.propagation.fspl import fspl_db
import geopandas as gpd
import pandas as pd
import unicodedata
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter


class TXOptimizer:
    """
    Optimizador de ubicación de Transmisores basado en evaluación PINN real.
    
    Para N estaciones comunitarias, encuentra las coordenadas que maximizan
    la cobertura de población en zonas de déficit (Suba, Usme, Ciudad Bolívar).
    """
    
    # Parámetros de emisora comunitaria (Clase D)
    TX_POWER_DBM = 54.0      # 250W PRA máximo Clase D
    FREQUENCY_MHZ = 94.9     # Frecuencia comunitaria típica
    TX_HEIGHT_M = 20.0       # Altura antena sobre terreno
    RADIUS_KM = 15.0         # Radio de cobertura comunitaria
    RESOLUTION = 60          # Resolución grid (rápido para evaluación)
    EPOCHS_FAST = 500        # Épocas para grid search
    EPOCHS_FINAL = 1500      # Épocas para evaluación final
    THRESHOLD_DBUVM = 66.0   # Umbral de cobertura de calidad
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.srtm_dir = Path("data/srtm")
        self.srtm_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar localidades y población
        self.localidades = None
        self.pop_by_loc = {}
        self._load_population_data()
    
    def _load_population_data(self):
        """Carga localidades geográficas y proyección de población 2026."""
        loc_path = Path("data/localidades_bogota.geojson")
        pob_path = Path("data/poblacion_bogota_localidad.csv")
        
        if loc_path.exists():
            self.localidades = gpd.read_file(loc_path).to_crs(epsg=4326)
        
        def normalize_name(name):
            if not isinstance(name, str): return ""
            name = name.upper().strip()
            name = "".join(c for c in unicodedata.normalize('NFD', name) 
                          if unicodedata.category(c) != 'Mn')
            return name
        
        if pob_path.exists():
            df_pob = pd.read_csv(pob_path, sep=';')
            df_2026 = df_pob[df_pob['ANO'] == 2026].groupby(
                'NOMBRE_LOCALIDAD')['POBLACION'].sum().reset_index()
            for _, row in df_2026.iterrows():
                self.pop_by_loc[normalize_name(row['NOMBRE_LOCALIDAD'])] = int(row['POBLACION'])
        
        self._normalize_name = normalize_name
    
    def _get_terrain_elevation(self, lat: float, lon: float) -> float:
        """Obtiene elevación del terreno en un punto."""
        try:
            tile_path = download_srtm_tile(
                int(np.floor(lat)), int(np.floor(lon)), 
                output_dir=self.srtm_dir
            )
            reader = SRTMReader(tile_path)
            z = reader.get_elevation(lat, lon)
            return z if not np.isnan(z) else 2600.0
        except Exception:
            return 2600.0
    
    def _generate_candidates(self, n_candidates: int = 15) -> List[Dict]:
        """
        Genera candidatos de ubicación en zonas de déficit.
        
        Criterios:
        1. En localidades con cobertura < 50% (Suba, Usme, C. Bolívar)
        2. En puntos elevados (cerros menores, lomas) para buena línea de vista
        3. Separación mínima de 3km entre candidatos
        """
        # Zonas de déficit con centroides manuales bien distribuidos
        deficit_zones = [
            # Suba: norte-occidente, disperso. Múltiples candidatos.
            {"name": "Suba Norte",      "lat": 4.760, "lon": -74.095, "pop_weight": 1.3},
            {"name": "Suba Centro",     "lat": 4.740, "lon": -74.085, "pop_weight": 1.3},
            {"name": "Suba Rincón",     "lat": 4.755, "lon": -74.110, "pop_weight": 1.2},
            # Ciudad Bolívar: sur, terreno montañoso
            {"name": "C.Bolívar Norte", "lat": 4.575, "lon": -74.155, "pop_weight": 0.7},
            {"name": "C.Bolívar Centro","lat": 4.555, "lon": -74.160, "pop_weight": 0.7},
            {"name": "C.Bolívar Sur",   "lat": 4.535, "lon": -74.170, "pop_weight": 0.6},
            # Usme: suroriente, detrás de cerros. Difícil cobertura.
            {"name": "Usme Norte",      "lat": 4.510, "lon": -74.110, "pop_weight": 0.4},
            {"name": "Usme Centro",     "lat": 4.490, "lon": -74.105, "pop_weight": 0.4},
            # Engativá: occidente, parcialmente cubierto (69.6%), mejora marginal
            {"name": "Engativá Oeste",  "lat": 4.710, "lon": -74.120, "pop_weight": 0.9},
            # Puntos altos estratégicos (lomas que dan línea de vista)
            {"name": "Cerro Seco",      "lat": 4.560, "lon": -74.150, "pop_weight": 0.8},
            {"name": "Alto Cazucá",     "lat": 4.575, "lon": -74.185, "pop_weight": 0.6},
            {"name": "Loma Suba",       "lat": 4.755, "lon": -74.075, "pop_weight": 1.0},
            # Borde occidental (señal debilitada por distancia, no por shadow)
            {"name": "Fontibón Oeste",  "lat": 4.680, "lon": -74.155, "pop_weight": 0.5},
            {"name": "Bosa Occidental", "lat": 4.600, "lon": -74.200, "pop_weight": 0.5},
            {"name": "Soacha Norte",    "lat": 4.585, "lon": -74.210, "pop_weight": 0.4},
        ]
        
        # Enriquecer con elevación real
        for c in deficit_zones:
            c["elevation_m"] = self._get_terrain_elevation(c["lat"], c["lon"])
            c["tx_z_m"] = c["elevation_m"] + self.TX_HEIGHT_M
        
        return deficit_zones[:n_candidates]
    
    def evaluate_candidate(self, lat: float, lon: float, 
                           epochs: int = 500, label: str = "") -> Dict:
        """
        Evalúa una ubicación candidata usando la PINN híbrida completa.
        
        Returns:
            Dict con: hab_cubiertos, area_km2, coverage_pct, métricas por localidad
        """
        z_terrain = self._get_terrain_elevation(lat, lon)
        tx_z = z_terrain + self.TX_HEIGHT_M
        
        trainer = PINNTrainer(
            tx_lat=lat, tx_lon=lon, tx_height_m=tx_z,
            tx_power_dbm=self.TX_POWER_DBM,
            frequency_mhz=self.FREQUENCY_MHZ,
            radius_km=self.RADIUS_KM,
            resolution=self.RESOLUTION,
            epochs_ia=epochs
        )
        
        cached = trainer.load_weights()
        if not cached:
            trainer.train(epochs=epochs)
            trainer.save_weights()
        
        heatmap = trainer.infer_grid().numpy()
        heatmap = gaussian_filter(heatmap, sigma=0.8)
        field = heatmap + 20 * np.log10(self.FREQUENCY_MHZ) + 77.2
        
        # Área de cobertura
        pixel_area = (2 * self.RADIUS_KM / self.RESOLUTION) ** 2
        area_km2 = float(np.sum(field >= self.THRESHOLD_DBUVM) * pixel_area)
        
        # Spatial join rápido (subsampled)
        lats_1d = trainer.boundary.lats_1d
        lons_1d = trainer.boundary.lons_1d
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        
        step = 3  # Subsample agresivo para velocidad
        lat_sub = lat_grid[::step, ::step].flatten()
        lon_sub = lon_grid[::step, ::step].flatten()
        sig_sub = field[::step, ::step].flatten()
        
        total_hab = 0
        if self.localidades is not None:
            points = [Point(x, y) for x, y in zip(lon_sub, lat_sub)]
            gdf = gpd.GeoDataFrame(
                {'signal': sig_sub, 'geometry': points}, crs="EPSG:4326"
            )
            joined = gpd.sjoin(gdf, self.localidades[['LocNombre', 'geometry']], 
                              how="inner", predicate="within")
            
            for loc_name in joined['LocNombre'].unique():
                pts = joined[joined['LocNombre'] == loc_name]
                n_good = int((pts['signal'] >= self.THRESHOLD_DBUVM).sum())
                pct = n_good / max(len(pts), 1) * 100
                pop = self.pop_by_loc.get(self._normalize_name(loc_name), 0)
                total_hab += int(pop * pct / 100)
        
        if label:
            print(f"    {label}: area={area_km2:.1f}km² | hab={total_hab:,} | z={z_terrain:.0f}m")
        
        return {
            "lat": lat, "lon": lon,
            "elevation_m": z_terrain,
            "area_km2": area_km2,
            "hab_cubiertos": total_hab,
            "cached": cached
        }
    
    def optimize(self, n_stations: int = 3, epochs: int = 500) -> Dict:
        """
        Encuentra las N mejores ubicaciones para estaciones comunitarias.
        
        Algoritmo:
        1. Genera candidatos en zonas de déficit
        2. Evalúa cada candidato con PINN completa
        3. Selección greedy: top-1, luego top-2 (excluyendo <5km del 1), etc.
        4. Re-evalúa los top-N con más épocas
        
        Returns:
            Dict con stations, baseline, improvement
        """
        print("=" * 70)
        print(f"🚀 OPTIMIZADOR TX — {n_stations} Estaciones Comunitarias")
        print(f"   PRA: {self.TX_POWER_DBM} dBm ({10**((self.TX_POWER_DBM-30)/10):.0f}W)")
        print(f"   Freq: {self.FREQUENCY_MHZ} MHz | Radio: {self.RADIUS_KM} km")
        print("=" * 70)
        
        # 1. Generar candidatos
        candidates = self._generate_candidates(n_candidates=15)
        print(f"\n📍 Evaluando {len(candidates)} candidatos...")
        
        # 2. Evaluar todos
        results = []
        for c in candidates:
            r = self.evaluate_candidate(
                c["lat"], c["lon"], epochs=epochs,
                label=c["name"]
            )
            r["name"] = c["name"]
            r["pop_weight"] = c["pop_weight"]
            # Score compuesto: hab × peso de déficit
            r["score"] = r["hab_cubiertos"] * c["pop_weight"]
            results.append(r)
        
        # 3. Selección greedy con restricción de separación
        results.sort(key=lambda x: x["score"], reverse=True)
        
        selected = []
        min_sep_deg = 0.04  # ~4.5km separación mínima
        
        print(f"\n🎯 Seleccionando top-{n_stations} con separación >{min_sep_deg*111:.1f}km...")
        
        for r in results:
            # Verificar separación con ya seleccionados
            too_close = False
            for s in selected:
                d = np.sqrt((r["lat"] - s["lat"])**2 + (r["lon"] - s["lon"])**2)
                if d < min_sep_deg:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(r)
                print(f"  ✅ #{len(selected)}: {r['name']} "
                      f"({r['lat']:.4f}, {r['lon']:.4f}) "
                      f"| hab={r['hab_cubiertos']:,} | score={r['score']:,.0f}")
            
            if len(selected) >= n_stations:
                break
        
        # 4. Formatear resultado final
        stations = []
        for i, s in enumerate(selected):
            stations.append({
                "id": i + 1,
                "name": s["name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "elevation_m": s["elevation_m"],
                "pra_dbm": self.TX_POWER_DBM,
                "pra_w": 10**((self.TX_POWER_DBM - 30) / 10),
                "frequency_mhz": self.FREQUENCY_MHZ,
                "area_km2": s["area_km2"],
                "hab_cubiertos": s["hab_cubiertos"]
            })
        
        total_new_hab = sum(s["hab_cubiertos"] for s in stations)
        baseline_hab = 5_291_994  # Del análisis social validado
        baseline_pct = 66.8
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTADO FINAL")
        print(f"{'='*70}")
        print(f"  Baseline (Estación 1457):  {baseline_hab:,} hab ({baseline_pct}%)")
        print(f"  Nuevas estaciones:         {total_new_hab:,} hab adicionales estimados")
        print(f"  (Nota: cobertura incremental real requiere evaluación combinada)")
        
        return {
            "stations": stations,
            "baseline": {
                "hab_cubiertos": baseline_hab,
                "pct_cubierta": baseline_pct,
                "estacion": "1457 (Cerro Antenas)"
            },
            "new_stations_hab": total_new_hab,
            "optimizer_params": {
                "pra_dbm": self.TX_POWER_DBM,
                "frequency_mhz": self.FREQUENCY_MHZ,
                "radius_km": self.RADIUS_KM,
                "n_candidates_evaluated": len(candidates)
            }
        }


if __name__ == "__main__":
    opt = TXOptimizer()
    res = opt.optimize(n_stations=3, epochs=500)
    print("\n✅ UBICACIONES ÓPTIMAS ENCONTRADAS:")
    for s in res["stations"]:
        print(f"  Estación {s['id']} ({s['name']}): "
              f"Lat={s['lat']:.4f}, Lon={s['lon']:.4f}, "
              f"PRA={s['pra_w']:.0f}W, Hab={s['hab_cubiertos']:,}")
