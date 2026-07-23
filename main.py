from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import time
import unicodedata
from typing import List, Dict
from pathlib import Path

# Configurar path
sys.path.append(os.path.dirname(__file__))

from src.propagation.fspl import fspl_db
from src.geo.coordinates import haversine_distance, transform_wgs84_to_magna
from src.propagation.coverage import generate_fspl_coverage, apply_terrain_mask
from src.parsers.srtm_reader import SRTMReader, download_srtm_tile
from src.ai.tx_optimizer import TXOptimizer
from src.services.station_service import station_service
from src.ai.inverse_solver import InverseSolver

app = FastAPI(title="RF Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# National FM Inventory Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/stations/departments")
def get_departments():
    """Lista de departamentos únicos con estaciones FM."""
    return station_service.get_departments()


@app.get("/api/stations/cities/{department}")
def get_cities(department: str):
    """Lista de municipios con estaciones FM en un departamento."""
    return station_service.get_cities(department)


@app.get("/api/stations")
def get_stations(dept: str = None, city: str = None, freq: float = None):
    """Consulta de estaciones con filtros opcionales."""
    return station_service.get_stations(department=dept, city=city, freq_mhz=freq)


@app.get("/api/stations/{station_id}")
def get_station(station_id: int):
    """Detalle de una estación por su ID ANE."""
    s = station_service.get_station_by_id(station_id)
    if not s:
        raise HTTPException(status_code=404, detail="Station not found")
    return s


class CochannelRequest(BaseModel):
    station_id: int
    radius_km: float = 100.0


@app.post("/api/interference/cochannel")
def cochannel_interference(req: CochannelRequest):
    """Busca estaciones co-canal (misma frecuencia) dentro de un radio."""
    station = station_service.get_station_by_id(req.station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    conflicts = station_service.find_cochannel(req.station_id, req.radius_km)
    return {
        "station": station,
        "conflicts": conflicts,
        "total_conflicts": len(conflicts),
    }


class CitizenRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 50.0


@app.post("/api/citizen/signal_at_point")
def signal_at_point(req: CitizenRequest):
    """Estima las señales FM recibidas en un punto geográfico (ciudadano)."""
    signals = station_service.predict_signals_at_point(
        req.lat, req.lon, req.radius_km
    )
    return {
        "point": {"lat": req.lat, "lon": req.lon},
        "signals": signals,
        "total_stations": len(signals),
    }


# Modelos
class Point(BaseModel):
    lat: float
    lon: float

class FSPLRequest(BaseModel):
    tx: Point
    rx: Point
    frequency_mhz: float
    tx_power_dbm: float

class PINNRequest(BaseModel):
    tx: Point
    frequency_mhz: float
    tx_power_dbm: float
    tx_azimuth_deg: float = 0.0
    tx_tilt_deg: float = 0.0
    tx_hpbw_h_deg: float = 65.0
    tx_hpbw_v_deg: float = 65.0
    radius_km: float
    res_px: int
    epochs_ia: int = 500
    city: str = None

class CompareProfileRequest(BaseModel):
    tx: Point
    rx: Point
    tx_height: float = 30.0
    rx_height: float = 2.0
    tx_power_dbm: float
    frequency_mhz: float
    tx_azimuth_deg: float = 0.0
    tx_tilt_deg: float = 0.0
    tx_hpbw_h_deg: float = 65.0
    tx_hpbw_v_deg: float = 65.0
    epochs_ia: int = 500
    n_points: int = 200
    city: str = None

class OptimizerRequest(BaseModel):
    n_stations: int = 3
    epochs: int = 50

# Utils
def resolve_srtm_tiles(lat1, lon1, lat2, lon2):
    lats = np.linspace(lat1, lat2, 5)
    lons = np.linspace(lon1, lon2, 5)
    tiles = set()
    for lat, lon in zip(lats, lons):
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        latin = int(np.floor(abs(lat)))
        lonin = int(np.floor(abs(lon)))
        tiles.add((lat, lon, f"{ns}{latin:02d}{ew}{lonin:03d}"))
    return list(tiles)

# Endpoints
@app.get("/")
def read_root():
    return {"status": "ok", "message": "RF Simulator API is running"}

@app.post("/api/fspl")
def calculate_fspl(req: FSPLRequest):
    dist_km = haversine_distance(req.tx.lat, req.tx.lon, req.rx.lat, req.rx.lon)
    fspl_loss = fspl_db(dist_km, req.frequency_mhz)
    rx_power = req.tx_power_dbm - fspl_loss
    field_strength = rx_power + 20 * np.log10(req.frequency_mhz) + 77.2
    return {"distance_km": dist_km, "fspl_db": fspl_loss, "rx_power_dbm": rx_power, "field_strength_dbuvm": field_strength}

@app.post("/api/pinn/train_and_infer")
def train_and_infer_pinn(req: PINNRequest):
    try:
        import time
        from scipy.ndimage import gaussian_filter
        from src.ai.pinn_trainer import PINNTrainer
        
        start_time = time.time()
        # 1. Elevación TX
        data_dir = Path("data/srtm")
        data_dir.mkdir(parents=True, exist_ok=True)
        tile_path = download_srtm_tile(int(np.floor(req.tx.lat)), int(np.floor(req.tx.lon)), output_dir=data_dir)
        reader = SRTMReader(tile_path)
        z_terrain = reader.get_elevation(req.tx.lat, req.tx.lon)
        if np.isnan(z_terrain): z_terrain = 0.0  # Generic fallback, works for any city
        
        # Determinar ciudad para clutter (slug)
        city_slug = "bogota"
        if req.city:
            # Normalizar para quitar tildes (medellín -> medellin)
            city_slug = unicodedata.normalize('NFKD', req.city).encode('ascii', 'ignore').decode('ascii').lower()
            city_slug = city_slug.replace(" ", "_").replace(".", "").replace(",", "")
        else:
            nearby = station_service.get_stations(freq_mhz=req.frequency_mhz)
            if nearby:
                ref_lat, ref_lon = req.tx.lat, req.tx.lon
                best_dist = float('inf')
                for s in nearby:
                    d = haversine_distance(ref_lat, ref_lon, s['lat'], s['lon'])
                    if d < best_dist:
                        best_dist = d
                        city_slug = unicodedata.normalize('NFKD', s['municipio']).encode('ascii', 'ignore').decode('ascii').lower()
                        city_slug = city_slug.replace(" ", "_").replace(".", "").replace(",", "").replace(",", "")

        absolute_tx_z = z_terrain + 30.0
        trainer = PINNTrainer(
            tx_lat=req.tx.lat, tx_lon=req.tx.lon, tx_height_m=absolute_tx_z,
            tx_power_dbm=req.tx_power_dbm, frequency_mhz=req.frequency_mhz,
            radius_km=req.radius_km, resolution=req.res_px, epochs_ia=req.epochs_ia,
            city_slug=city_slug
        )
        
        cached = trainer.load_weights()
        if not cached:
            trainer.train(epochs=req.epochs_ia)
            trainer.save_weights()
        
        train_time = time.time() - start_time
        
        ai_power_dbm = trainer.infer_grid().numpy()
        ai_power_dbm = gaussian_filter(ai_power_dbm, sigma=1.5)
        
        # Color mapping (Aesthetic Jet) - Restaurando gradiente
        # vmax ahora depende de la potencia TX para ver el centro rojo y desvanecimiento
        cmap = plt.get_cmap('jet')
        norm = mcolors.Normalize(vmin=-100, vmax=req.tx_power_dbm)
        rgba_image = cmap(norm(ai_power_dbm))
        
        # Radial Mask
        grid_h, grid_w = ai_power_dbm.shape
        cy, cx = grid_h // 2, grid_w // 2
        Y, X = np.ogrid[:grid_h, :grid_w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_radius = min(cx, cy)
        
        # Máscara binaria para métricas (solo dentro del radio de análisis)
        metric_mask = dist_from_center <= max_radius
        
        alpha = np.where(metric_mask, 0.7, 0.0)
        rgba_image[:, :, 3] = alpha
        
        rgba_image = np.flipud(rgba_image)
        import PIL.Image
        img = PIL.Image.fromarray((rgba_image * 255).astype(np.uint8), 'RGBA')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Métricas ANE (FILTRADAS POR MÁSCARA RADIAL)
        ai_field_dbuvm = ai_power_dbm + 20 * np.log10(req.frequency_mhz) + 77.2
        total_side_km = 2.0 * req.radius_km
        pixel_area_km2 = (total_side_km / req.res_px)**2
        
        # Aplicar máscara a los cálculos de área
        covered_66_73 = (ai_field_dbuvm >= 66) & (ai_field_dbuvm < 74) & metric_mask
        covered_74_85 = (ai_field_dbuvm >= 74) & (ai_field_dbuvm < 86) & metric_mask
        covered_86_95 = (ai_field_dbuvm >= 86) & (ai_field_dbuvm < 96) & metric_mask
        covered_96_plus = (ai_field_dbuvm >= 96) & metric_mask
        
        areas_by_band = {
            "66-73": float(np.sum(covered_66_73) * pixel_area_km2),
            "74-85": float(np.sum(covered_74_85) * pixel_area_km2),
            "86-95": float(np.sum(covered_86_95) * pixel_area_km2),
            "96+": float(np.sum(covered_96_plus) * pixel_area_km2)
        }
        
        total_area = float(np.sum(ai_field_dbuvm[metric_mask] >= 66) * pixel_area_km2)
        
        return {
            "image_base64": f"data:image/png;base64,{img_str}",
            "bounds": [
                [trainer.boundary.lats_1d.min(), trainer.boundary.lons_1d.min()], 
                [trainer.boundary.lats_1d.max(), trainer.boundary.lons_1d.max()]
            ],
            "train_time_sec": train_time,
            "cached": cached,
            "quant_metrics": {
                "area_sqkm": total_area,
                "loss_final": float(trainer.last_loss) if hasattr(trainer, 'last_loss') else 0.0
            },
            "regulatory_metrics": {
                "areas_km2": areas_by_band
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pinn/compare_profile")
def compare_profile(req: CompareProfileRequest):
    try:
        import math
        from src.ai.pinn_trainer import PINNTrainer
        # 1. Tiles SRTM
        required_tiles = resolve_srtm_tiles(req.tx.lat, req.tx.lon, req.rx.lat, req.rx.lon)
        data_dir = Path("data/srtm")
        readers = []
        for lat, lon, tile_name in required_tiles:
            try:
                tile_path = download_srtm_tile(int(np.floor(lat)), int(np.floor(lon)), output_dir=data_dir)
                readers.append(SRTMReader(tile_path))
            except: pass
                
        # 2. Perfil 1D
        lats = np.linspace(req.tx.lat, req.rx.lat, req.n_points)
        lons = np.linspace(req.tx.lon, req.rx.lon, req.n_points)
        elevations = np.zeros(req.n_points)
        dist_array = np.zeros(req.n_points)
        
        tx_base_m = 2600.0
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            if i > 0: dist_array[i] = dist_array[i-1] + haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            for r in readers:
                if r.info.lat_sw <= lat <= r.info.lat_ne and r.info.lon_sw <= lon <= r.info.lon_ne:
                    z = r.get_elevation(lat, lon)
                    if not np.isnan(z):
                        elevations[i] = z
                        if i == 0: tx_base_m = z
                    break
                    
        tx_z = tx_base_m + req.tx_height
        
        # Determinar ciudad
        city_slug = "bogota"
        if req.city:
            city_slug = unicodedata.normalize('NFKD', req.city).encode('ascii', 'ignore').decode('ascii').lower()
            city_slug = city_slug.replace(" ", "_").replace(".", "").replace(",", "")
            
        # 3. Inferencia PINN
        trainer = PINNTrainer(
            tx_lat=req.tx.lat, tx_lon=req.tx.lon, tx_height_m=tx_z,
            tx_power_dbm=req.tx_power_dbm, frequency_mhz=req.frequency_mhz,
            radius_km=30.0, resolution=50, city_slug=city_slug
        )
        if not trainer.load_weights():
            trainer.train(epochs=req.epochs_ia)
            trainer.save_weights()
            
        # Capa 3: Máscara de Sombra Topográfica (Knife-Edge vectorizado desde DEM)
        print(f"  [Trainer] Shadow Mask activa: {trainer.boundary.use_shadow_mask}")
        
        pinn_dbm = trainer.infer_points(lats, lons, elevations, rx_height_m=req.rx_height)
        
        # 4. Baselines (FSPL y Knife-Edge simplificado para el gráfico)
        fspl_dbm = []
        for d in dist_array:
            fspl_dbm.append(req.tx_power_dbm + 3.0 - fspl_db(max(0.001, d), req.frequency_mhz))

        return {
            "distances_km": dist_array.tolist(),
            "elevations_m": elevations.tolist(),
            "tx_z_m": tx_z,
            "fspl_dbm": fspl_dbm,
            "pinn_dbm": pinn_dbm.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimizer/stations")
def optimize_stations(req: OptimizerRequest):
    try:
        opt = TXOptimizer()
        results = opt.optimize(n_stations=req.n_stations, epochs=req.epochs)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/barrios")
def get_social_barrios():
    try:
        import geopandas as gpd
        import json
        barrios_path = Path("data/barrios/SECTOR.geojson")
        if not barrios_path.exists(): raise HTTPException(status_code=404, detail="GeoJSON not found")
        gdf = gpd.read_file(barrios_path)
        gdf = gdf.to_crs(epsg=4326)
        gdf['geometry'] = gdf['geometry'].simplify(0.0001)
        return json.loads(gdf.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- DETECCIÓN ILEGAL (PROBLEMA INVERSO) ---
class Measurement(BaseModel):
    lat: float
    lon: float
    dbm: float
    frequency_mhz: float
    uncertainty_db: float = 2.0

class DetectionRequest(BaseModel):
    measurements: List[Measurement]
    frequency_mhz: float
    search_center: Dict[str, float]
    radius_km: float = 20.0

@app.post("/api/interference/detect_illegal")
def detect_illegal(req: DetectionRequest):
    if len(req.measurements) < 4:
        raise HTTPException(status_code=400, detail="Se requieren al menos 4 mediciones para triangulación con análisis estadístico de incertidumbre")
    
    solver = InverseSolver(
        frequency_mhz=req.frequency_mhz,
        search_center=req.search_center,
        radius_km=req.radius_km
    )
    
    results = solver.solve([m.dict() for m in req.measurements])
    return results

@app.post("/api/social/analysis")
def social_analysis(req: PINNRequest):
    """
    Análisis de Impacto Social con Spatial Join real.
    
    Flujo:
    1. Genera heatmap PINN (con Shadow Mask Knife-Edge)
    2. Convierte a field strength (dBuV/m)
    3. Spatial join con SECTOR.geojson (693 sectores catastrales)
    4. Cruza con localidades_bogota.geojson para nombre de localidad
    5. Cruza con población 2026 por localidad
    6. Calcula % cobertura y habitantes cubiertos por localidad
    """
    try:
        import time
        import geopandas as gpd
        import pandas as pd
        import json
        import unicodedata
        from scipy.ndimage import gaussian_filter
        from shapely.geometry import Point
        from src.ai.pinn_trainer import PINNTrainer
        
        start_time = time.time()
        
        # ---- 1. GENERAR HEATMAP PINN (con Shadow Mask) ----
        data_dir = Path("data/srtm")
        data_dir.mkdir(parents=True, exist_ok=True)
        tile_path = download_srtm_tile(int(np.floor(req.tx.lat)), int(np.floor(req.tx.lon)), output_dir=data_dir)
        reader = SRTMReader(tile_path)
        z_terrain = reader.get_elevation(req.tx.lat, req.tx.lon)
        if np.isnan(z_terrain): z_terrain = 0.0  # Generic fallback, works for any city
        
        # Determinar ciudad
        city_slug = "bogota"
        if req.city:
            city_slug = unicodedata.normalize('NFKD', req.city).encode('ascii', 'ignore').decode('ascii').lower()
            city_slug = city_slug.replace(" ", "_").replace(".", "").replace(",", "")
            
        absolute_tx_z = z_terrain + 30.0
        trainer = PINNTrainer(
            tx_lat=req.tx.lat, tx_lon=req.tx.lon, tx_height_m=absolute_tx_z,
            tx_power_dbm=req.tx_power_dbm, frequency_mhz=req.frequency_mhz,
            radius_km=req.radius_km, resolution=req.res_px, epochs_ia=req.epochs_ia,
            city_slug=city_slug
        )
        
        if not trainer.load_weights():
            trainer.train(epochs=req.epochs_ia)
            trainer.save_weights()
        
        ai_power_dbm = trainer.infer_grid().numpy()
        ai_power_dbm = gaussian_filter(ai_power_dbm, sigma=1.0)
        
        # ---- 2. CONVERTIR A FIELD STRENGTH ----
        field_strength = ai_power_dbm + 20 * np.log10(req.frequency_mhz) + 77.2
        
        # ---- 3. CREAR GRID DE PUNTOS GEORREFERENCIADOS ----
        lats_1d = trainer.boundary.lats_1d
        lons_1d = trainer.boundary.lons_1d
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        
        # Subsample para performance (cada 2 píxeles — 2500 puntos en vez de 10000)
        step = 2
        lat_sub = lat_grid[::step, ::step].flatten()
        lon_sub = lon_grid[::step, ::step].flatten()
        sig_sub = field_strength[::step, ::step].flatten()
        pwr_sub = ai_power_dbm[::step, ::step].flatten()
        
        points = [Point(x, y) for x, y in zip(lon_sub, lat_sub)]
        gdf_grid = gpd.GeoDataFrame({
            'signal_dbuvm': sig_sub,
            'power_dbm': pwr_sub,
            'geometry': points
        }, crs="EPSG:4326")
        
        # ---- 4. SPATIAL JOIN CON LOCALIDADES ----
        loc_path = Path("data/localidades_bogota.geojson")
        if not loc_path.exists():
            raise HTTPException(status_code=404, detail="localidades_bogota.geojson not found")
        
        localidades = gpd.read_file(loc_path)
        if localidades.crs is None:
            localidades.set_crs(epsg=4326, inplace=True)
        else:
            localidades = localidades.to_crs(epsg=4326)
        
        grid_with_loc = gpd.sjoin(gdf_grid, localidades[['LocNombre', 'geometry']], 
                                   how="inner", predicate="within")
        
        # ---- 5. CARGAR POBLACIÓN 2026 ----
        pob_path = Path("data/poblacion_bogota_localidad.csv")
        
        def normalize_name(name):
            if not isinstance(name, str): return ""
            name = name.upper().strip()
            name = "".join(c for c in unicodedata.normalize('NFD', name) 
                          if unicodedata.category(c) != 'Mn')
            return name
        
        pop_by_loc = {}
        if pob_path.exists():
            df_pob = pd.read_csv(pob_path, sep=';')
            df_2026 = df_pob[df_pob['ANO'] == 2026].groupby('NOMBRE_LOCALIDAD')['POBLACION'].sum().reset_index()
            for _, row in df_2026.iterrows():
                pop_by_loc[normalize_name(row['NOMBRE_LOCALIDAD'])] = int(row['POBLACION'])
        
        # ---- 6. CALCULAR MÉTRICAS POR LOCALIDAD ----
        threshold_good = 66.0   # dBuV/m — cobertura de calidad
        threshold_min = 54.0    # dBuV/m — cobertura mínima
        
        stats = []
        total_pop_covered = 0
        total_pop = 0
        
        for loc_name in localidades['LocNombre'].unique():
            loc_pts = grid_with_loc[grid_with_loc['LocNombre'] == loc_name]
            if len(loc_pts) == 0:
                continue
            
            n_total = len(loc_pts)
            n_good = int((loc_pts['signal_dbuvm'] >= threshold_good).sum())
            n_min = int((loc_pts['signal_dbuvm'] >= threshold_min).sum())
            
            pct_good = (n_good / n_total) * 100.0
            pct_min = (n_min / n_total) * 100.0
            max_sig = float(loc_pts['signal_dbuvm'].max())
            mean_sig = float(loc_pts['signal_dbuvm'].mean())
            
            norm = normalize_name(loc_name)
            population = pop_by_loc.get(norm, 0)
            pop_covered = int(population * pct_good / 100.0)
            
            total_pop += population
            total_pop_covered += pop_covered
            
            stats.append({
                "localidad": loc_name,
                "cobertura_pct": round(pct_good, 1),
                "cobertura_min_pct": round(pct_min, 1),
                "max_signal": round(max_sig, 1),
                "mean_signal": round(mean_sig, 1),
                "poblacion": population,
                "hab_cubiertos": pop_covered,
                "n_puntos": n_total
            })
        
        # Sort by hab_cubiertos descending
        stats.sort(key=lambda x: x['hab_cubiertos'], reverse=True)
        
        elapsed = time.time() - start_time
        
        # ---- 7. RESUMEN GLOBAL ----
        area_covered_km2 = float(np.sum(field_strength >= threshold_good) * 
                                  (2 * req.radius_km / req.res_px) ** 2)
        
        return {
            "stats_by_localidad": stats,
            "summary": {
                "total_localidades": len(stats),
                "total_poblacion": total_pop,
                "total_hab_cubiertos": total_pop_covered,
                "pct_poblacion_cubierta": round(total_pop_covered / max(total_pop, 1) * 100, 1),
                "area_cobertura_km2": round(area_covered_km2, 1),
                "threshold_dbuvm": threshold_good,
                "elapsed_sec": round(elapsed, 1)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/download")
def download_report():
    report_path = Path("docs/REPORTE_TECNICO_ANE_1457.pdf")
    if not report_path.exists(): raise HTTPException(status_code=404, detail="Report not generated.")
    return FileResponse(path=report_path, filename="Reporte_Espectro_IA.pdf", media_type="application/pdf")
