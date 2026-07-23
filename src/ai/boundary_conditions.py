"""
Configuración de Condiciones de Frontera (Boundary Conditions) para PINNs.

Extrae la geometría del terreno (SRTM) y la ubicación del transmisor
para formular las funciones de pérdida estructurales de la red neuronal.

Arquitectura Residual: Los targets de frontera ahora son factores de corrección
en lugar de campos eléctricos directos.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from src.parsers.srtm_reader import SRTMReader
from src.propagation.fspl import fspl_db

class PINNBoundary:
    """
    Gestiona las condiciones espacio-temporales para el entrenamiento
    del modelo electromagnético Physics-Informed Neural Network.
    """
    def __init__(self, tx_lat: float, tx_lon: float, tx_height_m: float, 
                 frequency_mhz: float = 98.0,
                 tx_azimuth_deg: float = 0.0, tx_tilt_deg: float = 0.0, 
                 tx_hpbw_h_deg: float = 65.0, tx_hpbw_v_deg: float = 65.0,
                 radius_km: float = 20.0, resolution_points: int = 100, device: str = "cpu",
                 city_slug: str = "bogota"):
        print(f"  [Boundary] Init PINNBoundary on {device} (City: {city_slug})")
        self.device = torch.device(device)
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        self.tx_height_m = tx_height_m
        self.frequency_mhz = frequency_mhz
        self.tx_azimuth_deg = tx_azimuth_deg
        self.tx_tilt_deg = tx_tilt_deg
        self.tx_hpbw_h_deg = tx_hpbw_h_deg
        self.tx_hpbw_v_deg = tx_hpbw_v_deg
        self.radius_km = radius_km
        self.resolution = resolution_points
        self.city_slug = city_slug
        self.use_shadow_mask = True # Siempre activa por defecto
        
        # Auditoría de Clutter
        self.clutter_file = ""
        self.n_buildings = 0
        self.max_building_height = 0.0
        
        print(f"  [Boundary] Instantiating SRTMReader...")
        self.reader = None # SRTMReader(output_dir="data/srtm")
        print(f"  [Boundary] SRTMReader Ready. Generating Domain...")
        
        # Generar Dominio Espacial
        self._generate_domain()
        print(f"  [Boundary] Domain Generated.")

    def _generate_domain(self):
        """
        Crea el mallado espacial (Collocation Points) donde la PINN evaluará
        la ecuación Eikonal.
        """
        # Aproximación muy rápida 1 grado latitud ~ 111 km.
        km_per_lat = 111.0
        km_per_lon = 111.0 * np.cos(np.radians(self.tx_lat))
        
        deg_radius_lat = self.radius_km / km_per_lat
        deg_radius_lon = self.radius_km / km_per_lon
        
        self.lats_1d = np.linspace(self.tx_lat - deg_radius_lat, self.tx_lat + deg_radius_lat, self.resolution)
        self.lons_1d = np.linspace(self.tx_lon - deg_radius_lon, self.tx_lon + deg_radius_lon, self.resolution)
        
        lon_grid, lat_grid = np.meshgrid(self.lons_1d, self.lats_1d)
        
        # Coordenadas relativas en km
        self.x_km = (lon_grid - self.tx_lon) * km_per_lon
        self.y_km = (lat_grid - self.tx_lat) * km_per_lat
        self.dist_km_flat = np.sqrt(self.x_km.flatten()**2 + self.y_km.flatten()**2)
        
        # Grid 2D [lon, lat] para compatibilidad con la extracción de SRTM
        self.coords_2d = np.stack([lon_grid, lat_grid], axis=-1)
        
        # ----- EXTRAER ALTURAS REALES SRTM -----
        print(f"  [Boundary] Extracting AWS Skadi SRTM topography...")
        import math
        from src.parsers.srtm_reader import SRTMReader, download_srtm_tile
        
        lat_min, lat_max = np.min(self.lats_1d), np.max(self.lats_1d)
        lon_min, lon_max = np.min(self.lons_1d), np.max(self.lons_1d)
        
        lat_start = math.floor(lat_min)
        lat_end = math.floor(lat_max)
        lon_start = math.floor(lon_min)
        lon_end = math.floor(lon_max)
        
        self.srtm_cache = {}
        for lat_deg in range(lat_start, lat_end + 1):
            for lon_deg in range(lon_start, lon_end + 1):
                try:
                    tile_path = download_srtm_tile(lat=lat_deg, lon=lon_deg)
                    reader = SRTMReader(tile_path)
                    _ = reader.data # Preheat
                    self.srtm_cache[(lat_deg, lon_deg)] = reader
                except Exception as e:
                    print(f"  [Boundary] Error loading tile {lat_deg}, {lon_deg}: {e}")
        
        print(f"  [Boundary] Tiles loaded in cache: {list(self.srtm_cache.keys())}")
                    
        z_m_list = []
        for lat, lon in zip(lat_grid.flatten(), lon_grid.flatten()):
            lat_idx = math.floor(lat)
            lon_idx = math.floor(lon)
            if (lat_idx, lon_idx) in self.srtm_cache:
                z = self.srtm_cache[(lat_idx, lon_idx)].get_elevation(lat, lon)
                if np.isnan(z):
                    z_m_list.append(0.0) # Voids a 0m
                else:
                    z_m_list.append(z)
            else:
                z_m_list.append(0.0) # Altura base si falla tile
                
        self.z_m = np.array(z_m_list).reshape(lat_grid.shape)
        
        # --- NUEVO: Suavizado Gaussiano para eliminar micro-ruido topográfico ---
        # sigma=1.5 equivale a un radio de ~150-300m de suavizado, eliminando 
        # variaciones de ±200m que confunden a la Eikonal en FM.
        from scipy.ndimage import gaussian_filter
        self.z_m_smooth = gaussian_filter(self.z_m, sigma=1.5)
        
        # --- NUEVO: Corrección de Curvatura Terrestre (Factor k=4/3) ---
        # R_efectivo = 8500 km. El terreno 'sube' proporcional al cuadrado de la distancia.
        # h_curv_m = d^2 / (2 * R_ef) -> d en km, R_ef en miles de km (8.5)
        dist_km_2d = np.sqrt(self.x_km**2 + self.y_km**2)
        h_curv_m = (dist_km_2d ** 2) / (2 * 8.5)
        self.z_m_smooth = self.z_m_smooth + h_curv_m
        
        # Escalar Z a Kilómetros para que los gradientes espaciales tengan misma escala numérica
        self.z_km = self.z_m_smooth / 1000.0
        
        # Pre-computar rugosidad topográfica local (σ del DEM en ventana 9×9)
        from scipy.ndimage import generic_filter
        self.z_roughness_km = generic_filter(self.z_km, np.std, size=9, mode='nearest')
        
        # Clip: σ máxima = 0.05 km (50m). Restringimos para evitar sobre-atenuación local.
        self.z_roughness_km = np.clip(self.z_roughness_km, 0.0, 0.05)
        self.z_roughness_flat = self.z_roughness_km.flatten()

        # --- NUEVO: Clutter Urbano Real (Basado en OSM/Dataset dinámico) ---
        # Usar ruta absoluta para evitar problemas en Windows
        project_root = Path(__file__).parent.parent.parent
        osm_path = project_root / "data" / f"osm_buildings_{self.city_slug}.csv"
        self.urban_clutter = np.zeros_like(lat_grid)
        
        if osm_path.exists():
            import pandas as pd
            from scipy.interpolate import griddata
            print(f"  [Boundary] Cargando Clutter real desde {osm_path}...")
            osm_df = pd.read_csv(osm_path)
            
            # Interpolamos las alturas de los edificios sobre nuestro grid actual
            # Usamos 'linear' con fill_value=0 para zonas sin edificios registrados
            points = osm_df[['lon', 'lat']].values
            values = osm_df['height'].values
            
            # Rasterizamos el clutter sobre el grid del simulador
            self.urban_clutter = griddata(points, values, (lon_grid, lat_grid), method='linear', fill_value=0.0)
            
            # Aplicamos un suavizado ligero para que la PINN no sufra con gradientes infinitos en bordes de edificios
            from scipy.ndimage import gaussian_filter
            self.urban_clutter = gaussian_filter(self.urban_clutter, sigma=1.0)
            
            # Auditoría
            self.clutter_file = str(osm_path)
            self.n_buildings = len(osm_df)
            self.max_building_height = float(np.max(self.urban_clutter))
            
            # Normalización para slowness: 0m = 0.0, 150m = 1.0
            self.urban_clutter_norm = np.clip(self.urban_clutter / 150.0, 0.0, 1.0)
            print(f"  [Boundary] Clutter Real Integrado. Edificios: {self.n_buildings}, Altura máx: {self.max_building_height:.1f}m")
        else:
            self.clutter_file = "None"
            self.n_buildings = 0
            self.max_building_height = 0.0
            print(f"  [Boundary] Warning: No se encontró {osm_path}. Clutter desactivado.")
            self.urban_clutter_norm = np.zeros_like(lat_grid)

        self.urban_clutter_flat = self.urban_clutter_norm.flatten()
        
        print(f"  [Boundary] Topografía Suavizada (Sigma 1.5). Z Range: {np.min(self.z_m_smooth):.1f}m to {np.max(self.z_m_smooth):.1f}m")
        print(f"  [Boundary] Rugosidad σ (9×9, clip 300m): mean={np.mean(self.z_roughness_km)*1000:.1f}m")
              
        # Flatten para PyTorch
        self.x_tensor = torch.tensor(self.x_km.flatten(), dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(self.y_km.flatten(), dtype=torch.float32, device=self.device)
        self.z_tensor = torch.tensor(self.z_km.flatten(), dtype=torch.float32, device=self.device)
        
        self.domain_coords = torch.stack([self.x_tensor, self.y_tensor, self.z_tensor], dim=1)
        self.domain_coords.requires_grad_(True)

    def get_terrain_boundary(self) -> torch.Tensor:
        """
        Frontera de Dirichlet (Terreno)
        Retorna las coordenadas (x, y, z) espaciales que conforman la "pared" física
        de las montañas extraídas del DEM SRTM.
        """
        # Extraer array (N, 1) de la cuadrícula de relieve calculada en km
        x_flat = self.x_km.flatten()
        y_flat = self.y_km.flatten()
        z_terrain_km = self.z_km.flatten()
        
        # Matriz final de frontera [x_km, y_km, z_terreno_km] (N, 3)
        terrain_points = np.stack([x_flat, y_flat, z_terrain_km], axis=-1)
        return torch.tensor(terrain_points, dtype=torch.float32, device=self.device)

    def get_fspl_prior_grid(self, tx_power_dbm: float) -> np.ndarray:
        """
        Calcula el campo eléctrico FSPL analítico (V/m) para cada punto del grid 2D.
        Este es el "prior" que se multiplica por la corrección de la PINN.
        
        Returns:
            Array 2D (resolution × resolution) con E_field en V/m según FSPL.
        """
        # Distancia de cada punto al TX en km
        dist_km = np.sqrt(self.x_km**2 + self.y_km**2)
        dist_km = np.maximum(dist_km, 0.001)  # Evitar log(0) en el centro
        
        # FSPL en dB para cada punto
        from src.propagation.fspl import fspl_db as _fspl_db
        path_loss_db = _fspl_db(dist_km, self.frequency_mhz)
        
        # Potencia recibida en dBm
        rx_power_dbm = tx_power_dbm + 3.0 - path_loss_db  # +3 dBi antena
        
        # Convertir dBm a V/m (campo eléctrico)
        power_w = (10 ** (rx_power_dbm / 10)) / 1000.0
        power_w = np.maximum(power_w, 1e-25)
        e_field = np.sqrt(30 * power_w)  # P = E²/(120π), E = sqrt(30*P)
        
        return e_field

    def get_source_boundary(self, tx_power_dbm: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Frontera de Fuente Direccional (Arquitectura Residual).
        
        En la arquitectura residual, la red predice un FACTOR DE CORRECCIÓN.
        En los puntos de la fuente (la esfera de radiación), el factor de corrección
        debe ser ≈ 1.0 en el haz principal y decrecer según el patrón de antena.
        
        El patrón de antena real se codifica como el target de corrección,
        permitiendo a la PINN aprender la directividad.
        """
        import math
        lat_idx = math.floor(self.tx_lat)
        lon_idx = math.floor(self.tx_lon)
        
        base_elevation_m = 0.0
        if (lat_idx, lon_idx) in self.srtm_cache:
            z = self.srtm_cache[(lat_idx, lon_idx)].get_elevation(self.tx_lat, self.tx_lon)
            if not np.isnan(z):
                base_elevation_m = z
        
        total_z_km = (base_elevation_m + self.tx_height_m) / 1000.0
        
        # Esfera de Radiación
        num_theta = 18
        num_phi = 36
        sphere_radius_km = 0.01  # 10 metros
        
        coords_tx = []
        target_correction = []
        
        for i in range(num_theta):
            theta_deg = -90.0 + (180.0 / (num_theta - 1)) * i
            theta_rad = np.radians(theta_deg)
            for j in range(num_phi):
                phi_deg = (360.0 / num_phi) * j
                phi_rad = np.radians(phi_deg)
                
                x_point = sphere_radius_km * np.sin(phi_rad) * np.cos(theta_rad)
                y_point = sphere_radius_km * np.cos(phi_rad) * np.cos(theta_rad)
                z_point = total_z_km + (sphere_radius_km * np.sin(theta_rad))
                
                coords_tx.append([x_point, y_point, z_point])
                
                # Patrón de Antena ITU-R F.699
                diff_phi = ((phi_deg - self.tx_azimuth_deg + 180) % 360) - 180
                diff_theta = theta_deg - self.tx_tilt_deg
                
                att_h = 12 * (diff_phi / self.tx_hpbw_h_deg)**2
                att_v = 12 * (diff_theta / self.tx_hpbw_v_deg)**2
                total_att_db = min(att_h + att_v, 35.0)
                
                # Factor de corrección: 1.0 = haz principal, <1.0 = atenuación por patrón
                correction_factor = 10 ** (-total_att_db / 20.0)
                target_correction.append([correction_factor])
        
        return (
            torch.tensor(coords_tx, dtype=torch.float32, device=self.device),
            torch.tensor(target_correction, dtype=torch.float32, device=self.device)
        )

    def get_subterrain_collocation_points(self, num_points: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera puntos de colocalización DENTRO del volumen sólido de las montañas.
        
        Estos puntos fuerzan slowness extrema (s >> 1) en la Eikonal, impidiendo
        que el frente de onda atraviese la masa montañosa. El resultado es que la
        PINN aprende a difractar por encima de las cumbres, creando sombras reales.
        
        Returns:
            coords: (N, 3) tensor [x_km, y_km, z_km] — puntos dentro de la montaña
            slowness: (N, 1) tensor — factor de slowness (≈100 para roca sólida)
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Interpolador de elevación del terreno
        interp_z = RegularGridInterpolator(
            (self.y_km[:, 0], self.x_km[0, :]), self.z_km,
            bounds_error=False, fill_value=0.0
        )
        
        # Altura base del TX (para filtrar: solo montañas significativas)
        import math
        lat_idx = math.floor(self.tx_lat)
        lon_idx = math.floor(self.tx_lon)
        tx_base_km = 0.0
        if (lat_idx, lon_idx) in self.srtm_cache:
            z = self.srtm_cache[(lat_idx, lon_idx)].get_elevation(self.tx_lat, self.tx_lon)
            if not np.isnan(z):
                tx_base_km = z / 1000.0
        
        # Umbral: solo generar puntos sub-terreno donde la montaña es >100m sobre el TX
        z_threshold_km = tx_base_km + 0.1  # 100 metros sobre la base del TX
        
        # Muestreo: distribución uniforme en (x, y) dentro del dominio
        x_rand = np.random.uniform(-self.radius_km, self.radius_km, num_points * 3)
        y_rand = np.random.uniform(-self.radius_km, self.radius_km, num_points * 3)
        
        # Filtrar solo puntos donde la montaña es significativa
        query_pts = np.stack([y_rand, x_rand], axis=1)
        z_terrain = interp_z(query_pts)
        
        # Máscara: solo donde el terreno está por encima del umbral
        mask = z_terrain > z_threshold_km
        x_valid = x_rand[mask]
        y_valid = y_rand[mask]
        z_surface = z_terrain[mask]
        
        if len(x_valid) == 0:
            # Sin montañas significativas — retornar tensor vacío
            empty_coords = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            empty_slow = torch.ones((1, 1), dtype=torch.float32, device=self.device)
            return empty_coords, empty_slow
        
        # Limitar al número deseado
        if len(x_valid) > num_points:
            idx = np.random.choice(len(x_valid), num_points, replace=False)
            x_valid = x_valid[idx]
            y_valid = y_valid[idx]
            z_surface = z_surface[idx]
        
        n = len(x_valid)
        
        # Distribución de profundidad: 60% en primeros 100m bajo superficie, 40% más profundo
        is_shallow = np.random.rand(n) < 0.6
        depth_km = np.zeros(n)
        # Shallow: 0 a 0.1 km (0-100m) bajo la superficie
        depth_km[is_shallow] = np.random.uniform(0.001, 0.1, np.sum(is_shallow))
        # Deep: 0.1 a 0.5 km (100-500m) bajo la superficie
        depth_km[~is_shallow] = np.random.uniform(0.1, 0.5, np.sum(~is_shallow))
        
        # Z final: superficie - profundidad (pero no debajo de 0)
        z_sub = np.maximum(z_surface - depth_km, 0.001)
        
        coords = np.stack([x_valid, y_valid, z_sub], axis=1)
        
        # Slowness elevada: s = 10 (la señal viaja 10x más lento en roca)
        # Esto genera ~20 dB de atenuación por km de penetración
        # Un valor de 100 era demasiado agresivo — contaminaba el campo T en el aire.
        slowness = np.full((n, 1), 10.0)
        
        return (
            torch.tensor(coords, dtype=torch.float32, device=self.device),
            torch.tensor(slowness, dtype=torch.float32, device=self.device)
        )

    def get_collocation_points(self, num_points: int = 10000, max_radius_km: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling adaptativo del volumen aéreo (Importance Sampling).
        Concentra más puntos cerca del TX para resolver mejor la singularidad del gradiente.
        """
        if max_radius_km is None:
            max_radius_km = self.radius_km

        # 1. Muestreo Adaptativo de Radios (Distribución Log-Uniforme)
        # Esto pone muchos más puntos cerca de r=0 que una distribución uniforme.
        # r_min = 10m, r_max = max_radius_km
        r_min = 0.01
        r_rand = r_min * (max_radius_km / r_min) ** np.random.uniform(0, 1, num_points)
        theta_rand = np.random.uniform(0, 2 * np.pi, num_points)
        
        x_rand = r_rand * np.cos(theta_rand)
        y_rand = r_rand * np.sin(theta_rand)
        
        # 2. Interpolación de Altura de Terreno para los puntos aleatorios
        from scipy.interpolate import RegularGridInterpolator
        # lons_1d y lats_1d están en orden ascendente (linspace)
        interp_z = RegularGridInterpolator((self.y_km[:, 0], self.x_km[0, :]), self.z_km, bounds_error=False, fill_value=0.0)
        interp_rough = RegularGridInterpolator((self.y_km[:, 0], self.x_km[0, :]), self.z_roughness_km, bounds_error=False, fill_value=0.0)
        interp_clutter = RegularGridInterpolator((self.y_km[:, 0], self.x_km[0, :]), self.urban_clutter_norm, bounds_error=False, fill_value=0.0)
        
        query_pts = np.stack([y_rand, x_rand], axis=1)
        z_base_km = interp_z(query_pts)
        roughness_km = interp_rough(query_pts)
        clutter_factor = interp_clutter(query_pts)
        
        # 3. Elevación aleatoria sobre el terreno (también sesgada hacia abajo para captar difracción)
        # 70% de puntos abajo (< 500m), 30% arriba (hasta 3km)
        is_low = np.random.rand(num_points) < 0.7
        z_offset_km = np.zeros(num_points)
        z_offset_km[is_low] = np.random.uniform(0.001, 0.5, np.sum(is_low))
        z_offset_km[~is_low] = np.random.uniform(0.5, 3.0, np.sum(~is_low))
        
        z_rand = z_base_km + z_offset_km
        
        colpoints = np.stack([x_rand, y_rand, z_rand], axis=1)
        return (
            torch.tensor(colpoints, dtype=torch.float32, device=self.device),
            torch.tensor(z_base_km, dtype=torch.float32, device=self.device).view(-1, 1).to(torch.float32),
            torch.tensor(roughness_km, dtype=torch.float32, device=self.device).view(-1, 1).to(torch.float32),
            torch.tensor(clutter_factor, dtype=torch.float32, device=self.device).view(-1, 1).to(torch.float32)
        )

def boundary_loss(T_terrain: torch.Tensor, T_source: torch.Tensor, 
                  correction_target_source: torch.Tensor,
                  z_terrain_km: torch.Tensor = None,
                  tx_z_km: float = 0.0) -> torch.Tensor:
    """
    Función de pérdida estructural para la arquitectura de Fase (Eikonal).
    """
    # Terreno: Penalización por penetración física.
    # En el Eikonal, T >= d_geométrica. Si T < d, el rayo está "atajando" por dentro del terreno.
    # Esto crea sombras de difracción naturales.
    C_terrain = torch.exp(-T_terrain)
    # Penalizamos masivamente la existencia de señal (C_terrain > 0) dentro de la masa montañosa
    loss_terrain = torch.mean(C_terrain**2)
    
    # Condición Eikonal estricta: T(source) = 0
    loss_source = torch.mean(T_source**2)
    
    return loss_terrain, loss_source
