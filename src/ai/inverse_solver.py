import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Dict
from src.parsers.srtm_reader import SRTMReader, download_srtm_tile
from src.propagation.fspl import fspl_db

class InverseSolver:
    def __init__(self, frequency_mhz: float, search_center: Dict[str, float], radius_km: float = 20.0):
        self.frequency_mhz = frequency_mhz
        self.center = search_center
        self.radius_km = radius_km
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-cargar topografía de la zona de búsqueda
        self._load_topography()

    def _load_topography(self):
        """Carga los tiles de elevación necesarios para la zona de búsqueda (soporte multi-tile)."""
        lat, lon = self.center['lat'], self.center['lon']
        
        res = 200 # Grid de 200x200
        km_per_lat = 111.0
        km_per_lon = 111.0 * math.cos(math.radians(lat))
        
        deg_lat = self.radius_km / km_per_lat
        deg_lon = self.radius_km / km_per_lon
        
        self.lat_min, self.lat_max = lat - deg_lat, lat + deg_lat
        self.lon_min, self.lon_max = lon - deg_lon, lon + deg_lon
        
        lats = np.linspace(self.lat_min, self.lat_max, res)
        lons = np.linspace(self.lon_min, self.lon_max, res)
        
        # 1. Identificar e instalar todos los tiles necesarios upfront (Doble Tile / Multi Tile)
        needed_tiles = {} # {(lat_floor, lon_floor): reader}
        lat_range = range(int(math.floor(self.lat_min)), int(math.floor(self.lat_max)) + 1)
        lon_range = range(int(math.floor(self.lon_min)), int(math.floor(self.lon_max)) + 1)
        
        for t_lat in lat_range:
            for t_lon in lon_range:
                tile_key = (t_lat, t_lon)
                try:
                    tile_path = download_srtm_tile(t_lat, t_lon)
                    needed_tiles[tile_key] = SRTMReader(tile_path)
                except Exception as e:
                    print(f"Advertencia: No se pudo cargar tile {tile_key}: {e}")
                    needed_tiles[tile_key] = None
        
        # 2. Poblar el grid con seguridad "fronteriza"
        z_grid = np.zeros((res, res))
        for i, lt in enumerate(lats):
            for j, ln in enumerate(lons):
                t_lat, t_lon = int(math.floor(lt)), int(math.floor(ln))
                tile_key = (t_lat, t_lon)
                
                reader = needed_tiles.get(tile_key)
                if reader:
                    try:
                        # Seguridad: Clamping para evitar errores por precisión de punto flotante en fronteras
                        info = reader.info
                        safe_lt = max(info.lat_sw, min(info.lat_ne, lt))
                        safe_ln = max(info.lon_sw, min(info.lon_ne, ln))
                        
                        z = reader.get_elevation(safe_lt, safe_ln)
                        z_grid[i, j] = z if not np.isnan(z) else 2600.0
                    except:
                        z_grid[i, j] = 2600.0 # Fallback Bogotá
                else:
                    z_grid[i, j] = 2600.0
                    
        self.z_tensor = torch.tensor(z_grid, dtype=torch.float32, device=self.device)

    def solve(self, measurements: List[Dict[str, float]], iterations: int = 200) -> Dict:
        """
        Encuentra (lat, lon, power) que minimiza el error con las mediciones considerando topografía.
        measurements: list of {lat, lon, dbm}
        """
        import torch.nn.functional as F
        
        # grid_sample espera (N, H, W, 2) donde el último es (x, y) -> (lon, lat)
        grid = self.z_tensor.view(1, 1, self.z_tensor.shape[0], self.z_tensor.shape[1])
        # Coordenadas de medición (estáticas)
        m_lats = torch.tensor([m['lat'] for m in measurements], device=self.device)
        m_lons = torch.tensor([m['lon'] for m in measurements], device=self.device)
        m_dbm = torch.tensor([m['dbm'] for m in measurements], device=self.device)
        m_uncertainties = torch.tensor([m.get('uncertainty_db', 2.0) for m in measurements], dtype=torch.float32, device=self.device)
        
        # Pesos WLS rigurosos: inverso de la varianza (σ^2)
        weights = 1.0 / (m_uncertainties ** 2 + 1e-5)
        
        # Pre-calcular elevación de los receptores (interpolar en el grid cargado)
        m_lats_norm = (m_lats - self.lat_min) / (self.lat_max - self.lat_min) * 2 - 1
        m_lons_norm = (m_lons - self.lon_min) / (self.lon_max - self.lon_min) * 2 - 1
        rx_coords = torch.stack([m_lons_norm, m_lats_norm], dim=-1).view(1, 1, -1, 2)
        m_z_terrain = F.grid_sample(grid, rx_coords, align_corners=True).view(-1)
        m_z_abs = m_z_terrain + 2.0 

        # --- NUEVO: Grid Search Inicial para evitar mínimos locales ---
        with torch.no_grad():
            grid_size = 100  # Mayor resolución para no saltarse la cima
            gx = torch.linspace(0.01, 0.99, grid_size, device=self.device)
            gy = torch.linspace(0.01, 0.99, grid_size, device=self.device)
            grid_lats, grid_lons = torch.meshgrid(gx, gy, indexing='ij')
            candidates = torch.stack([grid_lats.flatten(), grid_lons.flatten()], dim=-1) # [400, 2]
            
            # Evaluar todos los candidatos de forma simplificada
            c_lats = self.lat_min + candidates[:, 0].unsqueeze(1) * (self.lat_max - self.lat_min)
            c_lons = self.lon_min + candidates[:, 1].unsqueeze(1) * (self.lon_max - self.lon_min)
            
            # Elevación de candidatos
            c_tx_coords = torch.stack([candidates[:, 1] * 2 - 1, candidates[:, 0] * 2 - 1], dim=-1).view(1, 1, -1, 2)
            c_z_terrain = F.grid_sample(grid, c_tx_coords, align_corners=True).view(-1, 1)
            c_z_abs = c_z_terrain + 50.0
            
            # Distancias a mediciones [400, N_m]
            dx = (m_lons.unsqueeze(0) - c_lons) * (111.0 * math.cos(math.radians(self.center['lat'])))
            dy = (m_lats.unsqueeze(0) - c_lats) * 111.0
            dz = (c_z_abs - m_z_abs.unsqueeze(0)) / 1000.0
            d3d = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
            
            # Pre-evaluación de obstrucción para el grid search (simplificada)
            # nu_c: [400, N_m]
            # (En el grid search omitimos el muestreo de perfil detallado por velocidad, 
            # pero usamos la altura del TX para favorecer cerros)
            
            # Error cuadrático medio INVARIANTE AL BIAS (Busca la forma, no el valor)
            c_fspl = 20 * torch.log10(d3d) + 20 * math.log10(self.frequency_mhz) + 32.44
            c_pred_base = 50.0 - c_fspl - 12.0 - 10.0
            
            # Calculamos el bias necesario para cada candidato minimizando la pérdida ponderada (WLS)
            c_diff = m_dbm.unsqueeze(0) - c_pred_base # [400, N_m]
            c_bias_needed = torch.sum(weights.unsqueeze(0) * c_diff, dim=1, keepdim=True) / torch.sum(weights)
            
            # La pérdida es el error residual ponderado WLS
            c_loss = torch.sum(weights.unsqueeze(0) * (c_pred_base + c_bias_needed - m_dbm.unsqueeze(0))**2, dim=1) / torch.sum(weights)
            
            best_idx = torch.argmin(c_loss)
            best_start_pos = candidates[best_idx]
            initial_bias = c_bias_needed[best_idx].item()
            print(f"Grid Search (Bias-Invariant WLS): Mejor inicio en {best_start_pos.cpu().numpy()} con Bias {initial_bias:.2f} y RMSE ponderado {torch.sqrt(c_loss[best_idx]):.2f}")

        # 1. Parámetros optimizables iniciados en la mejor zona geométrica
        tx_pos = torch.tensor(best_start_pos.tolist(), dtype=torch.float32, device=self.device, requires_grad=True)
        tx_power = torch.tensor([50.0], dtype=torch.float32, device=self.device, requires_grad=True) # dBm
        # BIAS: Iniciado con el valor detectado en el grid search
        bias = torch.tensor([initial_bias], dtype=torch.float32, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([tx_pos, tx_power, bias], lr=0.002)

        
        history = []
        
        for step in range(iterations):
            optimizer.zero_grad()
            
            # Denormalizar posición TX
            curr_lat = self.lat_min + tx_pos[0] * (self.lat_max - self.lat_min)
            curr_lon = self.lon_min + tx_pos[1] * (self.lon_max - self.lon_min)
            
            # Obtener elevación del TX candidato (DIFERENCIABLE)
            # Mapear tx_pos a [-1, 1] y swap a (x, y) -> (lon, lat)
            tx_coord_norm = torch.stack([tx_pos[1] * 2 - 1, tx_pos[0] * 2 - 1]).view(1, 1, 1, 2)
            tx_z_terrain = F.grid_sample(grid, tx_coord_norm, align_corners=True).view(1)
            tx_z_abs = tx_z_terrain + 50.0 # Calibración: Antena de 50m para cerros
            
            # Calcular distancias a cada punto de medición (Haversine simplificado)
            km_per_lat = 111.0
            km_per_lon = 111.0 * torch.cos(torch.deg2rad(curr_lat))
            
            dx = (m_lons - curr_lon) * km_per_lon
            dy = (m_lats - curr_lat) * km_per_lat
            
            # Distancia 3D considerando elevación (en km)
            dz = (tx_z_abs - m_z_abs) / 1000.0
            dist_3d_km = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
            
            # 4. Modelo FSPL 3D
            fspl = 20 * torch.log10(dist_3d_km) + 20 * math.log10(self.frequency_mhz) + 32.44
            
            # 5. Difracción Knife-Edge Diferenciable (Muestreo de Perfil)
            n_samples = 15
            t = torch.linspace(0.1, 0.9, n_samples, device=self.device).view(1, 1, n_samples, 1)

            
            # Paths normalizados entre TX y cada receptor [1, N_m, n_samples, 2]
            tx_pos_norm = (tx_pos * 2 - 1).view(1, 1, 1, 2)
            m_pos_norm = torch.stack([m_lats_norm, m_lons_norm], dim=-1).view(1, -1, 1, 2)
            paths = tx_pos_norm + t * (m_pos_norm - tx_pos_norm)

            
            # Muestrear elevaciones a lo largo de cada trayecto
            paths_grid = torch.stack([paths[..., 1], paths[..., 0]], dim=-1)
            path_z = F.grid_sample(grid, paths_grid, align_corners=True).view(-1, n_samples)
            
            # Altura de la línea de vista (LOS) en cada punto de muestra
            los_z = tx_z_abs + t.view(1, n_samples) * (m_z_abs.view(-1, 1) - tx_z_abs)
            
            # Obstrucción máxima (h)
            h_max = torch.max(path_z - los_z, dim=1)[0]
            
            # Parámetro de Fresnel nu (Aproximación para difracción)
            wavelength = 300.0 / self.frequency_mhz
            # nu = h * sqrt( (2/lambda) * (d1+d2)/(d1*d2) ) -> simplificado asumiendo d1=d2=dist/2
            nu = h_max * torch.sqrt((2.0 / wavelength) * (4.0 / (dist_3d_km * 1000.0 + 1.0)))
            
            # Pérdida por Difracción (Aproximación continua y diferenciable de Knife-Edge según ITU-R P.526)
            # Evita discontinuidades de 6dB que actúan como barreras de gradiente artificiales en las laderas.
            ke_loss_val = 6.9 + 20.0 * torch.log10(torch.sqrt((nu - 0.1)**2 + 1.0) + nu - 0.1)
            ke_loss = torch.where(nu > -0.78, ke_loss_val, torch.zeros_like(nu))
            
            # 6. Predicción Final (FSPL + Difracción + Clutter + ANE Offset + BIAS)
            clutter_loss = 12.0  # Urban clutter Bogota
            ane_offset = -10.0   # Ajuste regulatorio ANE
            
            # El bias permite que el modelo se ajuste a la ESCALA de tus datos 
            # sin importar si son dBm, dBuV o si tienen pre-amplificación.
            pred_dbm = tx_power - fspl - ke_loss - clutter_loss + ane_offset + bias
            
            # Función de pérdida ponderada (WLS)
            loss = torch.sum(weights * (pred_dbm - m_dbm)**2) / torch.sum(weights)
            
            # Regularización: evitar salir del radio de búsqueda
            reg = torch.mean(torch.relu(torch.abs(tx_pos - 0.5) - 0.5)**2)
            total_loss = loss + 100.0 * reg
            
            total_loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                history.append({
                    "step": step,
                    "lat": float(curr_lat),
                    "lon": float(curr_lon),
                    "power": float(tx_power),
                    "loss": float(loss),
                    "tx_z": float(tx_z_abs),
                    "bias": float(bias)
                })
        
        final_lat = self.lat_min + tx_pos[0] * (self.lat_max - self.lat_min)
        final_lon = self.lon_min + tx_pos[1] * (self.lon_max - self.lon_min)
        
        # Calcular rejilla de probabilidades vectorizada
        prob_matrix = self.compute_probability_grid(
            tx_power=float(tx_power.item()),
            bias=float(bias.item()),
            measurements=measurements,
            grid_res=80
        )
        
        # Generar base64
        heatmap_base64 = self.generate_heatmap_base64(prob_matrix)
        
        return {
            "detected_tx": {
                "lat": float(final_lat),
                "lon": float(final_lon),
                "power_dbm": float(tx_power.item()),
                "altitude_m": float(tx_z_abs.item()),
                "calibrated_bias_db": float(bias.item()),
                "heatmap_base64": heatmap_base64,
                "bounds": [
                    [float(self.lat_min), float(self.lon_min)],
                    [float(self.lat_max), float(self.lon_max)]
                ]
            },
            "error_rmse": float(torch.sqrt(loss).item()),
            "history": history
        }

    def compute_probability_grid(self, tx_power: float, bias: float, measurements: List[Dict[str, float]], grid_res: int = 80) -> np.ndarray:
        """
        Calcula la probabilidad (Boltzmann posterior) de que el TX esté en cada celda de un grid de grid_res x grid_res.
        Usa el modelo de física completo (FSPL + Knife-Edge + Clutter) vectorizado en PyTorch.
        """
        import torch.nn.functional as F
        
        grid = self.z_tensor.view(1, 1, self.z_tensor.shape[0], self.z_tensor.shape[1])
        m_lats = torch.tensor([m['lat'] for m in measurements], device=self.device)
        m_lons = torch.tensor([m['lon'] for m in measurements], device=self.device)
        m_dbm = torch.tensor([m['dbm'] for m in measurements], device=self.device)
        m_uncertainties = torch.tensor([m.get('uncertainty_db', 2.0) for m in measurements], dtype=torch.float32, device=self.device)
        weights = 1.0 / (m_uncertainties ** 2 + 1e-5)
        
        # Pre-calcular coordenadas de medición normales
        m_lats_norm = (m_lats - self.lat_min) / (self.lat_max - self.lat_min) * 2 - 1
        m_lons_norm = (m_lons - self.lon_min) / (self.lon_max - self.lon_min) * 2 - 1
        rx_coords = torch.stack([m_lons_norm, m_lats_norm], dim=-1).view(1, 1, -1, 2)
        m_z_terrain = F.grid_sample(grid, rx_coords, align_corners=True).view(-1)
        m_z_abs = m_z_terrain + 2.0
        
        with torch.no_grad():
            gx = torch.linspace(0.0, 1.0, grid_res, device=self.device)
            gy = torch.linspace(0.0, 1.0, grid_res, device=self.device)
            grid_lats, grid_lons = torch.meshgrid(gx, gy, indexing='ij')
            candidates = torch.stack([grid_lats.flatten(), grid_lons.flatten()], dim=-1) # [grid_res^2, 2]
            
            c_lats = self.lat_min + candidates[:, 0].unsqueeze(1) * (self.lat_max - self.lat_min)
            c_lons = self.lon_min + candidates[:, 1].unsqueeze(1) * (self.lon_max - self.lon_min)
            
            # Elevación de candidatos [grid_res^2, 1]
            c_tx_coords = torch.stack([candidates[:, 1] * 2 - 1, candidates[:, 0] * 2 - 1], dim=-1).view(1, 1, -1, 2)
            c_z_terrain = F.grid_sample(grid, c_tx_coords, align_corners=True).view(-1, 1)
            c_z_abs = c_z_terrain + 50.0 # Antena de 50m
            
            # Distancias a mediciones [grid_res^2, N_m]
            dx = (m_lons.unsqueeze(0) - c_lons) * (111.0 * math.cos(math.radians(self.center['lat'])))
            dy = (m_lats.unsqueeze(0) - c_lats) * 111.0
            dz = (c_z_abs - m_z_abs.unsqueeze(0)) / 1000.0
            dist_3d_km = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
            
            # FSPL
            fspl = 20 * torch.log10(dist_3d_km) + 20 * math.log10(self.frequency_mhz) + 32.44
            
            # Knife-Edge vectorizado para todo el grid
            n_samples = 15
            t = torch.linspace(0.1, 0.9, n_samples, device=self.device).view(1, 1, n_samples, 1)
            
            tx_pos_norm = (candidates * 2 - 1).view(-1, 1, 1, 2) # [grid_res^2, 1, 1, 2]
            m_pos_norm = torch.stack([m_lats_norm, m_lons_norm], dim=-1).view(1, -1, 1, 2) # [1, N_m, 1, 2]
            paths = tx_pos_norm + t * (m_pos_norm - tx_pos_norm) # [grid_res^2, N_m, n_samples, 2]
            
            # Muestrear
            paths_grid = torch.stack([paths[..., 1], paths[..., 0]], dim=-1)
            flat_queries = paths_grid.view(1, -1, n_samples, 2)
            flat_sampled_z = F.grid_sample(grid, flat_queries, align_corners=True).view(grid_res**2, len(measurements), n_samples)
            
            los_z = c_z_abs.view(-1, 1, 1) + t.view(1, 1, -1) * (m_z_abs.view(1, -1, 1) - c_z_abs.view(-1, 1, 1))
            
            h_max = torch.max(flat_sampled_z - los_z, dim=2)[0]
            
            wavelength = 300.0 / self.frequency_mhz
            nu = h_max * torch.sqrt((2.0 / wavelength) * (4.0 / (dist_3d_km * 1000.0 + 1.0)))
            
            ke_loss_val = 6.9 + 20.0 * torch.log10(torch.sqrt((nu - 0.1)**2 + 1.0) + nu - 0.1)
            ke_loss = torch.where(nu > -0.78, ke_loss_val, torch.zeros_like(nu))
            
            # Predicción final
            clutter_loss = 12.0
            ane_offset = -10.0
            pred_dbm = tx_power - fspl - ke_loss - clutter_loss + ane_offset + bias
            
            # RMSE ponderado para cada punto del grid (WLS)
            weighted_var = torch.sum(weights.unsqueeze(0) * (pred_dbm - m_dbm.unsqueeze(0))**2, dim=1) / torch.sum(weights)
            rmse = torch.sqrt(weighted_var) # [grid_res^2]
            
            # Probabilidad usando distribución Boltzmann posterior (sigma = 3.0 dB)
            sigma = 3.0
            prob = torch.exp(-(rmse**2) / (2.0 * sigma**2))
            
            max_prob = torch.max(prob)
            if max_prob > 0:
                prob = prob / max_prob
                
            prob_matrix = prob.view(grid_res, grid_res).cpu().numpy()
            
            return prob_matrix

    def generate_heatmap_base64(self, prob_matrix: np.ndarray) -> str:
        """
        Convierte una matriz de probabilidad en una imagen PNG base64 con un gradiente
        cálido (Rojo/Naranja/Amarillo) translúcido y de alta definición.
        """
        import PIL.Image
        import io
        import base64
        
        # Mapear de forma no lineal para resaltar el epicentro y limpiar el ruido
        P = np.clip(prob_matrix, 0.0, 1.0)
        
        h, w = P.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        
        # Canal Rojo: Siempre alto para el rango cálido
        rgba[..., 0] = 1.0
        
        # Canal Verde: Transición para formar amarillo y naranja
        rgba[..., 1] = np.where(P < 0.2, 0.2 + 0.3 * (P / 0.2),
                                np.where(P < 0.6, 0.5 - 0.2 * ((P - 0.2) / 0.4),
                                         0.3 - 0.3 * ((P - 0.6) / 0.4)))
        
        # Canal Azul: 0.0
        rgba[..., 2] = 0.0
        
        # Canal Alfa (Opacidad): Escalado no lineal para un desvanecimiento espectacular
        rgba[..., 3] = np.clip(P ** 1.8 * 0.85, 0.0, 0.85)
        
        # Flip vertical para coincidir con la orientación de coordenadas de Leaflet
        rgba = np.flipud(rgba)
        
        # Convertir a imagen PNG de 8 bits
        img_data = (rgba * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img_data, 'RGBA')
        
        # Interpolación bilineal de alta definición
        try:
            resample_filter = PIL.Image.Resampling.BILINEAR
        except AttributeError:
            resample_filter = PIL.Image.BILINEAR
            
        img = img.resize((500, 500), resample=resample_filter)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
