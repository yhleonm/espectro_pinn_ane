"""
Motor de Entrenamiento PINN (In-Situ) — Arquitectura Residual.

La red neuronal predice un FACTOR DE CORRECCIÓN TOPOGRÁFICA que se multiplica
por el valor analítico FSPL. Esto garantiza que el decaimiento macroscópico
(1/r²) sea correcto por construcción, y la PINN solo aprende las perturbaciones
causadas por el terreno, difracción y patrón de antena.

    Señal_final(x,y,z) = FSPL(d) × PINN_correction(x,y,z)
"""

import torch
import torch.optim as optim
import numpy as np
import hashlib
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

from src.ai.pinn_solver import HelmholtzPINN, eikonal_pde_loss_phase, phase_to_amplitude
from src.ai.boundary_conditions import PINNBoundary, boundary_loss
from src.propagation.fspl import fspl_db
import pandas as pd

PINN_CACHE_DIR = Path("data/pinn_cache")
CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 días

class PINNTrainer:
    """
    Gestiona el ciclo de vida del entrenamiento In-Situ de la PINN (Residual).
    """
    NORM_Z = 3.5  # km — Constante de normalización para la coordenada Z (unificada)
    def __init__(self, tx_lat: float, tx_lon: float, tx_height_m: float, tx_power_dbm: float, 
                 frequency_mhz: float, tx_azimuth_deg: float = 0.0, tx_tilt_deg: float = 0.0,
                 tx_hpbw_h_deg: float = 65.0, tx_hpbw_v_deg: float = 65.0,
                 radius_km: float = 20.0, resolution: int = 100,
                 epochs_ia: int = 500, device: str = None, city_slug: str = "bogota"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🔧 Inicializando PINN Residual en dispositivo: {self.device} (City: {city_slug})")
        
        self.frequency_mhz = frequency_mhz
        self.tx_power_dbm = tx_power_dbm
        self.epochs_ia = epochs_ia
        self.city_slug = city_slug
        
        # 1. Configurar Geometría (Boundaries)
        self.boundary = PINNBoundary(
            tx_lat=tx_lat, 
            tx_lon=tx_lon, 
            tx_height_m=tx_height_m,
            frequency_mhz=frequency_mhz,
            tx_azimuth_deg=tx_azimuth_deg,
            tx_tilt_deg=tx_tilt_deg,
            tx_hpbw_h_deg=tx_hpbw_h_deg,
            tx_hpbw_v_deg=tx_hpbw_v_deg,
            radius_km=radius_km, 
            resolution_points=resolution,
            device=self.device,
            city_slug=city_slug
        )
        
        # 2. Inicializar Modelo Neuronal (Arquitectura Residual)
        self.model = HelmholtzPINN(in_features=2, hidden_features=128, hidden_layers=4).to(self.device)
        
        # 3. Configurar Optimizador y Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=100, factor=0.5)
        
        # Caché de Fronteras (No cambian durante el entrenamiento)
        self.terrain_coords = self.boundary.get_terrain_boundary()
        self.tx_coords, self.tx_target_correction = self.boundary.get_source_boundary(self.tx_power_dbm)
        
        # Pre-calcular el Prior FSPL para el grid 2D
        self.fspl_prior_grid = self.boundary.get_fspl_prior_grid(self.tx_power_dbm)
        
        # 4. Cargar Datos Empíricos (ANE) para el Entrenamiento Híbrido
        self.empirical_coords_norm = None
        self.empirical_target = None
        self.last_loss = 0.0
        self._load_empirical_data()
        
        # Hash de parámetros para caché determinista
        self._cache_hash = PINNTrainer._param_hash(
            tx_lat=tx_lat, tx_lon=tx_lon, tx_power_dbm=tx_power_dbm,
            frequency_mhz=frequency_mhz, radius_km=radius_km, resolution=resolution,
            tx_azimuth_deg=tx_azimuth_deg, tx_tilt_deg=tx_tilt_deg,
            tx_hpbw_h_deg=tx_hpbw_h_deg, tx_hpbw_v_deg=tx_hpbw_v_deg,
            epochs_ia=epochs_ia
        )
        print(f"DEBUG: Cache hash = {self._cache_hash}")

    @staticmethod
    def _param_hash(**kwargs) -> str:
        """Genera un hash SHA-256 determinista a partir de los parámetros RF."""
        rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in sorted(kwargs.items())}
        raw = json.dumps(rounded, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_empirical_data(self):
        """Carga datos empíricos de la ANE para supervisión híbrida."""
        print("⚠️ Datos empíricos ANE desactivados (solo Popayán disponible, incompatible con TX actual).")
        print("   → PINN opera en modo físico puro (Eikonal + terrain boundary).")
        self.empirical_coords_norm = None
        self.empirical_target = None
        return

    def save_weights(self) -> Path:
        """Persiste model.state_dict() a disco con metadatos de timestamp."""
        PINN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = PINN_CACHE_DIR / f"{self._cache_hash}.pt"
        torch.save({
            "state_dict": self.model.state_dict(),
            "timestamp": time.time(),
            "last_loss": self.last_loss
        }, cache_path)
        print(f"💾 Pesos PINN guardados en: {cache_path}")
        return cache_path

    def load_weights(self) -> bool:
        """Restaura pesos si existe un checkpoint válido (< 24h). Retorna True si cargó exitosamente."""
        cache_path = PINN_CACHE_DIR / f"{self._cache_hash}.pt"
        if not cache_path.exists():
            return False
        
        checkpoint = torch.load(cache_path, map_location=self.device, weights_only=True)
        age_seconds = time.time() - checkpoint.get("timestamp", 0)
        
        if age_seconds > CACHE_EXPIRY_SECONDS:
            print(f"⏰ Caché expirado ({age_seconds/3600:.1f}h). Reentrenando...")
            cache_path.unlink()
            return False
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.last_loss = checkpoint.get("last_loss", 0.0)
        print(f"✅ Pesos PINN cargados desde caché ({age_seconds/60:.0f} min de antigüedad) | hash={self._cache_hash}")
        return True

    def train(self, epochs: int = 1500):
        """Bucle de entrenamiento mejorado (Camino B)."""
        print(f"🚀 Iniciando optimización in-situ (CAMINO B) | Epochs: {epochs}")
        pbar = tqdm(range(epochs), desc="Calibrando PINN")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        
        for epoch in pbar:
            if epoch < 200:
                # Warm-up de fuente
                lambdas = {"pde": 0.0, "source": 100.0, "terrain": 5.0}
            else:
                # Entrenamiento completo: PDE + Fuente + Obstáculo Físico
                lambdas = {"pde": 150.0, "source": 50.0, "terrain": 50.0}
            
            metrics = self._train_step_weighted(lambdas)
            scheduler.step()
            
            if epoch % 50 == 0:
                pbar.set_postfix({
                    "Loss": f"{metrics['loss_total']:.2e}",
                    "PDE": f"{metrics['loss_pde']:.2e}",
                    "Src": f"{metrics['loss_source']:.2e}"
                })
        print("✅ PINN Calibrada mediante Camino B.")

    def _train_step_weighted(self, lambdas: dict, num_collocation_points: int = 6000) -> dict:
        """Paso de entrenamiento con pesos de pérdida controlados externamente."""
        self.optimizer.zero_grad()
        colcoords, z_base, z_roughness, clutter_factor = self.boundary.get_collocation_points(num_points=num_collocation_points)
        norm_xy = self.boundary.radius_km
        
        colcoords_norm = colcoords.clone()
        colcoords_norm[:, :2] = colcoords_norm[:, :2] / norm_xy
        colcoords_norm[:, 2] = colcoords_norm[:, 2] / self.NORM_Z
        colcoords_norm.requires_grad_(True)
        
        # 2. Slowness Field (ALPHA Espacial + Clutter OSM)
        rough_mean = z_roughness.mean()
        alpha_topo = 1.0 + (z_roughness / (rough_mean if rough_mean > 0 else 1.0))
        alpha_urban = 1.0 + (clutter_factor * 1.5)
        alpha_topo_norm = (alpha_topo / alpha_topo.mean()) * 0.5
        srtm_slowness = (1.0 + alpha_topo_norm) * alpha_urban
        
        T_vacuum, coords_grad = self.model(colcoords_norm[:, :2], scale_km=norm_xy)
        loss_pde = eikonal_pde_loss_phase(T_vacuum, coords_grad, srtm_slowness=srtm_slowness, scale_xy=norm_xy)
        
        tx_norm = self.tx_coords.clone()
        tx_norm[:, :2] = tx_norm[:, :2] / norm_xy
        T_source, _ = self.model(tx_norm[:, :2], scale_km=norm_xy)
        
        terrain_norm = self.terrain_coords.clone()
        terrain_norm[:, :2] = terrain_norm[:, :2] / norm_xy
        T_terrain, _ = self.model(terrain_norm[:, :2], scale_km=norm_xy)
        
        loss_terrain, loss_source = boundary_loss(
            T_terrain=T_terrain, T_source=T_source, 
            correction_target_source=self.tx_target_correction,
            z_terrain_km=terrain_norm[:, 2:3] * self.NORM_Z,
            tx_z_km=self.boundary.tx_height_m / 1000.0
        )
        
        loss_total = (lambdas["pde"] * loss_pde + lambdas["source"] * loss_source + lambdas["terrain"] * loss_terrain)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.last_loss = loss_total.item()
        return {"loss_total": loss_total.item(), "loss_pde": loss_pde.item(), "loss_source": loss_source.item()}

    @torch.no_grad()
    def infer_grid(self) -> torch.Tensor:
        """
        Inferencia 2D del mapa de calor con Sombra Topográfica Real.
        
        Composición de tres capas:
        1. FSPL analítico (decaimiento 1/r²)
        2. Corrección de rugosidad PINN (T_excess → dB)
        3. Máscara de sombra topográfica (Knife-Edge desde DEM)
        """
        self.model.eval()
        x_flat = self.boundary.x_km.flatten()
        y_flat = self.boundary.y_km.flatten()
        all_coords = torch.tensor(np.stack([x_flat, y_flat], axis=-1), dtype=torch.float32, device=self.device)
        norm_xy = self.boundary.radius_km
        all_coords = all_coords / norm_xy
        
        T_pred_list = []
        batch_size = 8192
        for i in range(0, all_coords.shape[0], batch_size):
            batch = all_coords[i:i+batch_size]
            T_batch, _ = self.model(batch, scale_km=norm_xy)
            T_pred_list.append(T_batch.cpu().numpy())
                
        T_pred = np.vstack(T_pred_list).flatten()
        dist_phys_flat = np.sqrt(x_flat**2 + y_flat**2)
        T_excess = np.maximum(T_pred - dist_phys_flat, 0.0)
        
        clutter_flat = self.boundary.urban_clutter_flat
        T_excess_urban = T_excess * (1.0 + clutter_flat * 2.0)
        
        # Capa 2: Corrección de rugosidad PINN (ALPHA calibrado al benchmark)
        ALPHA = 0.0994
        T_excess_efectivo = np.maximum(T_excess_urban - 0.2, 0.0)
        correction_db = -T_excess_efectivo * ALPHA * 8.686

        dist_km = np.maximum(dist_phys_flat, 0.001)
        path_loss_db = fspl_db(dist_km, self.frequency_mhz)
        REGULATORY_OFFSET = -10.0
        pinn_dbm = self.tx_power_dbm + 3.0 + REGULATORY_OFFSET - path_loss_db + correction_db
        
        # Capa 3: Máscara de Sombra Topográfica (Knife-Edge vectorizado desde DEM)
        print(f"  [Trainer] Shadow Mask activa: {self.boundary.use_shadow_mask}")
        if self.boundary.use_shadow_mask:
            shadow_loss_db = self._compute_terrain_shadow()
            pinn_dbm = pinn_dbm - shadow_loss_db.flatten()
        else:
            print("  [Trainer] Shadow Mask DESACTIVADA")
        
        return torch.tensor(pinn_dbm.reshape(self.boundary.x_km.shape))

    def _compute_terrain_shadow(self) -> np.ndarray:
        """
        Calcula pérdida por difracción Knife-Edge para cada píxel del grid.
        
        Traza un perfil radial desde TX a cada píxel usando el DEM suavizado,
        y calcula el parámetro de Fresnel (nu) del obstáculo más crítico.
        
        Optimización: usa muestreo radial con 20 puntos por perfil (suficiente
        para resolución 100x100 en 30km) y vectoriza la interpolación.
        
        Returns:
            Array 2D (resolution × resolution) con pérdida de difracción en dB.
        """
        import math
        from scipy.interpolate import RegularGridInterpolator
        
        resolution = self.boundary.resolution
        z_km = self.boundary.z_km  # DEM suavizado en km
        x_km_grid = self.boundary.x_km
        y_km_grid = self.boundary.y_km
        
        # Interpolador del DEM
        interp_z = RegularGridInterpolator(
            (y_km_grid[:, 0], x_km_grid[0, :]), z_km,
            bounds_error=False, fill_value=z_km.min()
        )
        
        # Altura del TX (terreno + antena) en km
        tx_z_km = self.boundary.tx_height_m / 1000.0
        wavelength_km = (300.0 / self.frequency_mhz) / 1000.0  # λ en km
        rx_height_km = 0.002  # 2m receptor
        
        shadow_loss = np.zeros(resolution * resolution)
        n_profile = 50  # Mayor resolución para capturar picos en terrenos complejos (Medellín)
        
        for idx in range(resolution * resolution):
            rx_x = x_km_grid.flatten()[idx]
            rx_y = y_km_grid.flatten()[idx]
            
            dist_total = math.sqrt(rx_x**2 + rx_y**2)
            if dist_total < 0.5:  # Menos de 500m: sin difracción significativa
                continue
            
            # Perfil radial TX(0,0) → RX(rx_x, rx_y)
            t_vals = np.linspace(0, 1, n_profile + 2)[1:-1]  # Excluir extremos
            profile_x = rx_x * t_vals
            profile_y = rx_y * t_vals
            profile_pts = np.stack([profile_y, profile_x], axis=1)
            z_profile = interp_z(profile_pts)
            
            # Altura RX (terreno + rx_height)
            rx_z_km = interp_z(np.array([[rx_y, rx_x]]))[0] + rx_height_km
            
            # Línea de vista y parámetro de Fresnel
            max_nu = -1.0
            for i in range(n_profile):
                d1 = t_vals[i] * dist_total  # km desde TX
                d2 = dist_total - d1  # km hasta RX
                
                if d1 < 0.01 or d2 < 0.01:
                    continue
                
                # Altura de la línea de vista en este punto
                los_z = tx_z_km + (d1 / dist_total) * (rx_z_km - tx_z_km)
                
                # Clearance del obstáculo (positivo = bloquea)
                h_km = z_profile[i] - los_z
                
                # Parámetro de Fresnel: nu = h * sqrt(2*(d1+d2) / (λ*d1*d2))
                nu = h_km * math.sqrt(2.0 * (d1 + d2) / (wavelength_km * d1 * d2))
                if nu > max_nu:
                    max_nu = nu
            
            # ITU-R P.526 Knife-Edge approximation
            if max_nu > -0.78:
                L_dif = 6.9 + 20 * math.log10(
                    math.sqrt((max_nu - 0.1)**2 + 1) + max_nu - 0.1
                )
                shadow_loss[idx] = max(0.0, L_dif)
        
        print(f"  [Shadow Mask] Pérdida máx por difracción detectada: {np.max(shadow_loss):.1f} dB")
        return shadow_loss.reshape(resolution, resolution)

    @torch.no_grad()
    def infer_points(self, lats: np.ndarray, lons: np.ndarray, elevations_m: np.ndarray, rx_height_m: float) -> np.ndarray:
        """Evaluación 1D del perfil de propagación."""
        self.model.eval()
        km_per_lat = 111.0
        km_per_lon = 111.0 * np.cos(np.radians(self.boundary.tx_lat))
        x_km = (lons - self.boundary.tx_lon) * km_per_lon
        y_km = (lats - self.boundary.tx_lat) * km_per_lat
        eval_coords = np.stack([x_km, y_km], axis=-1)
        eval_tensor = torch.tensor(eval_coords, dtype=torch.float32, device=self.device)
        norm_xy = self.boundary.radius_km
        eval_tensor = eval_tensor / norm_xy
        
        T_raw, _ = self.model(eval_tensor, scale_km=norm_xy)
        T_profile = T_raw.cpu().numpy().flatten()
        dist_km_phys = np.sqrt(x_km**2 + y_km**2)
        T_excess = np.maximum(T_profile - dist_km_phys, 0.0)
        
        # ALPHA calibrado al benchmark (consistente con infer_grid)
        ALPHA = 0.0994
        T_excess_efectivo = np.maximum(T_excess - 0.2, 0.0)
        correction_db = -T_excess_efectivo * ALPHA * 8.686

        dist_km = np.maximum(dist_km_phys, 0.001)
        path_loss_db = fspl_db(dist_km, self.frequency_mhz)
        REGULATORY_OFFSET = -10.0
        pinn_dbm = self.tx_power_dbm + 3.0 + REGULATORY_OFFSET - path_loss_db + correction_db
        return pinn_dbm
