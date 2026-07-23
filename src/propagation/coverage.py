"""
Motor de Análisis de Cobertura Espacial Espacial (2D Grid).

Genera matrices (rasters) de pérdida de trayectoria y potencia recibida
sobre un área geográfica definida, emulando la "mancha de cobertura"
de sistemas como ICS Manager.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from src.propagation.fspl import fspl_db, received_power_dbm
from src.geo.coordinates import haversine_distance
import math
from src.geo.coordinates import haversine_distance

@dataclass
class CoverageGrid:
    """Representa una matriz de cobertura geográfica."""
    lats: np.ndarray      # Vector 1D de latitudes (eje Y)
    lons: np.ndarray      # Vector 1D de longitudes (eje X)
    power_dbm: np.ndarray # Matriz 2D de potencia recibida (shape: len(lats), len(lons))
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max)"""
        return (self.lons.min(), self.lats.min(), self.lons.max(), self.lats.max())


def generate_fspl_coverage(
    tx_lat: float,
    tx_lon: float,
    tx_power_dbm: float,
    tx_gain_dbi: float,
    frequency_mhz: float,
    radius_km: float = 20.0,
    resolution_points: int = 100,
) -> CoverageGrid:
    """
    Genera un Heatmap 2D basado puramente en Pérdida de Espacio Libre (FSPL).
    No considera terreno (ideal para un baseline rápido).
    
    Args:
        tx_lat, tx_lon: Centro de transmisión
        tx_power_dbm: Potencia del transmisor
        tx_gain_dbi: Ganancia focal de la antena TX
        frequency_mhz: Frecuencia de operación
        radius_km: Radio del área cuadrada a simular
        resolution_points: Pixeles por lado (ej. 100x100)
    """
    # Aproximación rápida: 1 grado de latitud ~ 111 km
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * np.cos(np.radians(tx_lat)))
    
    lats = np.linspace(tx_lat - lat_offset, tx_lat + lat_offset, resolution_points)
    lons = np.linspace(tx_lon - lon_offset, tx_lon + lon_offset, resolution_points)
    
    # Meshgrid para evaluar toda la matriz a la vez (vectorizado)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Calcular distancias desde el TX a cada punto de la grilla
    # Versión vectorizada de Haversine
    lat1 = np.radians(tx_lat)
    lon1 = np.radians(tx_lon)
    lat2 = np.radians(lat_grid)
    lon2 = np.radians(lon_grid)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    
    dist_km = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Prevenir distancia cero (infinito en log)
    dist_km = np.maximum(dist_km, 0.001)
    
    # Calcular pérdida y potencia
    power_grid = received_power_dbm(
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=0.0, # Asumimos antena receptora isótropa
        distance_km=dist_km,
        frequency_mhz=frequency_mhz
    )
    
    return CoverageGrid(lats=lats, lons=lons, power_dbm=power_grid)


def apply_terrain_mask(
    coverage: CoverageGrid,
    srtm_readers: list,
    tx_lat: float,
    tx_lon: float,
    tx_height: float = 30.0,
    rx_height: float = 2.0,
    frequency_mhz: float = 98.0
) -> CoverageGrid:
    """
    Motor Empírico: Trazado de Rayos 3D y Difracción Knife-Edge (ITU-R P.526).
    Se usa como Baseline Científico para comparar con la PINN.
    """
    print(f"  [Baseline ITU] Iniciando Trazado Radial Knife-Edge (Grid: {coverage.lats.size}x{coverage.lons.size})")
    
    # 1. Obtener altitud base del TX
    tx_base_m = 0.0
    for reader in srtm_readers:
        info = reader.info
        if info.lat_sw <= tx_lat <= info.lat_ne and info.lon_sw <= tx_lon <= info.lon_ne:
            z = reader.get_elevation(tx_lat, tx_lon)
            if not np.isnan(z):
                tx_base_m = z
            break
    
    tx_z = tx_base_m + tx_height
    wavelength_m = 300.0 / frequency_mhz
    
    # Pre-reservar memoria para la máscara de difracción
    diffraction_loss_db = np.zeros_like(coverage.power_dbm)
    
    # Malla de coordenadas
    lon_grid, lat_grid = np.meshgrid(coverage.lons, coverage.lats)
    
    # Para optimizar el trazado, procesamos pixel por pixel (Podría vectorizarse con Bresenham 3D)
    rows, cols = coverage.power_dbm.shape
    
    import sys
    
    for r in range(rows):
        for c in range(cols):
            rx_lat = lat_grid[r, c]
            rx_lon = lon_grid[r, c]
            
            # 2. Generar Perfil de Elevación Radial (Muestreado cada ~100 metros)
            dist_km = haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
            if dist_km < 0.1:
                continue # Demasiado cerca, FSPL es suficiente
                
            n_points = max(10, int(dist_km * 10)) # 10 muestras por km
            
            lats = np.linspace(tx_lat, rx_lat, n_points)
            lons = np.linspace(tx_lon, rx_lon, n_points)
            
            elevations = []
            for i in range(n_points):
                z = 0.0
                for reader in srtm_readers:
                    info = reader.info
                    # Fast bounds check
                    if info.lat_sw <= lats[i] <= info.lat_ne and info.lon_sw <= lons[i] <= info.lon_ne:
                        try:
                            z_val = reader.get_elevation(lats[i], lons[i])
                            if not np.isnan(z_val):
                                z = z_val
                        except:
                            pass
                        break
                elevations.append(z)
                
            rx_z = elevations[-1] + rx_height
            
            # 3. Detectar el Obstáculo Más Crítico (Highest Knife-Edge)
            # Calculamos la línea de vista (LoS) matemática ideal
            # Z_los(d) = tx_z + (d / D) * (rx_z - tx_z)
            
            max_nu = -float('inf')
            
            for i in range(1, n_points - 1): # Ignorar primer y último punto
                d1_km = (i / (n_points - 1)) * dist_km
                d2_km = dist_km - d1_km
                
                # Altura de la línea de vista en este punto
                los_z = tx_z + (d1_km / dist_km) * (rx_z - tx_z)
                
                # Altura física real de la montaña
                obst_z = elevations[i]
                
                # h = Altura del obstáculo bloqueando la Línea de Vista
                h_m = obst_z - los_z
                
                # Parámetro de difracción de Fresnel (v)
                # v = h * sqrt( (2 * (d1 + d2)) / (lambda * d1 * d2) )
                if d1_km > 0 and d2_km > 0:
                    d1_m = d1_km * 1000
                    d2_m = d2_km * 1000
                    nu = h_m * math.sqrt((2.0 * (d1_m + d2_m)) / (wavelength_m * d1_m * d2_m))
                    if nu > max_nu:
                        max_nu = nu
            
            # 4. Aproximación ITU-R P.526 para Pérdida por Filo de Cuchillo
            L_dif_db = 0.0
            if max_nu > -0.78:
                # Solo hay difracción si el obstáculo bloquea al menos el 60% de la primera zona de Fresnel
                nu = max_nu
                L_dif_db = 6.9 + 20 * math.log10(math.sqrt((nu - 0.1)**2 + 1) + nu - 0.1)
                L_dif_db = max(0.0, L_dif_db) # No ganancia, solo pérdida
                
            diffraction_loss_db[r, c] = L_dif_db
            
    print("  [Baseline ITU] Heatmap Empírico Completado.")
    
    # 5. Aplicar la pérdida estática de difracción al Heatmap
    new_power_dbm = coverage.power_dbm - diffraction_loss_db
    
    return CoverageGrid(lats=coverage.lats, lons=coverage.lons, power_dbm=new_power_dbm)
