"""
Servicio de Análisis de Enlaces RF y Perfil de Terreno.
Integración de SRTM, FSPL, curvatura terrestre y Zona de Fresnel.
"""

import os
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from api.schemas import LinkAnalysisRequest, LinkAnalysisResponse, LinkAnalysisSummary, TerrainPoint
from src.propagation.fspl import fspl_db, received_power_dbm
from src.geo.coordinates import haversine_distance

from src.parsers.srtm_reader import SRTMReader, download_srtm_tile


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _calculate_fresnel_radius_m(d1_km: float, d2_km: float, freq_mhz: float) -> float:
    """Calcula el radio de la 1a Zona de Fresnel en metros."""
    total_d_km = d1_km + d2_km
    if total_d_km <= 0 or freq_mhz <= 0:
        return 0.0
    freq_ghz = freq_mhz / 1000.0
    # Formula: R1 = 17.32 * sqrt((d1 * d2) / (f_GHz * D_km))
    return 17.32 * math.sqrt((d1_km * d2_km) / (freq_ghz * total_d_km))


class LinkService:
    def __init__(self, srtm_dir: Optional[str] = None):
        if srtm_dir is None:
            srtm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "srtm")
        self.srtm_dir = Path(srtm_dir)
        self.srtm_dir.mkdir(parents=True, exist_ok=True)
        self._tile_cache: Dict[str, Any] = {}

    def _get_elevation(self, lat: float, lon: float) -> float:
        """Obtiene la elevación del terreno en metros."""
        lat_floor = int(math.floor(lat))
        lon_floor = int(math.floor(lon))
        
        # Nombre estándar del tile HGT (ej. N04W075)
        lat_str = f"N{abs(lat_floor):02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
        lon_str = f"E{abs(lon_floor):03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
        tile_name = f"{lat_str}{lon_str}"
        
        if tile_name in self._tile_cache and self._tile_cache[tile_name] is not None:
            reader = self._tile_cache[tile_name]
            try:
                return reader.get_elevation(lat, lon)
            except Exception:
                pass

        # Buscar archivo local (.hgt, .zip, .tif)
        possible_files = list(self.srtm_dir.glob(f"{tile_name}*"))
        if possible_files:
            try:
                reader = SRTMReader(possible_files[0])
                self._tile_cache[tile_name] = reader
                return reader.get_elevation(lat, lon)
            except Exception as e:
                print(f"Error leyendo archivo SRTM local {possible_files[0]}: {e}")
                pass
                
        # Intentar descargar tile si no existe
        try:
            hgt_path = download_srtm_tile(lat_floor, lon_floor, output_dir=self.srtm_dir)
            reader = SRTMReader(hgt_path)
            self._tile_cache[tile_name] = reader
            return reader.get_elevation(lat, lon)
        except Exception as e:
            print(f"Error descargando/cargando tile SRTM {tile_name}: {e}")
            pass

        # Fallback sintético si no hay tile disponible
        return 0.0

    def analyze_link(self, req: LinkAnalysisRequest) -> LinkAnalysisResponse:
        total_dist_km = _haversine_km(req.tx.lat, req.tx.lon, req.rx.lat, req.rx.lon)
        if total_dist_km < 0.001:
            total_dist_km = 0.001

        # Muestreo a lo largo de la trayectoria
        n_pts = req.n_samples
        lats = np.linspace(req.tx.lat, req.rx.lat, n_pts)
        lons = np.linspace(req.tx.lon, req.rx.lon, n_pts)
        distances_km = np.linspace(0, total_dist_km, n_pts)

        # Elevaciones del terreno
        terrain_elevs = [self._get_elevation(lat, lon) for lat, lon in zip(lats, lons)]
        # Reemplazar NaNs por 0
        terrain_elevs = [0.0 if math.isnan(e) else float(e) for e in terrain_elevs]

        tx_elev = terrain_elevs[0]
        rx_elev = terrain_elevs[-1]

        tx_total_h = tx_elev + req.tx.height_m
        rx_total_h = rx_elev + req.rx.height_m

        # Curvatura terrestre (k=4/3 Earth radius factor -> dh = d1*d2 / 12.74 m)
        k_factor = 4.0 / 3.0
        earth_r_km = 6371.0 * k_factor

        points: List[TerrainPoint] = []
        los_blocked = False
        min_clearance_m = float('inf')
        min_clearance_pct = float('inf')

        for i in range(n_pts):
            d1_km = distances_km[i]
            d2_km = total_dist_km - d1_km
            frac = d1_km / total_dist_km

            # Altura de la línea de vista (LOS) sobre el nivel del mar
            los_h = tx_total_h + frac * (rx_total_h - tx_total_h)

            # Corrección por curvatura de la tierra
            earth_drop_m = (d1_km * d2_km) / (2.0 * earth_r_km) * 1000.0 if total_dist_km > 0 else 0.0
            effective_terrain_h = terrain_elevs[i] + earth_drop_m

            # Radio Fresnel 1a zona y 60%
            r1_m = _calculate_fresnel_radius_m(d1_km, d2_km, req.frequency_mhz)
            r60_m = 0.60 * r1_m

            # Despeje libre disponible = LOS_height - Effective_Terrain_height
            clearance_m = los_h - effective_terrain_h

            if r60_m > 0:
                clearance_pct = (clearance_m / r60_m) * 100.0
            else:
                clearance_pct = 100.0 if clearance_m >= 0 else -100.0

            # Guardar mínimos (excluyendo extremos donde r60 es muy cercano a cero)
            if 0 < i < n_pts - 1 and r60_m >= 0.1:
                if clearance_m < min_clearance_m:
                    min_clearance_m = clearance_m
                if clearance_pct < min_clearance_pct:
                    min_clearance_pct = clearance_pct

            is_obstructed = clearance_m < r60_m
            if is_obstructed and 0 < i < n_pts - 1:
                los_blocked = True

            points.append(TerrainPoint(
                distance_km=round(float(d1_km), 3),
                lat=round(float(lats[i]), 6),
                lon=round(float(lons[i]), 6),
                elevation_m=round(float(terrain_elevs[i]), 2),
                tx_line_height_m=round(float(los_h), 2),
                fresnel_60_radius_m=round(float(r60_m), 2),
                is_obstructed=is_obstructed
            ))

        if math.isinf(min_clearance_m):
            min_clearance_m = tx_total_h - tx_elev
            min_clearance_pct = 100.0

        # Pérdidas de espacio libre (FSPL)
        path_loss_db = float(fspl_db(total_dist_km, req.frequency_mhz))

        # Potencia recibida en dBm si se especificó potencia y ganancias
        rx_pwr_dbm = None
        if req.tx.power_dbm is not None:
            tx_g = req.tx.gain_dbi or 0.0
            rx_g = req.rx.gain_dbi or 0.0
            rx_pwr_dbm = float(received_power_dbm(
                tx_power_dbm=req.tx.power_dbm,
                tx_gain_dbi=tx_g,
                rx_gain_dbi=rx_g,
                distance_km=total_dist_km,
                frequency_mhz=req.frequency_mhz
            ))

        # Estado general
        if los_blocked or min_clearance_m < 0:
            status = "BLOCKED"
        elif min_clearance_pct < 100.0:
            status = "MARGINAL"
        else:
            status = "CLEAR"

        summary = LinkAnalysisSummary(
            distance_km=round(float(total_dist_km), 3),
            frequency_mhz=float(req.frequency_mhz),
            fspl_db=round(path_loss_db, 2),
            tx_elevation_m=round(float(tx_elev), 2),
            rx_elevation_m=round(float(rx_elev), 2),
            max_terrain_elevation_m=round(float(max(terrain_elevs)), 2),
            min_fresnel_clearance_m=round(float(min_clearance_m), 2),
            min_fresnel_clearance_percent=round(float(min_clearance_pct), 1),
            los_blocked=los_blocked,
            status=status,
            received_power_dbm=round(rx_pwr_dbm, 2) if rx_pwr_dbm is not None else None
        )

        return LinkAnalysisResponse(summary=summary, profile=points)
