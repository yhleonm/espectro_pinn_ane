"""
Utilidades de coordenadas geoespaciales para Colombia.

Incluye:
    - Constantes del territorio colombiano
    - Transformaciones entre WGS-84 y MAGNA-SIRGAS
    - Fórmula de Haversine para distancias
    - Selección automática de zona MAGNA-SIRGAS
"""

from __future__ import annotations

from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


# ---------------------------------------------------------------------------
# Constantes del territorio colombiano
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColombiaBounds:
    """Bounding box del territorio continental de Colombia."""
    lat_min: float = -4.23
    lat_max: float = 12.46
    lon_min: float = -81.73
    lon_max: float = -66.85
    name: str = "Colombia Continental"

    def contains(self, lat: float, lon: float) -> bool:
        """Verifica si una coordenada está dentro de Colombia."""
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)


colombia_bbox = ColombiaBounds()


# ---------------------------------------------------------------------------
# Zonas MAGNA-SIRGAS proyectadas
# ---------------------------------------------------------------------------

MAGNA_ZONES = {
    "far_west":    {"epsg": 3114, "epsg_2018": 11114, "lon_center": -80.077},
    "west":        {"epsg": 3115, "epsg_2018": 11115, "lon_center": -77.077},
    "bogota":      {"epsg": 3116, "epsg_2018": 11116, "lon_center": -74.077},
    "east_central":{"epsg": 3117, "epsg_2018": 11117, "lon_center": -71.077},
    "east":        {"epsg": 3118, "epsg_2018": 11118, "lon_center": -68.077},
}

# CRS geográfico base
MAGNA_GEO_EPSG = 4686
MAGNA_GEO_2018_EPSG = 20046


def select_magna_zone(lon: float) -> dict:
    """
    Selecciona la zona MAGNA-SIRGAS apropiada para una longitud dada.

    Args:
        lon: Longitud en grados decimales (negativa para W)

    Returns:
        dict con las claves 'epsg', 'epsg_2018', 'lon_center'
    """
    best_zone = "bogota"
    best_dist = float("inf")

    for name, zone in MAGNA_ZONES.items():
        dist = abs(lon - zone["lon_center"])
        if dist < best_dist:
            best_dist = dist
            best_zone = name

    return MAGNA_ZONES[best_zone]


def transform_wgs84_to_magna(
    lat: float,
    lon: float,
    use_2018: bool = True,
) -> Tuple[float, float]:
    """
    Transforma coordenadas WGS-84 a MAGNA-SIRGAS proyectado.

    Selecciona automáticamente la zona correcta según la longitud.

    Args:
        lat: Latitud (WGS-84)
        lon: Longitud (WGS-84)
        use_2018: Si True, usa MAGNA-SIRGAS 2018

    Returns:
        (easting, northing) en metros
    """
    zone = select_magna_zone(lon)
    epsg = zone["epsg_2018"] if use_2018 else zone["epsg"]

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return x, y


def transform_magna_to_wgs84(
    easting: float,
    northing: float,
    zone_epsg: int = 3116,
) -> Tuple[float, float]:
    """
    Transforma coordenadas MAGNA-SIRGAS proyectado a WGS-84.

    Args:
        easting: Coordenada X (metros)
        northing: Coordenada Y (metros)
        zone_epsg: EPSG de la zona MAGNA-SIRGAS

    Returns:
        (lat, lon) en grados decimales
    """
    transformer = Transformer.from_crs(
        f"EPSG:{zone_epsg}", "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calcula la distancia entre dos puntos usando la fórmula de Haversine.

    Args:
        lat1, lon1: Punto A (grados decimales)
        lat2, lon2: Punto B (grados decimales)

    Returns:
        Distancia en kilómetros
    """
    R = 6371.0  # Radio de la Tierra en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)

    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Ciudades principales (para referencia rápida)
# ---------------------------------------------------------------------------

COLOMBIA_CITIES = {
    "Bogotá":     {"lat": 4.7110, "lon": -74.0721, "elev": 2640},
    "Medellín":   {"lat": 6.2518, "lon": -75.5636, "elev": 1495},
    "Cali":       {"lat": 3.4516, "lon": -76.5225, "elev": 995},
    "Barranquilla":{"lat": 10.9685, "lon": -74.7813, "elev": 18},
    "Cartagena":  {"lat": 10.3910, "lon": -75.5144, "elev": 2},
    "Bucaramanga":{"lat": 7.1254, "lon": -73.1198, "elev": 959},
    "Pereira":    {"lat": 4.8133, "lon": -75.6961, "elev": 1411},
    "Manizales":  {"lat": 5.0689, "lon": -75.5174, "elev": 2153},
    "Cúcuta":     {"lat": 7.8891, "lon": -72.4967, "elev": 320},
    "Ibagué":     {"lat": 4.4389, "lon": -75.2322, "elev": 1285},
}


def srtm_tiles_for_colombia() -> list:
    """
    Retorna la lista de tiles SRTM necesarias para cubrir Colombia.

    Returns:
        Lista de strings con notación SRTM (ej: 'N04W075')
    """
    tiles = []
    for lat in range(-5, 13):  # -4.23 a 12.46
        for lon in range(-82, -66):  # -81.73 a -66.85
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
    return tiles
