"""
Lector de archivos SRTM/HGT — Modelo Digital de Elevación (DEM).

Lee tiles SRTM de la NASA (1-arc-sec o 3-arc-sec) y proporciona
métodos para extraer elevaciones, perfiles de terreno y metadatos
de cobertura geográfica.

Referencia del formato:
    - 16-bit signed integers, big-endian
    - SRTM1: 3601×3601 (~30m), SRTM3: 1201×1201 (~90m)
    - Nomenclatura: N04W075.hgt = lat 4°N-5°N, lon 75°W-74°W
"""

from __future__ import annotations

import os
import re
import zipfile
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SRTM1_SIZE = 3601
SRTM3_SIZE = 1201
VOID_VALUE = -32768

# Patrón regex para extraer lat/lon del nombre del archivo
_FILENAME_RE = re.compile(
    r"([NS])(\d{2})([EW])(\d{3})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclass para metadatos del tile
# ---------------------------------------------------------------------------

@dataclass
class TileInfo:
    """Metadatos de un tile SRTM."""
    filepath: Path
    lat_sw: float           # Latitud de la esquina suroeste
    lon_sw: float           # Longitud de la esquina suroeste
    resolution: int         # 1 (SRTM1) o 3 (SRTM3)
    grid_size: int          # 3601 o 1201
    arc_seconds: float      # 1.0 o 3.0

    @property
    def lat_ne(self) -> float:
        return self.lat_sw + 1.0

    @property
    def lon_ne(self) -> float:
        return self.lon_sw + 1.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Retorna (lon_min, lat_min, lon_max, lat_max)."""
        return (self.lon_sw, self.lat_sw, self.lon_ne, self.lat_ne)


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class SRTMReader:
    """
    Lector de archivos SRTM/HGT.

    Uso básico:
        reader = SRTMReader("data/srtm/N04W075.hgt")
        elevation = reader.get_elevation(4.711, -74.072)
        profile = reader.elevation_profile((4.5, -74.5), (4.8, -74.0), n_points=200)
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._data: Optional[np.ndarray] = None
        self._info: Optional[TileInfo] = None

    # ---- Propiedades ----

    @property
    def info(self) -> TileInfo:
        """Metadatos del tile."""
        if self._info is None:
            self._info = self._parse_tile_info()
        return self._info

    @property
    def data(self) -> np.ndarray:
        """Matriz de elevaciones (float32, NaN para voids)."""
        if self._data is None:
            self._data = self._load()
        return self._data

    # ---- API pública ----

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Obtiene la elevación en una coordenada (lat, lon) en grados decimales.

        Usa interpolación bilineal para mayor precisión.

        Returns:
            Elevación en metros, o NaN si es un void.
        """
        info = self.info
        data = self.data

        # Validar que la coordenada esté dentro del tile
        if not (info.lat_sw <= lat <= info.lat_ne and
                info.lon_sw <= lon <= info.lon_ne):
            raise ValueError(
                f"Coordenada ({lat}, {lon}) fuera del tile "
                f"{info.bounds}"
            )

        # Convertir lat/lon a índices de fila/columna (float)
        row_f = (info.lat_ne - lat) * (info.grid_size - 1)
        col_f = (lon - info.lon_sw) * (info.grid_size - 1)

        # Interpolación bilineal
        row0 = int(np.floor(row_f))
        col0 = int(np.floor(col_f))
        row1 = min(row0 + 1, info.grid_size - 1)
        col1 = min(col0 + 1, info.grid_size - 1)

        frac_row = row_f - row0
        frac_col = col_f - col0

        z00 = data[row0, col0]
        z01 = data[row0, col1]
        z10 = data[row1, col0]
        z11 = data[row1, col1]

        # Si alguno es NaN, retornar NaN
        if any(np.isnan(v) for v in [z00, z01, z10, z11]):
            return float("nan")

        z = (z00 * (1 - frac_row) * (1 - frac_col) +
             z01 * (1 - frac_row) * frac_col +
             z10 * frac_row * (1 - frac_col) +
             z11 * frac_row * frac_col)

        return float(z)

    def elevation_profile(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        n_points: int = 200,
    ) -> dict:
        """
        Calcula el perfil de elevación entre dos puntos.

        Args:
            point_a: (lat, lon) inicio
            point_b: (lat, lon) fin
            n_points: Número de muestras a lo largo del trayecto

        Returns:
            dict con:
                - 'lats': array de latitudes
                - 'lons': array de longitudes
                - 'elevations': array de elevaciones (m)
                - 'distances': array de distancias acumuladas (km)
        """
        lats = np.linspace(point_a[0], point_b[0], n_points)
        lons = np.linspace(point_a[1], point_b[1], n_points)

        elevations = np.array([
            self.get_elevation(lat, lon) for lat, lon in zip(lats, lons)
        ])

        # Calcular distancias acumuladas usando Haversine
        distances = np.zeros(n_points)
        for i in range(1, n_points):
            distances[i] = distances[i - 1] + _haversine_km(
                lats[i - 1], lons[i - 1], lats[i], lons[i]
            )

        return {
            "lats": lats,
            "lons": lons,
            "elevations": elevations,
            "distances": distances,
        }

    def get_stats(self) -> dict:
        """Estadísticas de elevación del tile."""
        valid = self.data[~np.isnan(self.data)]
        return {
            "min_elevation": float(np.min(valid)) if len(valid) > 0 else None,
            "max_elevation": float(np.max(valid)) if len(valid) > 0 else None,
            "mean_elevation": float(np.mean(valid)) if len(valid) > 0 else None,
            "std_elevation": float(np.std(valid)) if len(valid) > 0 else None,
            "void_pixels": int(np.sum(np.isnan(self.data))),
            "total_pixels": self.data.size,
            "void_percentage": float(np.sum(np.isnan(self.data)) / self.data.size * 100),
        }

    # ---- Internos ----

    def _parse_tile_info(self) -> TileInfo:
        """Extrae metadatos. Si es TIF, usa rasterio."""
        filepath = self.filepath
        
        # Si es un TIF, podemos leer todo desde rasterio
        if filepath.suffix.lower() == ".tif":
            import rasterio
            with rasterio.open(filepath) as src:
                bounds = src.bounds
                # rasterio bounds: (left, bottom, right, top)
                lon_sw = bounds.left
                lat_sw = bounds.bottom
                grid_size = src.width
                
                # Asumimos que si grid_size > 2000 es resolución alta (~30m)
                resolution = 1 if grid_size > 2000 else 3
                
                return TileInfo(
                    filepath=filepath,
                    lat_sw=lat_sw,
                    lon_sw=lon_sw,
                    resolution=resolution,
                    grid_size=grid_size,
                    arc_seconds=float(resolution),
                )
        
        # Fallback original para .hgt
        stem = filepath.stem.upper()
        match = _FILENAME_RE.search(stem)
        if not match:
            raise ValueError(f"No se puede parsear nombre de tile: {filepath.name}")

        ns, lat_str, ew, lon_str = match.groups()
        lat = int(lat_str) * (1 if ns == "N" else -1)
        lon = int(lon_str) * (1 if ew == "E" else -1)

        file_size = self._get_file_size()
        expected_srtm1 = SRTM1_SIZE * SRTM1_SIZE * 2
        expected_srtm3 = SRTM3_SIZE * SRTM3_SIZE * 2

        if file_size == expected_srtm1:
            resolution, grid_size = 1, SRTM1_SIZE
        elif file_size == expected_srtm3:
            resolution, grid_size = 3, SRTM3_SIZE
        else:
            if abs(file_size - expected_srtm1) < abs(file_size - expected_srtm3):
                resolution, grid_size = 1, SRTM1_SIZE
            else:
                resolution, grid_size = 3, SRTM3_SIZE

        return TileInfo(
            filepath=filepath,
            lat_sw=float(lat),
            lon_sw=float(lon),
            resolution=resolution,
            grid_size=grid_size,
            arc_seconds=float(resolution),
        )

    def _get_file_size(self) -> int:
        if self.filepath.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.filepath) as zf:
                hgt_files = [f for f in zf.namelist() if f.endswith(".hgt")]
                if not hgt_files:
                    raise ValueError(f"No .hgt found in {self.filepath}")
                return zf.getinfo(hgt_files[0]).file_size
        return self.filepath.stat().st_size

    def _load(self) -> np.ndarray:
        """Carga el archivo TIF o HGT en un array NumPy."""
        # Soporte para TIF
        if self.filepath.suffix.lower() == ".tif":
            import rasterio
            with rasterio.open(self.filepath) as src:
                data = src.read(1)
                # TIFs a veces usan otros nodata o el mismo -32768
                nodata = src.nodata if src.nodata is not None else VOID_VALUE
                result = data.astype(np.float32)
                result[data == nodata] = np.nan
                return result
                
        # Soporte para HGT
        info = self.info
        raw_bytes = self._read_bytes()

        data = np.frombuffer(raw_bytes, dtype=">i2")  # Big-endian int16
        data = data.reshape((info.grid_size, info.grid_size))

        result = data.astype(np.float32)
        result[data == VOID_VALUE] = np.nan

        return result

    def _read_bytes(self) -> bytes:
        if self.filepath.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.filepath) as zf:
                hgt_files = [f for f in zf.namelist() if f.endswith(".hgt")]
                return zf.read(hgt_files[0])
        return self.filepath.read_bytes()

    def __repr__(self) -> str:
        info = self.info
        return (
            f"SRTMReader(tile='{self.filepath.stem}', "
            f"bounds={info.bounds}, "
            f"resolution=SRTM{info.resolution})"
        )


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine entre dos puntos en km."""
    R = 6371.0  # Radio de la Tierra en km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def download_srtm_tile(
    lat: int,
    lon: int,
    output_dir: str | Path = "data/srtm",
    version: str = "v3.0",
) -> Path:
    """
    Descarga un tile SRTM desde el registro público de AWS (Mapzen Skadi).
    
    Provee archivos .hgt (SRTM1 30m) comprimidos en .gz sin necesidad de autenticación.
    
    Args:
        lat: Latitud de la esquina SW
        lon: Longitud de la esquina SW
        output_dir: Directorio de salida
        version: Ignorado

    Returns:
        Path al archivo descargado (.hgt)
    """
    import requests
    import gzip
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    
    lat_str = f"{ns}{abs(lat):02d}"
    lon_str = f"{ew}{abs(lon):03d}"
    tile_name = f"{lat_str}{lon_str}"
    
    output_path = output_dir / f"{tile_name}.hgt"
    
    if output_path.exists():
        print(f"  Tile {tile_name} ya existe en {output_path}")
        return output_path

    # AWS Mapzen Skadi format: skadi/{N|S}{lat}/{N|S}{lat}{E|W}{lon}.hgt.gz
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_str}/{tile_name}.hgt.gz"
    gz_path = output_dir / f"{tile_name}.hgt.gz"

    print(f"  ⬇ Descargando tile {tile_name} desde AWS Skadi...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        
        # Guardar archivo gz
        with open(gz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Descomprimir a .hgt
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        # Limpiar gz
        gz_path.unlink()
        
        print(f"  ✓ Descargado y descomprimido: {output_path}")
    except Exception as e:
        print(f"  ✗ No se pudo descargar {tile_name}: {e}")
        if gz_path.exists():
            gz_path.unlink()
        raise

    return output_path
