"""
Servicio de acceso al inventario nacional de estaciones FM — ANE Colombia.

Carga el CSV del inventario (data/ane_inventario_fm.csv), parsea las
coordenadas DMS a decimales, y provee filtros por departamento, ciudad,
frecuencia y análisis de co-canal e interferencia.

Singleton a nivel de módulo: importar `station_service` directamente.

Dependencias:
    - src.parsers.dms_parser (DMS → decimal)
    - src.geo.coordinates (haversine_distance)
    - src.propagation.fspl (fspl_db)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.parsers.dms_parser import parse_dms_to_decimal
from src.geo.coordinates import haversine_distance
from src.propagation.fspl import fspl_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ruta por defecto del inventario ANE
# ---------------------------------------------------------------------------
_DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ane_inventario_fm.csv"


# ---------------------------------------------------------------------------
# Mapeo de columnas CSV → nombres internos
# ---------------------------------------------------------------------------
_COLUMN_MAP = {
    "ID Estación": "id",
    "Frecuencia (MHz)": "frequency_mhz",
    "Frecuencia Enlace (MHz)": "frecuencia_enlace_mhz",
    "PRA (kW)": "pra_kw",
    "Clase": "clase",
    "Estado": "estado",
    "Servicio": "servicio",
    "Altura (m)": "altura_m",
    "Departamento estación Tx": "departamento",
    "Municipio estación Tx": "municipio",
    "Código Dane estación Tx": "codigo_dane",
    "Longitud DMS estación Tx": "lon_dms",
    "Latitud DMS estación Tx": "lat_dms",
}


class StationService:
    """
    Servicio singleton para el inventario nacional de estaciones FM.

    Carga el CSV del ANE una sola vez y expone métodos de consulta
    con coordenadas ya convertidas a grados decimales.
    """

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self._csv_path = csv_path or _DEFAULT_CSV_PATH
        self._stations: list[dict] = []
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Carga interna
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Carga lazy del CSV. Solo se ejecuta una vez."""
        if self._loaded:
            return
        self._load_csv()
        self._loaded = True

    def _load_csv(self) -> None:
        """Lee el CSV del inventario ANE, parsea DMS y construye la lista."""
        if not self._csv_path.exists():
            logger.error("Inventario ANE no encontrado: %s", self._csv_path)
            self._stations = []
            return

        try:
            # El CSV tiene 4 filas de metadata antes de los headers reales
            df = pd.read_csv(
                self._csv_path,
                skiprows=4,
                encoding="utf-8",
                dtype=str,  # leer todo como string para parsear manualmente
            )

            # Limpiar columna fantasma por metadata
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            # Renombrar columnas conocidas
            rename = {k: v for k, v in _COLUMN_MAP.items() if k in df.columns}
            df.rename(columns=rename, inplace=True)

            # Parsear coordenadas DMS → decimal
            df["lon"] = df["lon_dms"].apply(parse_dms_to_decimal) if "lon_dms" in df.columns else np.nan
            df["lat"] = df["lat_dms"].apply(parse_dms_to_decimal) if "lat_dms" in df.columns else np.nan

            # Convertir campos numéricos
            for col in ("id", "frequency_mhz", "pra_kw", "altura_m", "codigo_dane"):
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", ".") if col != "id" else df[col],
                        errors="coerce",
                    )

            # Calcular PRA en watts
            if "pra_kw" in df.columns:
                df["pra_w"] = df["pra_kw"] * 1000.0

            # Eliminar filas sin coordenadas válidas
            valid_mask = df["lat"].notna() & df["lon"].notna()
            n_dropped = (~valid_mask).sum()
            if n_dropped > 0:
                logger.warning(
                    "Se descartaron %d estaciones sin coordenadas válidas", n_dropped
                )
            df = df[valid_mask].copy()

            # Convertir ID a int donde sea posible
            if "id" in df.columns:
                df["id"] = df["id"].astype("Int64")

            self._df = df
            self._stations = self._dataframe_to_dicts(df)

            logger.info(
                "Inventario ANE cargado: %d estaciones con coordenadas válidas",
                len(self._stations),
            )

        except Exception:
            logger.exception("Error al cargar el inventario ANE")
            self._stations = []

    @staticmethod
    def _dataframe_to_dicts(df: pd.DataFrame) -> list[dict]:
        """Convierte el DataFrame a lista de dicts con campos estandarizados."""
        records: list[dict] = []
        output_cols = [
            "id", "frequency_mhz", "pra_kw", "clase", "estado", "servicio",
            "altura_m", "departamento", "municipio", "codigo_dane",
            "lat", "lon", "pra_w",
        ]
        for _, row in df.iterrows():
            rec: dict = {}
            for col in output_cols:
                val = row.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    rec[col] = None
                elif isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    rec[col] = float(val)
                else:
                    rec[col] = val
                # Asegurar int para id y pd.NA handling
                if col == "id" and rec[col] is not None:
                    try:
                        rec[col] = int(rec[col])
                    except (ValueError, TypeError):
                        rec[col] = None
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # API pública — consultas
    # ------------------------------------------------------------------

    def get_departments(self) -> list[str]:
        """Retorna lista ordenada de departamentos únicos."""
        self._ensure_loaded()
        deps = sorted({
            s["departamento"]
            for s in self._stations
            if s.get("departamento")
        })
        return deps

    def get_cities(self, department: str) -> list[str]:
        """
        Retorna lista ordenada de municipios para un departamento dado.

        Args:
            department: Nombre del departamento (case-insensitive).
        """
        self._ensure_loaded()
        dep_upper = department.strip().upper()
        cities = sorted({
            s["municipio"]
            for s in self._stations
            if s.get("departamento", "").strip().upper() == dep_upper
            and s.get("municipio")
        })
        return cities

    def get_stations(
        self,
        department: Optional[str] = None,
        city: Optional[str] = None,
        freq_mhz: Optional[float] = None,
    ) -> list[dict]:
        """
        Retorna estaciones filtradas.

        Args:
            department: Filtrar por departamento (case-insensitive).
            city: Filtrar por municipio (case-insensitive).
            freq_mhz: Filtrar por frecuencia exacta (MHz).

        Returns:
            Lista de dicts con datos de cada estación.
        """
        self._ensure_loaded()
        results = self._stations

        if department:
            dep_upper = department.strip().upper()
            results = [
                s for s in results
                if s.get("departamento", "").strip().upper() == dep_upper
            ]

        if city:
            city_upper = city.strip().upper()
            results = [
                s for s in results
                if s.get("municipio", "").strip().upper() == city_upper
            ]

        if freq_mhz is not None:
            results = [
                s for s in results
                if s.get("frequency_mhz") is not None
                and abs(s["frequency_mhz"] - freq_mhz) < 0.05
            ]

        return results

    def get_station_by_id(self, station_id: int) -> Optional[dict]:
        """
        Busca una estación por su ID del ANE.

        Args:
            station_id: ID numérico de la estación.

        Returns:
            Dict con datos de la estación, o None si no existe.
        """
        self._ensure_loaded()
        for s in self._stations:
            if s.get("id") == station_id:
                return s
        return None

    def find_cochannel(
        self,
        station_id: int,
        radius_km: float = 100.0,
    ) -> list[dict]:
        """
        Encuentra estaciones co-canal (misma frecuencia) dentro de un radio.

        Args:
            station_id: ID de la estación de referencia.
            radius_km: Radio de búsqueda en km.

        Returns:
            Lista de dicts con datos de estaciones co-canal + distancia.
        """
        self._ensure_loaded()

        ref = self.get_station_by_id(station_id)
        if not ref or ref.get("frequency_mhz") is None:
            return []

        ref_freq = ref["frequency_mhz"]
        ref_lat = ref["lat"]
        ref_lon = ref["lon"]

        conflicts: list[dict] = []
        for s in self._stations:
            # Saltar la misma estación
            if s.get("id") == station_id:
                continue

            # Solo misma frecuencia (tolerancia 0.05 MHz)
            if s.get("frequency_mhz") is None:
                continue
            if abs(s["frequency_mhz"] - ref_freq) >= 0.05:
                continue

            # Verificar distancia
            if s.get("lat") is None or s.get("lon") is None:
                continue

            dist = haversine_distance(ref_lat, ref_lon, s["lat"], s["lon"])
            if dist <= radius_km:
                entry = {**s, "distance_km": round(float(dist), 2)}
                conflicts.append(entry)

        # Ordenar por distancia
        conflicts.sort(key=lambda x: x["distance_km"])
        return conflicts

    def predict_signals_at_point(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
    ) -> list[dict]:
        """
        Estima las señales FM recibidas en un punto geográfico.

        Usa FSPL para estimar la potencia recibida de cada estación
        dentro del radio especificado.

        Args:
            lat: Latitud del punto receptor (grados decimales).
            lon: Longitud del punto receptor (grados decimales).
            radius_km: Radio de búsqueda en km.

        Returns:
            Lista de dicts con datos de estación + señal estimada,
            ordenada por potencia recibida descendente.
        """
        self._ensure_loaded()

        signals: list[dict] = []
        for s in self._stations:
            if s.get("lat") is None or s.get("lon") is None:
                continue
            if s.get("frequency_mhz") is None or s.get("pra_kw") is None:
                continue

            dist = haversine_distance(lat, lon, s["lat"], s["lon"])
            if dist > radius_km:
                continue

            # Potencia TX en dBm: PRA(kW) → PRA(W) → PRA(mW) → dBm
            pra_w = s["pra_kw"] * 1000.0
            if pra_w <= 0:
                continue
            tx_power_dbm = 10.0 * np.log10(pra_w * 1000.0)  # W → mW → dBm

            # FSPL
            dist_safe = max(dist, 0.001)  # evitar log(0)
            loss = fspl_db(dist_safe, s["frequency_mhz"])
            rx_power_dbm = float(tx_power_dbm - loss)

            # Field strength (dBμV/m)
            field_strength = rx_power_dbm + 20 * np.log10(s["frequency_mhz"]) + 77.2

            signals.append({
                **s,
                "distance_km": round(float(dist), 2),
                "rx_power_dbm": round(rx_power_dbm, 1),
                "field_strength_dbuvm": round(float(field_strength), 1),
            })

        # Ordenar por potencia recibida (mayor primero)
        signals.sort(key=lambda x: x["rx_power_dbm"], reverse=True)
        return signals


# ---------------------------------------------------------------------------
# Singleton a nivel de módulo
# ---------------------------------------------------------------------------
station_service = StationService()
