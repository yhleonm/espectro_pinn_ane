"""
Parser para archivos .vec de ATDI (ICS Telecom / HTZ Warfare).

Formato: texto plano delimitado por comas con la estructura:
    point,coordcode,x,y,size,color,ident,comment,object#,Selected,elevation,BMP_filename

Este parser es un andamiaje inicial — se refinará cuando se obtengan
archivos .vec reales del entorno colombiano.
"""

from __future__ import annotations

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False


@dataclass
class VecPoint:
    """Punto vectorial de un archivo .vec de ATDI."""
    point_type: str         # Tipo de punto/marcador
    coordcode: str          # Código de grilla/proyección
    x: float                # Longitud o easting
    y: float                # Latitud o northing
    size: int               # Tamaño del símbolo
    color: int              # Color (código numérico)
    ident: str              # Identificador único
    comment: str            # Comentario/etiqueta
    object_num: int         # Número de objeto
    selected: bool          # Seleccionado?
    elevation: float        # Elevación en metros
    bmp_filename: str       # Archivo BMP asociado (opcional)


class VecParser:
    """
    Parser para archivos .vec de ATDI.

    Uso:
        parser = VecParser("data/estaciones.vec")
        df = parser.to_dataframe()
        points = parser.points
    """

    # Nombres esperados de las columnas según el estándar ATDI
    COLUMNS = [
        "point", "coordcode", "x", "y", "size", "color",
        "ident", "comment", "object_num", "selected",
        "elevation", "bmp_filename",
    ]

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._points: Optional[List[VecPoint]] = None
        self._df: Optional[pd.DataFrame] = None

    @property
    def points(self) -> List[VecPoint]:
        """Lista de puntos vectoriales parseados."""
        if self._points is None:
            self._points = self._parse()
        return self._points

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los puntos a un DataFrame de pandas."""
        if self._df is None:
            self._df = self._parse_to_df()
        return self._df

    def _parse(self) -> List[VecPoint]:
        """Parsea el archivo .vec y retorna lista de VecPoint."""
        points = []
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                # Ignorar líneas vacías o comentarios
                if not row or (row[0].strip().startswith("#")):
                    continue

                try:
                    point = self._parse_row(row)
                    if point is not None:
                        points.append(point)
                except Exception as e:
                    print(f"  ⚠ Línea {row_num} ignorada: {e}")
                    continue

        return points

    def _parse_row(self, row: list) -> Optional[VecPoint]:
        """Parsea una fila CSV en un VecPoint."""
        # Asegurarse de tener al menos los campos mínimos
        while len(row) < 12:
            row.append("")

        return VecPoint(
            point_type=row[0].strip(),
            coordcode=row[1].strip(),
            x=_safe_float(row[2]),
            y=_safe_float(row[3]),
            size=_safe_int(row[4]),
            color=_safe_int(row[5]),
            ident=row[6].strip(),
            comment=row[7].strip(),
            object_num=_safe_int(row[8]),
            selected=row[9].strip() in ("1", "True", "true"),
            elevation=_safe_float(row[10]),
            bmp_filename=row[11].strip(),
        )

    def _parse_to_df(self) -> pd.DataFrame:
        """Parsea directamente a DataFrame para eficiencia."""
        try:
            df = pd.read_csv(
                self.filepath,
                header=None,
                names=self.COLUMNS,
                comment="#",
                on_bad_lines="warn",
            )
        except Exception:
            # Fallback: construir desde la lista de puntos
            df = pd.DataFrame([
                {
                    "point": p.point_type, "coordcode": p.coordcode,
                    "x": p.x, "y": p.y, "size": p.size, "color": p.color,
                    "ident": p.ident, "comment": p.comment,
                    "object_num": p.object_num, "selected": p.selected,
                    "elevation": p.elevation, "bmp_filename": p.bmp_filename,
                }
                for p in self.points
            ])

        return df

    def filter_by_bounds(
        self,
        lon_min: float, lat_min: float,
        lon_max: float, lat_max: float,
    ) -> pd.DataFrame:
        """Filtra puntos dentro de un bounding box."""
        df = self.to_dataframe()
        return df[
            (df["x"] >= lon_min) & (df["x"] <= lon_max) &
            (df["y"] >= lat_min) & (df["y"] <= lat_max)
        ]

    def __repr__(self) -> str:
        n = len(self.points) if self._points is not None else "?"
        return f"VecParser(file='{self.filepath.name}', points={n})"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _safe_float(value: str, default: float = 0.0) -> float:
    """Conversión segura a float."""
    try:
        return float(value.strip()) if value.strip() else default
    except (ValueError, AttributeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    """Conversión segura a int."""
    try:
        return int(float(value.strip())) if value.strip() else default
    except (ValueError, AttributeError):
        return default
