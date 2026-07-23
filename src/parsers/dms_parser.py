"""
Parser de coordenadas DMS (Grados, Minutos, Segundos) del inventario ANE.

Maneja el formato específico colombiano:
    - Separador decimal español (coma): "74° 3' 13,6\" W"
    - Caracteres Unicode: °, ', "
    - Hemisferio N/S/E/W como sufijo
    - Valores malformados o vacíos → None

Referencia:
    ANE — Agencia Nacional del Espectro (Colombia)
    Formato de exportación del Visor de Espectro
"""

from __future__ import annotations

import re
from typing import Optional


# Patrón regex para DMS con separador decimal español
# Captura: grados° minutos' segundos,decimales" hemisferio
_DMS_PATTERN = re.compile(
    r"""
    ^\s*
    (\d+)             # Grados (grupo 1)
    \s*[°º]\s*        # Símbolo de grados
    (\d+)             # Minutos (grupo 2)
    \s*[''′‘’]\s*     # Símbolo de minutos (incluye ’)
    ([\d,\.]+)        # Segundos con posible decimal (grupo 3)
    \s*["″“”""]?\s*   # Símbolo de segundos (incluye ”)
    ([NSEWnsew])      # Hemisferio (grupo 4)
    \s*$
    """,
    re.VERBOSE,
)


def parse_dms_to_decimal(dms_str: str) -> Optional[float]:
    """
    Convierte una cadena DMS del formato ANE colombiano a grados decimales.

    Formato esperado: '74° 3' 13,6" W'

    Args:
        dms_str: Cadena en formato DMS con separador decimal español.

    Returns:
        Grados decimales (negativo para W y S), o None si el formato
        es inválido o el valor está vacío.

    Examples:
        >>> parse_dms_to_decimal('74° 3\\'  13,6" W')
        -74.05378...
        >>> parse_dms_to_decimal('4° 42\\' 39,6" N')
        4.71100...
        >>> parse_dms_to_decimal('')
        None
        >>> parse_dms_to_decimal('INVALIDO')
        None
    """
    if not dms_str or not isinstance(dms_str, str):
        return None

    # Limpiar caracteres no estándar
    cleaned = dms_str.strip()
    if not cleaned:
        return None

    match = _DMS_PATTERN.match(cleaned)
    if not match:
        return None

    try:
        degrees = int(match.group(1))
        minutes = int(match.group(2))

        # Manejar separador decimal español (coma → punto)
        seconds_str = match.group(3).replace(",", ".")
        seconds = float(seconds_str)

        hemisphere = match.group(4).upper()

        # Validaciones de rango
        if minutes < 0 or minutes >= 60:
            return None
        if seconds < 0.0 or seconds >= 60.0:
            return None
        if degrees < 0:
            return None

        # Conversión a decimal
        decimal = degrees + minutes / 60.0 + seconds / 3600.0

        # Hemisferio Sur u Oeste → negativo
        if hemisphere in ("S", "W"):
            decimal = -decimal

        return decimal

    except (ValueError, TypeError):
        return None


def parse_dms_pair(
    lon_dms: str,
    lat_dms: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    Parsea un par de coordenadas DMS (longitud, latitud).

    Args:
        lon_dms: Longitud en formato DMS (ej: '74° 3' 13,6" W')
        lat_dms: Latitud en formato DMS (ej: '4° 42' 39,6" N')

    Returns:
        Tupla (lon_decimal, lat_decimal). Cualquiera puede ser None
        si el parseo falla.
    """
    return parse_dms_to_decimal(lon_dms), parse_dms_to_decimal(lat_dms)
