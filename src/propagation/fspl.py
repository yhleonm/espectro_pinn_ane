"""
Free Space Path Loss (FSPL) — Pérdida de propagación en espacio libre.

Modelo fundamental de propagación basado en la ecuación de Friis.
Es el punto de partida antes de aplicar modelos de terreno.

FSPL(dB) = 20·log10(d) + 20·log10(f) + 20·log10(4π/c)

Donde:
    d = distancia en metros
    f = frecuencia en Hz
    c = velocidad de la luz (299,792,458 m/s)
"""

from __future__ import annotations

import numpy as np

# Constantes físicas
C_LIGHT = 299_792_458.0  # m/s — velocidad de la luz en el vacío


def free_space_path_loss(
    distance_m: float | np.ndarray,
    frequency_hz: float,
) -> float | np.ndarray:
    """
    Calcula la pérdida de trayectoria en espacio libre (FSPL) en dB.

    Args:
        distance_m: Distancia en metros (escalar o array)
        frequency_hz: Frecuencia en Hz

    Returns:
        FSPL en dB (valor positivo = pérdida)
    """
    distance_m = np.asarray(distance_m, dtype=np.float64)
    # Evitar log(0)
    distance_m = np.maximum(distance_m, 1e-10)

    fspl = (
        20.0 * np.log10(distance_m) +
        20.0 * np.log10(frequency_hz) +
        20.0 * np.log10(4.0 * np.pi / C_LIGHT)
    )
    return fspl


def fspl_db(
    distance_km: float | np.ndarray,
    frequency_mhz: float,
) -> float | np.ndarray:
    """
    Versión simplificada con unidades prácticas para RF.

    FSPL(dB) = 32.45 + 20·log10(f_MHz) + 20·log10(d_km)

    Args:
        distance_km: Distancia en kilómetros
        frequency_mhz: Frecuencia en MHz

    Returns:
        FSPL en dB
    """
    distance_km = np.asarray(distance_km, dtype=np.float64)
    distance_km = np.maximum(distance_km, 1e-10)

    return 32.45 + 20.0 * np.log10(frequency_mhz) + 20.0 * np.log10(distance_km)


def received_power_dbm(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    distance_km: float | np.ndarray,
    frequency_mhz: float,
    additional_losses_db: float = 0.0,
) -> float | np.ndarray:
    """
    Potencia recibida usando la ecuación de Friis.

    P_rx = P_tx + G_tx + G_rx - FSPL - L_add

    Args:
        tx_power_dbm: Potencia del transmisor en dBm
        tx_gain_dbi: Ganancia de la antena transmisora (dBi)
        rx_gain_dbi: Ganancia de la antena receptora (dBi)
        distance_km: Distancia en km
        frequency_mhz: Frecuencia en MHz
        additional_losses_db: Pérdidas adicionales (cables, conectores, etc.)

    Returns:
        Potencia recibida en dBm
    """
    path_loss = fspl_db(distance_km, frequency_mhz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss - additional_losses_db


# ---------------------------------------------------------------------------
# Frecuencias de referencia para radiodifusión en Colombia
# ---------------------------------------------------------------------------

# Banda FM: 88-108 MHz
FM_BAND = {"min_mhz": 88.0, "max_mhz": 108.0, "name": "FM Broadcast"}

# Banda AM: 535-1705 kHz
AM_BAND = {"min_mhz": 0.535, "max_mhz": 1.705, "name": "AM Broadcast"}

# Banda UHF TV: 470-698 MHz (canales 14-51)
UHF_TV_BAND = {"min_mhz": 470.0, "max_mhz": 698.0, "name": "UHF TV"}

# Banda VHF TV: 174-216 MHz (canales 7-13)
VHF_TV_BAND = {"min_mhz": 174.0, "max_mhz": 216.0, "name": "VHF TV"}


def typical_fm_coverage_km(
    tx_power_watts: float,
    tx_height_m: float = 30.0,
    frequency_mhz: float = 98.0,
    sensitivity_dbm: float = -90.0,
    tx_gain_dbi: float = 3.0,
) -> float:
    """
    Estima el radio de cobertura teórico de una estación FM
    en espacio libre (sin obstáculos).

    Args:
        tx_power_watts: Potencia en watts
        tx_height_m: Altura de la antena (no usada en FSPL puro)
        frequency_mhz: Frecuencia central
        sensitivity_dbm: Sensibilidad del receptor
        tx_gain_dbi: Ganancia de antena

    Returns:
        Radio de cobertura estimado en km
    """
    tx_power_dbm = 10.0 * np.log10(tx_power_watts * 1000.0)  # W → mW → dBm

    # Resolver FSPL = P_tx + G_tx - Sensitivity
    max_path_loss = tx_power_dbm + tx_gain_dbi - sensitivity_dbm

    # max_path_loss = 32.45 + 20*log10(f) + 20*log10(d)
    # 20*log10(d) = max_path_loss - 32.45 - 20*log10(f)
    log_d = (max_path_loss - 32.45 - 20.0 * np.log10(frequency_mhz)) / 20.0
    return 10.0 ** log_d
