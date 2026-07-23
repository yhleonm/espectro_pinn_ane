"""
Modelos Pydantic v2 para la API de Análisis de Enlaces RF y Perfil de Terreno.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class AntennaSite(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitud en grados decimales (-90 a 90)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitud en grados decimales (-180 a 180)")
    height_m: float = Field(default=10.0, ge=0.0, le=500.0, description="Altura de la antena sobre el nivel del suelo en metros")
    power_dbm: Optional[float] = Field(default=43.0, description="Potencia del transmisor en dBm (Requerido para Tx)")
    gain_dbi: Optional[float] = Field(default=0.0, description="Ganancia de la antena en dBi")


class LinkAnalysisRequest(BaseModel):
    tx: AntennaSite = Field(..., description="Datos del sitio Transmisor (Tx)")
    rx: AntennaSite = Field(..., description="Datos del sitio Receptor (Rx)")
    frequency_mhz: float = Field(..., gt=0.0, le=100000.0, description="Frecuencia de operación en MHz")
    n_samples: int = Field(default=100, ge=10, le=500, description="Número de puntos a muestrear a lo largo del perfil")

    @field_validator('n_samples')
    def check_samples(cls, v: int) -> int:
        if v < 10 or v > 500:
            raise ValueError('n_samples debe estar entre 10 y 500')
        return v


class TerrainPoint(BaseModel):
    distance_km: float
    lat: float
    lon: float
    elevation_m: float
    tx_line_height_m: float
    fresnel_60_radius_m: float
    is_obstructed: bool


class LinkAnalysisSummary(BaseModel):
    distance_km: float
    frequency_mhz: float
    fspl_db: float
    tx_elevation_m: float
    rx_elevation_m: float
    max_terrain_elevation_m: float
    min_fresnel_clearance_m: float
    min_fresnel_clearance_percent: float
    los_blocked: bool
    status: str = Field(..., description="'CLEAR', 'MARGINAL', o 'BLOCKED'")
    received_power_dbm: Optional[float] = None


class LinkAnalysisResponse(BaseModel):
    summary: LinkAnalysisSummary
    profile: List[TerrainPoint]
