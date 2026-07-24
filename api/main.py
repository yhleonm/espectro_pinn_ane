"""
Aplicación FastAPI Principal — RF Link & Terrain Analytics API.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import time
import sys
import os

# Asegurar importación desde la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from api.schemas import LinkAnalysisRequest, LinkAnalysisResponse
from api.services.link_service import LinkService

app = FastAPI(
    title="📡 RF Link & Terrain Analytics API",
    description=(
        "API REST para análisis de factibilidad de enlaces de microondas y radiodifusión RF.\n\n"
        "Calcula perfiles de terreno con elevación SRTM (NASA), pérdidas por espacio libre (FSPL), "
        "curvatura terrestre y porcentaje de despeje de la 1ª Zona de Fresnel (60%).\n\n"
        "Validado y auditado con datos de planificación espectral (ANE Colombia)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicio
link_service = LinkService()


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Verifica la salud de la API."""
    return {
        "status": "online",
        "service": "RF Link & Terrain Analytics API",
        "version": "1.0.0"
    }


@app.get("/debug-srtm", tags=["Health"])
def debug_srtm():
    import traceback
    try:
        elev = link_service._get_elevation(4.6097, -74.0817)
        files = [str(p.name) for p in link_service.srtm_dir.glob("*")]
        return {
            "elevation": elev,
            "srtm_dir": str(link_service.srtm_dir),
            "dir_exists": link_service.srtm_dir.exists(),
            "files": files
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post(
    "/api/v1/link-analysis",
    response_model=LinkAnalysisResponse,
    tags=["RF Analytics"],
    summary="Analiza la factibilidad y perfil de terreno de un enlace RF",
    status_code=status.HTTP_200_OK
)
def analyze_rf_link(request: LinkAnalysisRequest):
    """
    Realiza el análisis completo de un enlace de radiofrecuencia entre dos coordenadas Tx y Rx:
    
    - **Perfil de terreno:** Muestreo de altitudes SRTM.
    - **Pérdida por Espacio Libre (FSPL):** Modelo exacto en dB.
    - **Despeje de Zona de Fresnel:** Cálculo del elipsoide del 60% de despeje.
    - **Detección de Obstrucciones:** Verificación de Línea de Vista (LOS).
    """
    start_time = time.time()
    try:
        response = link_service.analyze_link(request)
        elapsed_ms = (time.time() - start_time) * 1000.0
        # Añadir header de tiempo de procesamiento en logs si es necesario
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la solicitud de enlace: {str(e)}"
        )
