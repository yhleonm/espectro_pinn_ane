# Espectro IA & RF Link Analytics API

Microservicio de alto rendimiento para el análisis de perfil de terreno, despeje de Zona de Fresnel, curvatura terrestre ($k=4/3$) y pérdidas por espacio libre (FSPL) en enlaces de radiofrecuencia (RF).

## 🚀 Características Principales

- **Modelo Digital de Elevación (DEM)**: Procesamiento dinámico de datos SRTM 30m.
- **Curvatura Terrestre**: Factor de refracción atmosférica $k = 4/3$.
- **Despeje de Zona de Fresnel**: Cálculo numérico exacto de la 1a Zona de Fresnel y 60% de margen de despeje.
- **Pérdidas de Espacio Libre (FSPL)**: Matriz de atenuación en dB y potencia recibida en dBm.
- **Validado con la ANE**: Calibrado con estaciones reales de la Agencia Nacional del Espectro (Colombia).

## 🛠️ Ejecución Local

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Documentación Swagger disponible en `http://localhost:8000/docs`.

## 📦 Docker & Despliegue

```bash
docker build -t rf-link-analytics-api .
docker run -p 8000:8000 rf-link-analytics-api
```
