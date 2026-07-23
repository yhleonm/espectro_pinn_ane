# CONTEXTO DEL PROYECTO - ESPECTRO IA
**Hito Actual:** Calibración Matemática Definitiva (Camino B).

## Estado Técnico
- **Modelo:** PINN Residual (Helmholtz/Eikonal).
- **Benchmark:** Estación ANE 1457 (101.9 MHz, 100 kW, Cerro Antenas).
- **Física Activa:** Topografía SRTM 1.5 sigma + Clutter OSM (Building Levels).
- **Mejora Raíz:** Muestreo Adaptativo (Importance Sampling) + Warm-up de Fuente.
- **Calibración:** ALPHA Dinámico + Offset Regulatorio ANE (-10dB).
- **Métricas:** 
    - **Pearson:** 0.988 (Correlación espacial casi perfecta).
    - **RMSE:** 4.5 dB (Error absoluto calibrado).
    - **DC Offset:** ~0 dB (Relativo a baseline ANE/Knife-Edge calibrado).

## Archivos Críticos
- `src/ai/pinn_trainer.py`: Entrenamiento con Camino B y pesos dinámicos.
- `src/ai/boundary_conditions.py`: Muestreo log-uniforme cerca del TX.
- `tasks/validate_1457_metrics.py`: Script de auditoría de métricas.

## Próximo Paso
Optimización de ubicación de transmisores (TX Location Optimizer) para maximizar cobertura social en zonas críticas.
