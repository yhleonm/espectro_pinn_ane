import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path
import time

# Configuraciones del path para imports relativos
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Imports propios
from src.parsers.srtm_reader import SRTMReader, download_srtm_tile
from src.propagation.fspl import free_space_path_loss, fspl_db, received_power_dbm
from src.geo.coordinates import colombia_bbox, transform_wgs84_to_magna, haversine_distance, COLOMBIA_CITIES

# ==============================================================================
# Configuración de Streamlit
# ==============================================================================
st.set_page_config(
    page_title="RF Simulator - Perfil de Elevación",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Simulador RF: Análisis de Perfil de Terreno (Colombia)")
st.markdown("""
Este dashboard interactivo permite visualizar perfiles de terreno utilizando **datos SRTM de la NASA** y calcular pérdidas de espacio libre (FSPL) entre dos puntos geográficos en Colombia.
""")

# ==============================================================================
# Estado de la Aplicación
# ==============================================================================
if "tx_coord" not in st.session_state:
    st.session_state.tx_coord = (COLOMBIA_CITIES["Bogotá"]["lat"], COLOMBIA_CITIES["Bogotá"]["lon"])

if "rx_coord" not in st.session_state:
    st.session_state.rx_coord = (COLOMBIA_CITIES["Ibagué"]["lat"], COLOMBIA_CITIES["Ibagué"]["lon"])

# ==============================================================================
# Interfaz: Panel Lateral (Configuración)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # --- SECCIÓN DE REPORTE (FORZADA AL INICIO) ---
    st.markdown("### 📄 Reporte del Proyecto")
    
    # Botón de Generar
    if st.button("🚀 GENERAR REPORTE PDF", type="primary"):
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "tasks", "generate_ane_report.py")
        subprocess.run(["python", script_path])
        st.success("Reporte generado.")

    # Botón de Descarga siempre intentando leer el archivo
    report_path = os.path.join(os.path.dirname(__file__), "docs", "REPORTE_TECNICO_ANE_1457.pdf")
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            pdf_data = f.read()
            st.download_button(
                label="📥 DESCARGAR PDF FINAL",
                data=pdf_data,
                file_name="Reporte_Espectro_IA.pdf",
                mime="application/pdf",
                key="download_btn_main"
            )
    else:
        st.error("El archivo PDF no se encuentra. Haz clic en Generar.")

    st.markdown("---")
    
    st.subheader("1. Coordenadas (WGS-84)")

# ==============================================================================
# Procesamiento de Datos
# ==============================================================================

# 1. Distancia Básica
dist_km = haversine_distance(*st.session_state.tx_coord, *st.session_state.rx_coord)
fspl_loss = fspl_db(dist_km, frequency_mhz)
rx_power_fspl = tx_power_dbm - fspl_loss # Asumiendo ganancia 0dBi para simplificar ahora

# Dividimos en dos columnas arriba
col_metrics, col_map = st.columns([1, 2])

with col_metrics:
    st.subheader("📊 Análisis de Enlace (Espacio Libre)")
    col1, col2 = st.columns(2)
    col1.metric("Distancia Total", f"{dist_km:.2f} km")
    col2.metric("Pérdida (FSPL)", f"{fspl_loss:.2f} dB")
    
    col3, col4 = st.columns(2)
    col3.metric("Frecuencia", f"{frequency_mhz} MHz")
    # Indicador de color para nivel de señal
    signal_color = "normal" if rx_power_fspl > -90 else "inverse"
    col4.metric("RX Esperada (FSPL)", f"{rx_power_fspl:.2f} dBm", delta_color=signal_color)
    
    st.info("💡 La potencia RX mostrada asume línea de vista perfecta sin obstáculos. El perfil de elevación SRTM indicará si realmente hay línea de vista.", icon="ℹ️")

    # Mapeo a MAGNA-SIRGAS para validar
    magna_x_tx, magna_y_tx = transform_wgs84_to_magna(*st.session_state.tx_coord)
    magna_x_rx, magna_y_rx = transform_wgs84_to_magna(*st.session_state.rx_coord)
    
    with st.expander("Ver Coordenadas MAGNA-SIRGAS (Origen Bogotá)"):
        st.write(f"**TX:** Este {magna_x_tx:.1f} m, Norte {magna_y_tx:.1f} m")
        st.write(f"**RX:** Este {magna_x_rx:.1f} m, Norte {magna_y_rx:.1f} m")

# Crear pestañas para organizar las vistas
tab_1d, tab_2d = st.tabs(["📶 Análisis de Enlace (1D)", "🗺️ Mancha de Cobertura (2D)"])

with tab_1d:
    # 2. Visualización Folium (Mapa Satelital Real) - 1D
    st.subheader("🗺️ Vista de Satélite (Enlace)")
    
    import folium
    from streamlit_folium import st_folium

    # Calcular centro del mapa
    mid_lat = (st.session_state.tx_coord[0] + st.session_state.rx_coord[0]) / 2
    mid_lon = (st.session_state.tx_coord[1] + st.session_state.rx_coord[1]) / 2
    
    m_1d = folium.Map(location=[mid_lat, mid_lon], zoom_start=8)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m_1d)

    folium.PolyLine(
        locations=[
            [st.session_state.tx_coord[0], st.session_state.tx_coord[1]], 
            [st.session_state.rx_coord[0], st.session_state.rx_coord[1]]
        ],
        color='yellow',
        weight=4,
        opacity=0.8
    ).add_to(m_1d)
    
    folium.Marker(
        [st.session_state.tx_coord[0], st.session_state.tx_coord[1]],
        tooltip="Transmisor (TX)",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m_1d)
    
    folium.Marker(
        [st.session_state.rx_coord[0], st.session_state.rx_coord[1]],
        tooltip="Receptor (RX)",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m_1d)

    st_folium(m_1d, use_container_width=True, height=450, returned_objects=[], key="map_1d")

with tab_2d:
    st.subheader("🗾 Mancha de Cobertura (Estilo ICS Manager)")
    
    # Parámetros adicionales para cobertura
    radius_km = st.slider("Radio de Análisis (km)", 5, 100, 20, 5)
    res_px = st.slider("Resolución de grilla (Pixeles)", 50, 400, 100, 50)
    
    if st.button("🗺️ Calcular Heatmap 2D (FSPL)", key="btn_heatmap"):
        with st.spinner("Calculando modelo espacial vectorizado..."):
            from src.propagation.coverage import generate_fspl_coverage
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # Generar grilla de potencias
            grid = generate_fspl_coverage(
                tx_lat=st.session_state.tx_coord[0],
                tx_lon=st.session_state.tx_coord[1],
                tx_power_dbm=tx_power_dbm,
                tx_gain_dbi=3.0,
                frequency_mhz=frequency_mhz,
                radius_km=radius_km,
                resolution_points=res_px
            )
            
            # Crear mapa de colores tipo ICS (Rojo fuerte, Verde medio, Azul debil)
            cmap = plt.get_cmap('jet')
            norm = mcolors.Normalize(vmin=-120, vmax=-40) # Mapear de -120dBm a -40dBm
            
            rgba_image = cmap(norm(grid.power_dbm))
            transparency_mask = grid.power_dbm < -110
            rgba_image[transparency_mask, 3] = 0.0  # Alfa = 0
            
            m_2d = folium.Map(location=[st.session_state.tx_coord[0], st.session_state.tx_coord[1]], zoom_start=10)
            
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite'
            ).add_to(m_2d)
            
            bounds = [[grid.bounds[1], grid.bounds[0]], [grid.bounds[3], grid.bounds[2]]]
            
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=bounds,
                opacity=0.6,
                name="Cobertura FSPL",
                interactive=True,
                cross_origin=False,
                zindex=1
            ).add_to(m_2d)
            
            folium.Marker(
                [st.session_state.tx_coord[0], st.session_state.tx_coord[1]],
                tooltip=f"TX: {tx_power_dbm} dBm",
                icon=folium.Icon(color="red")
            ).add_to(m_2d)
            
            st_folium(m_2d, use_container_width=True, height=600, returned_objects=[], key="map_2d")
            st.success("✅ Heatmap 2D simulado puramente sobre Pérdida de Espacio Libre. El siguiente paso integrará oclusión topográfica (SRTM).")

    
    st.divider()
    st.subheader("🧠 Motor IA (Physics-Informed Neural Network)")
    st.info("Este módulo entrena la red SIREN Helmholtz en milisegundos usando PyTorch (CUDA) basándose en la topografía local.")
    
    col_ia1, col_ia2 = st.columns(2)
    with col_ia1:
        epochs_ia = st.number_input("Épocas de Entrenamiento", min_value=100, max_value=5000, value=500, step=100)
    
    if st.button("🚀 Entrenar PINN y Generar Heatmap", type="primary"):
        with st.status("Entrenando Physics-Informed Neural Network (PINN)...", expanded=True) as status:
            import time
            from src.ai.pinn_trainer import PINNTrainer
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            st.write("🔧 Inicializando PyTorch (CUDA)...")
            start_time = time.time()
            
            # Inicializar el entrenador principal (GPU)
            trainer = PINNTrainer(
                tx_lat=st.session_state.tx_coord[0],
                tx_lon=st.session_state.tx_coord[1],
                tx_height_m=st.session_state.tx_coord[0] + 30.0, # WIP: Necesitamos el Z real del SRTM
                tx_power_dbm=tx_power_dbm,
                frequency_mhz=frequency_mhz,
                radius_km=radius_km,
                resolution=res_px
            )
            
            st.write(f"🏃 Ejecutando {epochs_ia} iteraciones de Adam Optimizer sobre MDE Helmholtz...")
            
            # Progress bar para Streamlit
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(epochs_ia):
                metrics = trainer.train_step()
                if epoch % 50 == 0:
                    progress_bar.progress((epoch + 1) / epochs_ia)
                    status_text.text(f"Epoch {epoch}/{epochs_ia} | Loss: {metrics['loss_total']:.4e} | PDE: {metrics['loss_pde']:.4e}")
            
            progress_bar.progress(1.0)
            train_time = time.time() - start_time
            st.write(f"✅ PINN Convergida en {train_time:.2f} segundos.")
            
            st.write("🗺️ Infiriendo el campo electromagnético en la grilla y renderizando folio...")
            # Extraer mapa de calor infiriendo a través de la IA
            ai_grid = trainer.infer_grid().numpy()
            
            # Normalizar los valores de la IA (-V/m a V/m usualmente) hacia a pseudocolores.
            # Convertimos la Amplitud E en dBm aproxcimados para mapeo de color visual
            power_w = (np.abs(ai_grid) ** 2) / 120 * np.pi
            # Evitar log 0
            power_w = np.clip(power_w, a_min=1e-18, a_max=None)
            ai_power_dbm = 10 * np.log10(power_w * 1000)
            
            # Mapeo de Colores ICS
            cmap = plt.get_cmap('jet')
            norm = mcolors.Normalize(vmin=-120, vmax=-40)
            rgba_image = cmap(norm(ai_power_dbm))
            transparency_mask = ai_power_dbm < -110
            rgba_image[transparency_mask, 3] = 0.0
            
            m_pinn = folium.Map(location=[st.session_state.tx_coord[0], st.session_state.tx_coord[1]], zoom_start=10)
            
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite'
            ).add_to(m_pinn)
            
            bounds = [
                [trainer.boundary.lats_1d.min(), trainer.boundary.lons_1d.min()], 
                [trainer.boundary.lats_1d.max(), trainer.boundary.lons_1d.max()]
            ]
            
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=bounds,
                opacity=0.6,
                name="Cobertura IA (PINN)",
                interactive=True,
                cross_origin=False,
                zindex=1
            ).add_to(m_pinn)
            
            folium.Marker(
                [st.session_state.tx_coord[0], st.session_state.tx_coord[1]],
                tooltip=f"TX: {tx_power_dbm} dBm",
                icon=folium.Icon(color="green", icon="flash")
            ).add_to(m_pinn)
            
            status.update(label="Predicción Lista", state="complete", expanded=False)
            
        st_folium(m_pinn, use_container_width=True, height=600, returned_objects=[], key="map_pinn")

# ==============================================================================
# Perfil de Elevación SRTM
# ==============================================================================
st.markdown("---")
st.subheader("⛰️ Perfil Topográfico SRTM")

def resolve_srtm_tiles(lat1, lon1, lat2, lon2):
    """Obtiene los nombres de tiles SRTM requeridos para la línea."""
    # Para la demo, simplemente obtenemos el tile de lat/lon de midpoint o descargamos el principal
    # En un sistema real se iteraría sobre la línea de intersección.
    # Por ahora, usamos una interpolación simple y determinamos los tiles requeridos.
    lats = np.linspace(lat1, lat2, 5)
    lons = np.linspace(lon1, lon2, 5)
    tiles = set()
    for lat, lon in zip(lats, lons):
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        latin = int(np.floor(abs(lat)))
        lonin = int(np.floor(abs(lon)))
        tiles.add((lat, lon, f"{ns}{latin:02d}{ew}{lonin:03d}"))
    return tiles

# Crear la estructura de carpetas si no existe
data_dir = Path("data/srtm")
data_dir.mkdir(parents=True, exist_ok=True)

if st.button("🚀 Descargar Datos y Generar Perfil"):
    with st.spinner("Determinando tiles SRTM necesarios..."):
        required_tiles = resolve_srtm_tiles(*st.session_state.tx_coord, *st.session_state.rx_coord)
        st.write(f"Tiles requeridos: {', '.join([t[2] for t in required_tiles])}")
        
        readers = []
        progress_bar = st.progress(0)
        
        for i, (lat, lon, tile_name) in enumerate(required_tiles):
            st.text(f"Preparando {tile_name}...")
            # Aquí idealmente se descargarían. Para la demo, asumimos que se tienen local o 
            # se intenta descargar desde la función que creamos.
            # En V3 SRTM3 90m (la más rápida para Demos online)
            try:
                # Usa SRTM3 open para facilitar la prueba a Yerson
                tile_path = download_srtm_tile(int(np.floor(lat)), int(np.floor(lon)), output_dir=data_dir)
                readers.append(SRTMReader(tile_path))
            except Exception as e:
                st.error(f"Fallo descargando {tile_name}: {e}. ¿Sin conexión a Internet?")
            
            progress_bar.progress((i + 1) / len(required_tiles))
            
    if readers:
        with st.spinner("Interpolando perfil de elevación..."):
            # Generamos perfiles de todos y promediamos/unimos
            # Forma simplificada para la demo: se lee del primer lector válido para cada punto
            lats = np.linspace(st.session_state.tx_coord[0], st.session_state.rx_coord[0], n_points_profile)
            lons = np.linspace(st.session_state.tx_coord[1], st.session_state.rx_coord[1], n_points_profile)
            
            elevations = []
            valid_distances = []
            
            # Cálculo de distancia acumulada
            dist_array = np.zeros(n_points_profile)
            for i in range(1, n_points_profile):
                dist_array[i] = dist_array[i-1] + haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
                
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                z = None
                # Busca qué lector tiene el punto
                for reader in readers:
                    info = reader.info
                    if info.lat_sw <= lat <= info.lat_ne and info.lon_sw <= lon <= info.lon_ne:
                        try:
                            z = reader.get_elevation(lat, lon)
                        except Exception:
                            pass
                        break
                
                if z is not None and not np.isnan(z):
                    elevations.append(z)
                    valid_distances.append(dist_array[i])
                else:
                    # Rellenar con interpolación simple u 0 si no hay tile
                    elevations.append(0.0)
                    valid_distances.append(dist_array[i])

            # Agregar alturas de antenas al primer y último punto
            tx_z = elevations[0] + tx_height
            rx_z = elevations[-1] + rx_height

            # Dibujar con Plotly
            fig = go.Figure()

            # Área del terreno
            fig.add_trace(go.Scatter(
                x=valid_distances,
                y=elevations,
                fill='tozeroy',
                mode='lines',
                line=dict(color='SaddleBrown'),
                name='Terreno (SRTM)'
            ))

            # Línea de vista (LoS)
            fig.add_trace(go.Scatter(
                x=[valid_distances[0], valid_distances[-1]],
                y=[tx_z, rx_z],
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dash'),
                name='Línea de Vista (LoS)',
                marker=dict(size=8, color=['red', 'blue'])
            ))

            fig.update_layout(
                title=f"Perfil Topográfico ({dist_km:.1f} km)",
                xaxis_title="Distancia (km)",
                yaxis_title="Elevación (m.s.n.m)",
                template="plotly_dark",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("¡Perfil generado! En un motor completo, aquí se calcularía la zona de Fresnel y atenuación por difracción de aristas sobre este perfil.")
    else:
        st.warning("Pulsa el botón para generar el perfil de terreno con datos SRTM.")
