import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, ImageOverlay, useMap, useMapEvents, GeoJSON, CircleMarker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix para los iconos de Leaflet en Vite/React
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

const txIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const optIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [20, 32],
  iconAnchor: [10, 32],
  popupAnchor: [1, -30],
  shadowSize: [32, 32]
});

const blueIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [20, 32],
  iconAnchor: [10, 32],
  popupAnchor: [1, -30],
  shadowSize: [32, 32]
});

const greenIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const rxIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-gold.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

// Componente para auto-ajustar vista del mapa
function ChangeView({ bounds }) {
  const map = useMap();
  React.useEffect(() => {
    if (bounds && bounds.length === 2) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [bounds, map]);
  return null;
}

// Componente para centrar mapa dinámicamente
function MapCenterUpdater({ center }) {
  const map = useMap();
  React.useEffect(() => {
    if (center && center.lat && center.lon) {
      map.setView([center.lat, center.lon], 11, { animate: true });
    }
  }, [center, map]);
  return null;
}

// Componente para capturar clics en el mapa
function MapClickHandler({ onClick }) {
  useMapEvents({
    click: (e) => {
      if (onClick) onClick({ lat: e.latlng.lat, lon: e.latlng.lng });
    }
  });
  return null;
}

const MapControl = ({ 
  tx, rx, mode, coverageData, socialData, optResults,
  mapCenter, interferenceStations, citizenPoint, citizenSignals,
  onMapClick, onRxChange, stationInfo, measurements, detectedTx
}) => {
  const defaultLat = mapCenter?.lat || tx?.lat || 4.6097;
  const defaultLon = mapCenter?.lon || tx?.lon || -74.0817;

  const getSocialStyle = (feature) => {
    const coverage = feature.properties['Cobertura (%)'] || 0;
    let color = '#ef4444'; // Rojo (Déficit)
    if (coverage > 50) color = '#10b981'; // Verde
    return { fillColor: color, weight: 0.5, opacity: 1, color: 'white', fillOpacity: 0.4 };
  };

  const onEachFeature = (feature, layer) => {
    layer.bindPopup(`<strong>Barrio:</strong> ${feature.properties.SCANOMBRE || 'N/A'}`);
  };

  // Popup dinámico para el transmisor
  const txPopupContent = stationInfo
    ? `Transmisor ID ${stationInfo.id}<br/>${stationInfo.frequency_mhz} MHz — ${stationInfo.pra_kw} kW<br/>${stationInfo.municipio || ''}, ${stationInfo.departamento || ''}`
    : 'Transmisor Principal';

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <MapContainer center={[defaultLat, defaultLon]} zoom={11} style={{ height: '100%', width: '100%', background:'#020617' }}>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />

        {/* Centrado dinámico del mapa */}
        {mapCenter && <MapCenterUpdater center={mapCenter} />}

        {/* Captura de clics en el mapa */}
        {onMapClick && <MapClickHandler onClick={onMapClick} />}

        {/* 2D Coverage Mode Overlay (RESTORED GRADIENT) */}
        {coverageData && coverageData.image_base64 && coverageData.bounds && (
          <ImageOverlay url={coverageData.image_base64} bounds={coverageData.bounds} opacity={0.7} zIndex={1000} />
        )}

        {/* Social Mode GeoJSON */}
        {mode === 'social' && socialData && (
          <GeoJSON data={socialData} style={getSocialStyle} onEachFeature={onEachFeature} />
        )}

        {/* Primary TX Marker */}
        <Marker position={[tx.lat, tx.lon]} icon={txIcon}>
          <Popup><span dangerouslySetInnerHTML={{ __html: txPopupContent }} /></Popup>
        </Marker>
        
        {/* RX Marker (Draggable for profile) */}
        <Marker 
          position={[rx.lat, rx.lon]} 
          icon={rxIcon} 
          draggable={true}
          eventHandlers={{
            dragend: (e) => {
              const marker = e.target;
              const position = marker.getLatLng();
              if (onRxChange) onRxChange({ lat: position.lat, lon: position.lng });
            },
          }}
        >
          <Popup>🎯 Receptor (Perfil 1D)<br/>Arrastra para cambiar el punto</Popup>
        </Marker>

        {/* Interference Station Markers (blue) */}
        {interferenceStations && interferenceStations.map((station, i) => (
          <Marker key={`interf-${i}`} position={[station.lat, station.lon]} icon={blueIcon}>
            <Popup>
              📡 Co-Canal ID {station.station_id}<br/>
              {station.frequency_mhz} MHz<br/>
              Dist: {station.distance_km?.toFixed(1)} km<br/>
              C/I: {station.ci_ratio_db?.toFixed(1)} dB
            </Popup>
          </Marker>
        ))}

        {/* Citizen Point Marker (green) */}
        {citizenPoint && (
          <Marker position={[citizenPoint.lat, citizenPoint.lon]} icon={greenIcon}>
            <Popup>📍 Punto de consulta<br/>{citizenPoint.lat.toFixed(4)}, {citizenPoint.lon.toFixed(4)}</Popup>
          </Marker>
        )}

        {/* Marcadores de Medición (Detección Ilegal) */}
        {measurements && measurements.map((m, i) => (
          <CircleMarker 
            key={`meas-${i}`}
            center={[m.lat, m.lon]}
            radius={8}
            pathOptions={{ color: '#f97316', fillColor: '#f97316', fillOpacity: 0.8 }}
          >
            <Popup>
              <strong>Medición #{i+1}</strong><br/>
              Frecuencia: {m.frequency_mhz ? `${m.frequency_mhz} MHz` : 'N/A'}<br/>
              Nivel: {m.dbm} dBm<br/>
              Incertidumbre: ±{m.uncertainty_db !== undefined ? m.uncertainty_db : 2.0} dB
            </Popup>
          </CircleMarker>
        ))}

        {/* Mapa de Calor de Probabilidad / Incertidumbre */}
        {detectedTx && detectedTx.heatmap_base64 && detectedTx.bounds && (
          <ImageOverlay 
            url={detectedTx.heatmap_base64} 
            bounds={detectedTx.bounds} 
            opacity={0.65} 
            zIndex={999} 
          />
        )}

        {/* Ajuste dinámico de vista a la zona de incertidumbre */}
        {detectedTx && detectedTx.bounds && (
          <ChangeView bounds={detectedTx.bounds} />
        )}

        {/* Transmisor Sospechoso Detectado */}
        {detectedTx && (
          <Marker position={[detectedTx.lat, detectedTx.lon]} icon={L.divIcon({
            className: 'custom-icon',
            html: '<div style="font-size: 30px; filter: drop-shadow(0 0 10px red);">☠️</div>',
            iconSize: [40, 40],
            iconAnchor: [20, 20]
          })}>
            <Popup>
              <strong>Transmisor Sospechoso</strong><br/>
              Lat: {detectedTx.lat.toFixed(6)}<br/>
              Lon: {detectedTx.lon.toFixed(6)}<br/>
              Potencia: {detectedTx.power_dbm.toFixed(1)} dBm
            </Popup>
          </Marker>
        )}

        {/* Optimization Results */}
        {optResults && optResults.map(r => (
          <Marker key={r.id} position={[r.lat, r.lon]} icon={optIcon}>
            <Popup>🎯 Punto Óptimo {r.id}<br/>Potencia: {r.pra_w.toFixed(1)}W</Popup>
          </Marker>
        ))}

        {coverageData && coverageData.bounds && <ChangeView bounds={coverageData.bounds} />}
      </MapContainer>
    </div>
  );
};

export default MapControl;
