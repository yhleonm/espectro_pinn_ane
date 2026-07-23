import React, { useState, useEffect } from 'react';
import { getSignalAtPoint } from '../api';

const CitizenLookup = ({ onResults, mapClickPoint }) => {
  const [lat, setLat] = useState('');
  const [lon, setLon] = useState('');
  const [radius, setRadius] = useState(50);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Auto-llenar coordenadas y buscar desde clic en el mapa
  useEffect(() => {
    if (mapClickPoint) {
      setLat(mapClickPoint.lat.toFixed(6));
      setLon(mapClickPoint.lon.toFixed(6));
      // Disparar búsqueda automática
      handleSearch();
    }
  }, [mapClickPoint]);

  const getQualityIndicator = (signal) => {
    if (signal >= 66) return { icon: '🟢', label: 'Buena', color: '#10b981' };
    if (signal >= 54) return { icon: '🟡', label: 'Marginal', color: '#f59e0b' };
    return { icon: '🔴', label: 'Débil', color: '#ef4444' };
  };

  const handleSearch = async () => {
    // Usar valores locales si están disponibles, sino usar los del estado
    const currentLat = mapClickPoint ? mapClickPoint.lat : parseFloat(lat);
    const currentLon = mapClickPoint ? mapClickPoint.lon : parseFloat(lon);
    
    if (isNaN(currentLat) || isNaN(currentLon)) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await getSignalAtPoint({
        lat: currentLat,
        lon: currentLon,
        radius_km: radius
      });
      setResults(data);
      if (onResults) onResults(data.signals || data);
    } catch (e) {
      console.error('Error en búsqueda de señal:', e);
      setError('Error al consultar señales. Verifique la simulación.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '0.85rem', color: '#f8fafc' }}>
        📻 ¿Qué emisoras me llegan?
      </h3>
      
      <div className="card" style={{ padding: '10px', marginBottom: '10px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
          <div className="input-group">
            <label>Latitud</label>
            <input
              type="number"
              step="0.000001"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              placeholder="4.6097"
            />
          </div>
          <div className="input-group">
            <label>Longitud</label>
            <input
              type="number"
              step="0.000001"
              value={lon}
              onChange={(e) => setLon(e.target.value)}
              placeholder="-74.0817"
            />
          </div>
        </div>

        <div className="input-group" style={{ marginBottom: '10px' }}>
          <label>Radio de búsqueda</label>
          <select value={radius} onChange={(e) => setRadius(parseInt(e.target.value))}>
            <option value={25}>25 km</option>
            <option value={50}>50 km</option>
            <option value={100}>100 km</option>
          </select>
        </div>

        <button
          className="btn-primary"
          onClick={handleSearch}
          disabled={loading || !lat || !lon}
          style={{ background: 'linear-gradient(45deg, #10b981, #06b6d4)' }}
        >
          {loading ? '⏳ Buscando...' : '🔍 Buscar Señal'}
        </button>
      </div>

      {results && (
        <div className="glass-panel" style={{ padding: '10px', marginTop: '10px' }}>
          <div style={{ fontSize: '0.7rem', color: '#10b981', marginBottom: '8px' }}>
            {results.total_stations || (results.signals || results).length} emisoras encontradas
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.65rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                <th style={{ textAlign: 'left', padding: '4px' }}>Estación</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>Freq</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>Pot (dBm)</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>Señal (dBμV/m)</th>
                <th style={{ textAlign: 'center', padding: '4px' }}>Calidad</th>
              </tr>
            </thead>
            <tbody>
              {(results.signals || results)
                .sort((a, b) => (b.rx_power_dbm || 0) - (a.rx_power_dbm || 0))
                .map((signal, i) => {
                  const quality = getQualityIndicator(signal.field_strength_dbuvm || 0);
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b', color: quality.color }}>
                      <td style={{ padding: '4px' }}>
                        ID {signal.id || 'N/A'}
                      </td>
                      <td style={{ textAlign: 'right', padding: '4px' }}>
                        {signal.frequency_mhz}
                      </td>
                      <td style={{ textAlign: 'right', padding: '4px', color: '#fb923c', fontWeight: 'bold' }}>
                        {signal.rx_power_dbm?.toFixed(1)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '4px' }}>
                        {signal.field_strength_dbuvm?.toFixed(1)}
                      </td>
                      <td style={{ textAlign: 'center', padding: '4px' }}>
                        {quality.icon} {quality.label}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CitizenLookup;
