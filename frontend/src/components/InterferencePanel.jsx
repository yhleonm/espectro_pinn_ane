import React, { useState } from 'react';
import { getCochannelInterference } from '../api';

const InterferencePanel = ({ station, onAnalyze }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!station) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getCochannelInterference({ station_id: station.id });
      setResults(data);
      if (onAnalyze) onAnalyze(data);
    } catch (e) {
      console.error('Error en análisis de interferencia:', e);
      setError('Error al analizar interferencia. Verifique la conexión.');
    } finally {
      setLoading(false);
    }
  };

  const getCIColor = (ci) => {
    if (ci >= 30) return '#10b981'; // Verde - sin interferencia
    if (ci >= 20) return '#f59e0b'; // Amarillo - moderada
    return '#ef4444'; // Rojo - interferencia significativa
  };

  const getCILabel = (ci) => {
    if (ci >= 30) return '✅ Limpio';
    if (ci >= 20) return '⚠️ Moderada';
    return '🔴 Crítica';
  };

  if (!station) {
    return (
      <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.8rem', padding: '2rem' }}>
        Seleccione una estación para analizar interferencia co-canal
      </div>
    );
  }

  return (
    <div>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '0.85rem', color: '#f8fafc' }}>
        📡 Interferencia Co-Canal
      </h3>
      <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginBottom: '10px' }}>
        Estación ID {station.id} — {station.frequency_mhz} MHz — {station.departamento}, {station.municipio}
      </div>

      <button
        className="btn-primary"
        onClick={handleAnalyze}
        disabled={loading}
        style={{ background: 'linear-gradient(45deg, #3b82f6, #8b5cf6)', marginBottom: '12px' }}
      >
        {loading ? '⏳ Analizando...' : '🔍 Analizar Interferencia Co-Canal'}
      </button>

      {error && (
        <div style={{ color: '#ef4444', fontSize: '0.75rem', marginBottom: '8px' }}>
          {error}
        </div>
      )}

      {results && results.conflicts && (
        <div>
          <div style={{ fontSize: '0.7rem', color: '#10b981', marginBottom: '8px' }}>
            {results.conflicts.length} estaciones co-canal encontradas en {station.frequency_mhz} MHz
            {results.summary && ` | Rango: ${results.summary.min_distance_km?.toFixed(1)} - ${results.summary.max_distance_km?.toFixed(1)} km`}
          </div>

          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.65rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                <th style={{ textAlign: 'left', padding: '4px' }}>ID</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>Freq (MHz)</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>Dist (km)</th>
                <th style={{ textAlign: 'right', padding: '4px' }}>C/I (dB)</th>
                <th style={{ textAlign: 'center', padding: '4px' }}>Estado</th>
              </tr>
            </thead>
            <tbody>
              {results.conflicts.map((conflict, i) => (
                <tr
                  key={i}
                  style={{
                    borderBottom: '1px solid #1e293b',
                    color: getCIColor(conflict.ci_ratio_db)
                  }}
                >
                  <td style={{ padding: '4px' }}>{conflict.station_id}</td>
                  <td style={{ textAlign: 'right', padding: '4px' }}>{conflict.frequency_mhz}</td>
                  <td style={{ textAlign: 'right', padding: '4px' }}>{conflict.distance_km?.toFixed(1)}</td>
                  <td style={{ textAlign: 'right', padding: '4px', fontWeight: 'bold' }}>
                    {conflict.ci_ratio_db?.toFixed(1)}
                  </td>
                  <td style={{ textAlign: 'center', padding: '4px' }}>
                    {getCILabel(conflict.ci_ratio_db)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {results.conflicts.length === 0 && (
            <div style={{ textAlign: 'center', color: '#10b981', padding: '1rem', fontSize: '0.8rem' }}>
              ✅ No se encontraron estaciones co-canal conflictivas
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InterferencePanel;
