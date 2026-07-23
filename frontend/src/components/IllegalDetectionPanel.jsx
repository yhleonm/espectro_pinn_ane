import React, { useState } from 'react';
import { detectIllegal } from '../api';

const IllegalDetectionPanel = ({ tx, onDetectionResults, onMeasurementsChange, mapClickPoint }) => {
  const [measurements, setMeasurements] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  
  // Estados para el formulario de Nueva Medición
  const [freq, setFreq] = useState('90.9');
  const [manualLat, setManualLat] = useState('');
  const [manualLon, setManualLon] = useState('');
  const [manualDbm, setManualDbm] = useState('-70');
  const [manualUncertainty, setManualUncertainty] = useState('2.0');

  // Estado para la cacería activa
  const [targetFreq, setTargetFreq] = useState('');

  // Sincronizar con el mapa cada vez que cambian las mediciones locales
  React.useEffect(() => {
    if (onMeasurementsChange) {
      onMeasurementsChange(measurements);
    }
  }, [measurements]);
  
  // Actualizar campos manuales cuando se hace clic en el mapa
  React.useEffect(() => {
    if (mapClickPoint) {
      setManualLat(mapClickPoint.lat.toFixed(6));
      setManualLon(mapClickPoint.lon.toFixed(6));
    }
  }, [mapClickPoint]);

  // Obtener frecuencias únicas registradas
  const uniqueFrequencies = React.useMemo(() => {
    const freqs = measurements.map(m => m.frequency_mhz);
    return [...new Set(freqs)].sort((a, b) => a - b);
  }, [measurements]);

  // Sincronizar la frecuencia objetivo seleccionada
  React.useEffect(() => {
    if (uniqueFrequencies.length > 0 && (!targetFreq || !uniqueFrequencies.includes(parseFloat(targetFreq)))) {
      setTargetFreq(uniqueFrequencies[0].toString());
    }
  }, [uniqueFrequencies, targetFreq]);

  // Filtrar mediciones para la frecuencia seleccionada para cacería
  const filteredMeasurements = React.useMemo(() => {
    if (!targetFreq) return [];
    const tFreq = parseFloat(targetFreq);
    return measurements.filter(m => Math.abs(m.frequency_mhz - tFreq) < 0.01);
  }, [measurements, targetFreq]);

  const addMeasurement = () => {
    const lat = parseFloat(manualLat);
    const lon = parseFloat(manualLon);
    const dbm = parseFloat(manualDbm);
    const frequency_mhz = parseFloat(freq);
    const uncertainty_db = parseFloat(manualUncertainty);

    if (isNaN(lat) || isNaN(lon)) {
      setErrorMsg("La latitud y longitud deben ser números válidos.");
      return;
    }
    if (isNaN(dbm)) {
      setErrorMsg("El nivel medido debe ser un número válido.");
      return;
    }
    if (isNaN(frequency_mhz) || frequency_mhz <= 0) {
      setErrorMsg("La frecuencia debe ser un número positivo válido.");
      return;
    }
    if (isNaN(uncertainty_db) || uncertainty_db <= 0) {
      setErrorMsg("La incertidumbre debe ser mayor a 0 dB.");
      return;
    }

    setErrorMsg(null);
    setMeasurements([...measurements, { 
      lat, 
      lon, 
      dbm, 
      frequency_mhz, 
      uncertainty_db 
    }]);
    
    // Mantener coordenadas para registro rápido continuo, reiniciar nivel medido
    setManualDbm('-70');
  };

  const removeMeasurement = (index) => {
    setMeasurements(measurements.filter((_, i) => i !== index));
  };

  const clearAllMeasurements = () => {
    setMeasurements([]);
    setResults(null);
    setErrorMsg(null);
  };

  const handleLaunchHunt = async () => {
    if (filteredMeasurements.length < 4) {
      setErrorMsg(`Se requieren al menos 4 puntos para la frecuencia ${targetFreq} MHz para realizar un análisis WLS estadístico.`);
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    setResults(null);
    
    try {
      console.log(`Lanzando cacería WLS para ${targetFreq} MHz con:`, filteredMeasurements);
      const res = await detectIllegal({
        measurements: filteredMeasurements,
        frequency_mhz: parseFloat(targetFreq),
        search_center: { lat: tx.lat, lon: tx.lon },
        radius_km: 15.0
      });
      
      if (res && res.detected_tx) {
        setResults(res);
        onDetectionResults(res, filteredMeasurements);
      } else {
        setErrorMsg("El servidor no devolvió resultados válidos.");
      }
    } catch (error) {
      console.error("Error en detección:", error);
      setErrorMsg(`Error del Servidor: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel p-4" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="text-xl font-bold" style={{ margin: 0 }}>🕵️ Cacería de Transmisores Ilegales</h3>
        {measurements.length > 0 && (
          <button 
            onClick={clearAllMeasurements}
            style={{ fontSize: '0.7rem', padding: '2px 8px', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '4px', color: '#f87171', cursor: 'pointer' }}
          >
            Limpiar Todo
          </button>
        )}
      </div>
      
      {/* FORMULARIO DE ENTRADA MANUAL / CLICK MAPA */}
      <div className="card p-3 bg-white/5 border-orange-500/30" style={{ borderRadius: '8px', border: '1px solid rgba(249, 115, 22, 0.2)', background: 'rgba(255,255,255,0.02)' }}>
        <h4 className="text-xs font-bold uppercase text-orange-400 mb-2" style={{ margin: '0 0 8px 0' }}>Registrar Medición en Campo</h4>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '8px' }}>
          <div className="input-group">
            <label className="text-[10px]" style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>Latitud</label>
            <input 
              type="number" 
              value={manualLat} 
              onChange={e => setManualLat(e.target.value)} 
              step="0.000001" 
              placeholder="Haz clic en el mapa"
              style={{ width: '100%', padding: '4px 8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: 'white' }} 
            />
          </div>
          <div className="input-group">
            <label className="text-[10px]" style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>Longitud</label>
            <input 
              type="number" 
              value={manualLon} 
              onChange={e => setManualLon(e.target.value)} 
              step="0.000001" 
              placeholder="Haz clic en el mapa"
              style={{ width: '100%', padding: '4px 8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: 'white' }} 
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
          <div className="input-group">
            <label className="text-[10px]" style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>Frecuencia (MHz)</label>
            <input 
              type="number" 
              value={freq} 
              onChange={e => setFreq(e.target.value)} 
              step="0.1" 
              style={{ width: '100%', padding: '4px 8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: 'white' }} 
            />
          </div>
          <div className="input-group">
            <label className="text-[10px]" style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>Nivel (dBm)</label>
            <input 
              type="number" 
              value={manualDbm} 
              onChange={e => setManualDbm(e.target.value)} 
              style={{ width: '100%', padding: '4px 8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: 'white' }} 
            />
          </div>
        </div>

        <div className="input-group mb-3">
          <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>
            Incertidumbre Receptor (dB) — <span style={{ color: '#fb923c' }}>Standard: ±2.0 dB (ITU-R)</span>
          </label>
          <select 
            value={manualUncertainty} 
            onChange={e => setManualUncertainty(e.target.value)}
            style={{ width: '100%', padding: '4px 8px', fontSize: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', color: 'white', cursor: 'pointer' }}
          >
            <option value="0.5" style={{ background: '#1e293b' }}>±0.5 dB — Calibrado Alta Gama / Estación de Monitoreo</option>
            <option value="1.0" style={{ background: '#1e293b' }}>±1.0 dB — Receptor Portátil Profesional</option>
            <option value="2.0" style={{ background: '#1e293b' }}>±2.0 dB — Estándar ITU-R SM.378 (Receptor Portátil)</option>
            <option value="3.0" style={{ background: '#1e293b' }}>±3.0 dB — Dongle SDR (RTL-SDR sin calibrar)</option>
            <option value="4.0" style={{ background: '#1e293b' }}>±4.0 dB — Entorno con alto ruido / Obstáculos densos</option>
          </select>
        </div>

        <button 
          onClick={addMeasurement}
          className="btn-secondary w-full py-1 text-sm bg-orange-600/20 border-orange-500/50 hover:bg-orange-600/40"
          style={{ width: '100%', padding: '6px', background: 'rgba(249, 115, 22, 0.15)', border: '1px solid rgba(249, 115, 22, 0.4)', borderRadius: '4px', color: '#ffedd5', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}
        >
          ➕ Registrar Punto de Medición
        </button>
      </div>

      {/* DETECCIÓN Y CAZERÍA MULTIFRECUENCIA ACTIVADA */}
      {uniqueFrequencies.length > 0 && (
        <div style={{ padding: '12px', background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '8px' }}>
          <h4 className="text-xs font-bold uppercase text-blue-400 mb-2" style={{ margin: '0 0 8px 0' }}>Objetivo de Cacería IA (WLS)</h4>
          <div className="input-group mb-2">
            <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>Frecuencia Sospechosa a Localizar</label>
            <select 
              value={targetFreq} 
              onChange={e => {
                setTargetFreq(e.target.value);
                setResults(null);
              }}
              style={{ width: '100%', padding: '6px 8px', fontSize: '13px', background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(59, 130, 246, 0.4)', borderRadius: '6px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}
            >
              {uniqueFrequencies.map(f => (
                <option key={f} value={f} style={{ background: '#0f172a' }}>{f} MHz ({measurements.filter(m => Math.abs(m.frequency_mhz - f) < 0.01).length} puntos)</option>
              ))}
            </select>
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', marginTop: '6px' }}>
            <span style={{ color: filteredMeasurements.length >= 4 ? '#4ade80' : '#f87171' }}>
              Puntos activos: <strong>{filteredMeasurements.length}</strong> / 4 requeridos
            </span>
            {filteredMeasurements.length > 0 && (
              <span style={{ opacity: 0.6 }}>
                (Unidades WLS: {(100 / filteredMeasurements.length).toFixed(0)}% por punto)
              </span>
            )}
          </div>
        </div>
      )}

      {/* POOL DE MEDICIONES REGISTRADAS */}
      <div className="space-y-1 mb-2" style={{ flex: '1 1 auto', overflowY: 'auto', maxHeight: '180px' }}>
        <p className="text-[10px] uppercase opacity-50 mb-1" style={{ fontSize: '10px', fontWeight: 'bold', margin: '0 0 4px 0' }}>
          Pool General de Mediciones ({measurements.length})
        </p>
        
        {measurements.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '16px 0', color: '#64748b', fontSize: '11px', border: '1px dashed rgba(255,255,255,0.05)', borderRadius: '6px' }}>
            No hay puntos. Haz clic en el mapa y presiona Registrar.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {measurements.map((m, i) => {
              const isActive = targetFreq && Math.abs(m.frequency_mhz - parseFloat(targetFreq)) < 0.01;
              return (
                <div 
                  key={i} 
                  className="flex gap-2 items-center rounded text-[11px]" 
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between', 
                    background: isActive ? 'rgba(249, 115, 22, 0.1)' : 'rgba(0, 0, 0, 0.2)', 
                    padding: '6px 8px', 
                    borderRadius: '4px',
                    border: isActive ? '1px solid rgba(249, 115, 22, 0.3)' : '1px solid rgba(255,255,255,0.03)',
                    opacity: isActive ? 1.0 : 0.45,
                    transition: 'opacity 0.2s, border 0.2s'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ color: isActive ? '#f97316' : '#94a3b8', fontWeight: 'bold' }}>#{i+1}</span>
                    <span style={{ color: '#cbd5e1' }}>
                      {m.lat.toFixed(5)}, {m.lon.toFixed(5)} @ <strong>{m.frequency_mhz} MHz</strong>
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: isActive ? '#fbcfe8' : '#cbd5e1', fontWeight: 'bold' }}>{m.dbm} dBm</span>
                    <span style={{ fontSize: '9px', background: 'rgba(255,255,255,0.08)', padding: '1px 4px', borderRadius: '3px', color: '#94a3b8' }}>
                      ±{m.uncertainty_db} dB
                    </span>
                    <button 
                      onClick={() => removeMeasurement(i)} 
                      style={{ background: 'none', border: 'none', color: '#f87171', cursor: 'pointer', padding: '0 2px', fontSize: '12px' }}
                      title="Eliminar"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <button 
        onClick={handleLaunchHunt}
        disabled={loading || filteredMeasurements.length < 4}
        className={`w-full py-3 rounded-lg font-bold transition-all`}
        style={{ 
          width: '100%', 
          padding: '12px', 
          borderRadius: '8px', 
          border: 'none', 
          fontWeight: 'bold', 
          cursor: filteredMeasurements.length < 4 ? 'not-allowed' : 'pointer',
          background: filteredMeasurements.length < 4 ? '#334155' : 'linear-gradient(135deg, #ef4444 0%, #f97316 100%)',
          color: filteredMeasurements.length < 4 ? '#64748b' : 'white',
          opacity: filteredMeasurements.length < 4 ? 0.6 : 1.0,
          boxShadow: filteredMeasurements.length < 4 ? 'none' : '0 4px 12px rgba(239, 68, 68, 0.25)',
          transition: 'all 0.2s'
        }}
      >
        {loading ? '🛰️ Triangulando WLS...' : `📡 Iniciar Cacería IA (${targetFreq || '--'} MHz)`}
      </button>

      {errorMsg && (
        <div style={{ padding: '8px 12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '6px', color: '#fca5a5', fontSize: '11px' }}>
          ⚠️ {errorMsg}
        </div>
      )}

      {results && (
        <div style={{ padding: '12px', background: 'rgba(34, 197, 94, 0.08)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: '8px', marginTop: '4px' }}>
          <h4 style={{ margin: '0 0 6px 0', color: '#4ade80', fontWeight: 'bold', fontSize: '13px' }}>📍 Sospechoso Localizado (WLS)</h4>
          <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '4px', color: '#cbd5e1' }}>
            <div>Coordenadas: <span style={{ color: 'white', fontWeight: 'bold' }}>{results.detected_tx.lat.toFixed(6)}, {results.detected_tx.lon.toFixed(6)}</span></div>
            <div>Potencia Estimada: <span style={{ color: 'white', fontWeight: 'bold' }}>{results.detected_tx.power_dbm.toFixed(1)} dBm</span></div>
            <div>Altura Antena: <span style={{ color: 'white', fontWeight: 'bold' }}>{results.detected_tx.altitude_m.toFixed(0)} m</span></div>
            <div>Bias Calibrado: <span style={{ color: 'white', fontWeight: 'bold' }}>{results.detected_tx.calibrated_bias_db.toFixed(1)} dB</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', paddingTop: '4px', borderTop: '1px solid rgba(255,255,255,0.05)', fontSize: '11px' }}>
              <span>Error RMSE WLS:</span>
              <strong style={{ color: '#4ade80' }}>{results.error_rmse.toFixed(2)} dB</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IllegalDetectionPanel;
