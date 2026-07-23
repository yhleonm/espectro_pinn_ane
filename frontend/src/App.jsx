import React, { useState, useEffect } from 'react';
import { 
  calculateFspl, 
  calculatePinn, 
  getSocialBarrios, 
  getSocialAnalysis,
  optimizeStations, 
  downloadReport, 
  calculateCompareProfile,
  getCochannelInterference,
  getSignalAtPoint
} from './api';
import MapControl from './components/MapControl';
import ProfileChart from './components/ProfileChart';
import StationSelector from './components/StationSelector';
import InterferencePanel from './components/InterferencePanel';
import CitizenLookup from './components/CitizenLookup';
import IllegalDetectionPanel from './components/IllegalDetectionPanel';

function App() {
  const [activeTab, setActiveTab] = useState('sim'); // 'sim', 'social', 'interference', 'citizen', 'opt'
  
  // Params
  const [tx, setTx] = useState({ lat: 4.6075, lon: -74.0543 });
  const [rx, setRx] = useState({ lat: 4.5781, lon: -74.2144 }); // Default Soacha
  const [freq, setFreq] = useState(101.9);
  const [txPower, setTxPower] = useState(80.0);
  const [radius, setRadius] = useState(30.0);
  const [epochsIa, setEpochsIa] = useState(800);
  
  // Results
  const [fsplResult, setFsplResult] = useState(null);
  const [coverageData, setCoverageData] = useState(null);
  const [socialData, setSocialData] = useState(null);
  const [optResults, setOptResults] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [pinnMetrics, setPinnMetrics] = useState(null);
  
  // Nacional Extension State
  const [mapCenter, setMapCenter] = useState({ lat: 4.6075, lon: -74.0543 });
  const [interferenceData, setInterferenceData] = useState(null);
  const [citizenPoint, setCitizenPoint] = useState(null);
  const [citizenSignals, setCitizenSignals] = useState(null);
  const [selectedStationInfo, setSelectedStationInfo] = useState(null);
  
  // Estados para Detección Ilegal
  const [measurements, setMeasurements] = useState([]);
  const [detectedTx, setDetectedTx] = useState(null);
  const [mapClickPoint, setMapClickPoint] = useState(null);
  
  // Loaders
  const [loading, setLoading] = useState(false);
  const [socialAnalysis, setSocialAnalysis] = useState(null);

  useEffect(() => {
    handleFspl();
  }, [tx, freq, txPower]);

  const handleFspl = async () => {
    try {
      const res = await calculateFspl({ tx, rx: tx, frequency_mhz: freq, tx_power_dbm: txPower });
      setFsplResult(res);
    } catch (e) {}
  };

  const handleRunPinn = async () => {
    setLoading(true);
    setPinnMetrics(null);
    try {
      const res = await calculatePinn({
        tx, frequency_mhz: freq, tx_power_dbm: txPower,
        radius_km: radius, res_px: 100, epochs_ia: epochsIa,
        city: selectedStationInfo?.municipio
      });
      setCoverageData(res);
      // Extraemos métricas regulatorias si vienen en el nuevo formato del backend
      setPinnMetrics({ 
        train_time_sec: res.train_time_sec, 
        cached: res.cached,
        regulatory: res.regulatory_metrics 
      });
    } catch (e) {
      alert("Error en Simulación PINN");
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);
    try {
      const res = await optimizeStations(3);
      setOptResults(res);
      setActiveTab('opt');
    } catch (e) {
      alert("Error en Optimizador");
    } finally {
      setLoading(false);
    }
  };

  const handleFetchSocial = async () => {
    setLoading(true);
    try {
      const [barrios, analysis] = await Promise.all([
        getSocialBarrios(),
        getSocialAnalysis({
          tx, frequency_mhz: freq, tx_power_dbm: txPower,
          radius_km: radius, res_px: 100, epochs_ia: epochsIa
        })
      ]);
      setSocialData(barrios);
      setSocialAnalysis(analysis);
    } catch (e) {
      console.error('Social analysis error:', e);
      // Fallback: at least load barrios
      try { const b = await getSocialBarrios(); setSocialData(b); } catch(e2) {}
    }
    finally { setLoading(false); }
  };

  const handleRunProfile = async () => {
    setLoading(true);
    try {
      const res = await calculateCompareProfile({
        tx, rx, 
        tx_height: 30.0, rx_height: 2.0, 
        tx_power_dbm: txPower, frequency_mhz: freq,
        epochs_ia: epochsIa, n_points: 200,
        city: selectedStationInfo?.municipio
      });
      setProfileData(res);
    } catch (e) {
      alert("Error generando perfil");
    } finally {
      setLoading(false);
    }
  };

  // Callback: estación seleccionada desde StationSelector
  const handleStationSelect = (station) => {
    setTx({ lat: station.lat, lon: station.lon });
    // Por defecto, poner RX a ~5km al sur-oeste
    setRx({ lat: station.lat - 0.04, lon: station.lon - 0.04 });
    setFreq(station.frequency_mhz);
    setTxPower(10 * Math.log10(station.pra_kw) + 60);
    setSelectedStationInfo(station);
    setMapCenter({ lat: station.lat, lon: station.lon });
    // Reset resultados previos al cambiar estación
    setCoverageData(null);
    setProfileData(null);
    setPinnMetrics(null);
    setInterferenceData(null);
    setCitizenSignals(null);
  };

  // Callback: ciudad cambiada desde StationSelector
  const handleCityChange = (cityInfo) => {
    setMapCenter({ lat: cityInfo.lat, lon: cityInfo.lon });
    // Al cambiar ciudad, resetear RX al nuevo centro
    setRx({ lat: cityInfo.lat - 0.02, lon: cityInfo.lon - 0.02 });
  };

  return (
    <div className="app-container">
      {/* CONTROL PANEL */}
      <aside className="sidebar">
        <h2 className="text-gradient">ESPECTRO IA 2.0</h2>
        
        {/* Selector Nacional reemplaza el preset dropdown */}
        <StationSelector 
          onStationSelect={handleStationSelect}
          onCityChange={handleCityChange}
        />

        <div className="card">
          <h3>📡 Configuración TX</h3>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'8px'}}>
            <div className="input-group">
              <label>Latitud</label>
              <input type="number" value={tx.lat} onChange={e => setTx({...tx, lat: parseFloat(e.target.value)})} />
            </div>
            <div className="input-group">
              <label>Longitud</label>
              <input type="number" value={tx.lon} onChange={e => setTx({...tx, lon: parseFloat(e.target.value)})} />
            </div>
          </div>
        </div>

        <div className="card">
          <h3>⚡ Parámetros RF</h3>
          <div className="input-group">
            <label>Frecuencia (MHz)</label>
            <input type="number" value={freq} onChange={e => setFreq(parseFloat(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Potencia (dBm)</label>
            <input type="number" value={txPower} onChange={e => setTxPower(parseFloat(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Radio Análisis (km)</label>
            <input type="number" value={radius} onChange={e => setRadius(parseFloat(e.target.value))} />
          </div>
          <div className="input-group">
            <label>Épocas PINN</label>
            <input type="number" value={epochsIa} onChange={e => setEpochsIa(parseInt(e.target.value))} />
          </div>
        </div>

        <button className="btn-primary" onClick={handleRunPinn} disabled={loading} style={{background: 'linear-gradient(45deg, #10b981, #3b82f6)'}}>
          {loading ? "ENTRENANDO IA..." : "🚀 EJECUTAR SIMULACIÓN IA"}
        </button>

        <button className="btn-primary" onClick={handleOptimize} disabled={loading} style={{background: 'linear-gradient(45deg, #f59e0b, #ef4444)', marginTop:'10px'}}>
          🔍 OPTIMIZAR UBICACIONES
        </button>

        {pinnMetrics && (
          <div className="glass-panel" style={{marginTop:'10px', fontSize:'0.75rem', color:'#10b981'}}>
            Tiempo: {pinnMetrics.train_time_sec?.toFixed(1)}s | 
            {pinnMetrics.cached ? " ✅ Caché" : " 🧠 New Train"}
          </div>
        )}

        <div style={{marginTop:'auto', padding:'10px', fontSize:'0.7rem', color:'#64748b', textAlign:'center'}}>
          Basado en PINN Residual (v4) <br/>
          Colombia 2026 — {selectedStationInfo?.departamento || 'Nacional'}
        </div>
      </aside>

      {/* DASHBOARD */}
      <main className="main-content">
        <header className="header-row">
          <div className="tab-buttons">
            <button className={`tab-btn ${activeTab === 'sim' ? 'active' : ''}`} onClick={() => setActiveTab('sim')}>📡 Propagación</button>
            <button className={`tab-btn ${activeTab === 'social' ? 'active' : ''}`} onClick={() => { setActiveTab('social'); handleFetchSocial(); }}>🏘️ Social</button>
            <button className={`tab-btn ${activeTab === 'interference' ? 'active' : ''}`} onClick={() => setActiveTab('interference')}>📡 Interferencia</button>
            <button className={`tab-btn ${activeTab === 'illegal' ? 'active' : ''}`} onClick={() => setActiveTab('illegal')}>🔦 Detección</button>
            <button className={`tab-btn ${activeTab === 'citizen' ? 'active' : ''}`} onClick={() => setActiveTab('citizen')}>📻 Ciudadano</button>
            <button className={`tab-btn ${activeTab === 'opt' ? 'active' : ''}`} onClick={() => setActiveTab('opt')}>🎯 Optimización</button>
          </div>
          <button onClick={downloadReport} style={{background:'transparent', border:'1px solid #1e293b', color:'#94a3b8', padding:'5px 15px', borderRadius:'6px', cursor:'pointer'}}>
            📄 PDF Report
          </button>
        </header>

        {/* TOP METRICS */}
        <section className="metric-grid">
          <div className="metric-box">
            <div className="metric-label">Área Cobertura</div>
            <div className="metric-value">{coverageData?.quant_metrics?.area_sqkm?.toFixed(1) || "---"} km²</div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Población Est.</div>
            <div className="metric-value" style={{color:'#10b981'}}>
              {socialAnalysis ? `~${(socialAnalysis.summary.total_hab_cubiertos / 1000000).toFixed(1)}M` 
                : coverageData ? `~${(coverageData.quant_metrics.area_sqkm * 1200 / 1000000).toFixed(1)}M` : "---"}
            </div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Déficit Crítico</div>
            <div className="metric-value" style={{color:'#ef4444'}}>
              {activeTab === 'opt' ? 'Reduciendo...' : (coverageData ? `${Math.max(0, 100 - (coverageData.quant_metrics.area_sqkm / (radius * radius * Math.PI) * 100)).toFixed(0)}%` : "---")}
            </div>
          </div>
          <div className="metric-box">
            <div className="metric-label">Confianza IA</div>
            <div className="metric-value">
              {coverageData?.quant_metrics?.loss_final ? `${Math.max(0, 100 - coverageData.quant_metrics.loss_final * 100).toFixed(1)}%` : "99.9%"}
            </div>
          </div>
        </section>

        {/* MAIN DISPLAY AREA (Map Full with Overlays) */}
        <div style={{ position: 'relative', flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', overflow: 'hidden' }}>
          
          <section className="map-viewport" style={{ flex: 1, position: 'relative' }}>
             <MapControl 
               tx={tx} 
               rx={rx} 
               onRxChange={setRx}
               mode={activeTab === 'social' ? 'social' : '2d'} 
               coverageData={activeTab === 'sim' ? coverageData : null}
               socialData={socialData}
               optResults={optResults}
               mapCenter={mapCenter}
               interferenceStations={activeTab === 'interference' ? interferenceData?.conflicts : null}
               citizenPoint={activeTab === 'citizen' ? citizenPoint : null}
               citizenSignals={activeTab === 'citizen' ? citizenSignals : null}
               onMapClick={(point) => setMapClickPoint(point)}
               measurements={measurements}
               detectedTx={activeTab === 'illegal' ? detectedTx : null}
               stationInfo={selectedStationInfo}
             />
             
             {/* Floating Heatmap Legend */}
             <div className="glass-panel" style={{position:'absolute', bottom:'20px', left:'20px', zIndex:1000, width:'200px'}}>
                <h4 style={{margin:'0 0 10px 0', fontSize:'0.7rem'}}>Nivel de Señal (IA)</h4>
                <div style={{height:'8px', background:'linear-gradient(to right, blue, cyan, green, yellow, red)', borderRadius:'4px'}}></div>
                <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.55rem', marginTop:'4px'}}>
                  <span>Bajo</span>
                  <span>Alto</span>
                </div>
             </div>

             {/* ANE Regulatory Metrics Overlay (SIEMPRE VISIBLE SI HAY DATOS) */}
             {pinnMetrics?.regulatory && (
               <div className="glass-panel" style={{position:'absolute', top:'20px', right:'20px', zIndex:1000, width:'180px', background:'rgba(15, 23, 42, 0.9)'}}>
                  <h4 style={{margin:'0 0 8px 0', fontSize:'0.75rem', color:'#10b981'}}>Bandas ANE (km²)</h4>
                  <div style={{fontSize:'0.65rem', display:'flex', flexDirection:'column', gap:'4px'}}>
                    <div style={{display:'flex', justifyContent:'space-between'}}>
                      <span>66-73:</span> <strong>{pinnMetrics.regulatory.areas_km2["66-73"]?.toFixed(0)}</strong>
                    </div>
                    <div style={{display:'flex', justifyContent:'space-between'}}>
                      <span>74-85:</span> <strong>{pinnMetrics.regulatory.areas_km2["74-85"]?.toFixed(0)}</strong>
                    </div>
                    <div style={{display:'flex', justifyContent:'space-between'}}>
                      <span>86-95:</span> <strong>{pinnMetrics.regulatory.areas_km2["86-95"]?.toFixed(0)}</strong>
                    </div>
                    <div style={{display:'flex', justifyContent:'space-between'}}>
                      <span>≥ 96:</span> <strong>{pinnMetrics.regulatory.areas_km2["96+"]?.toFixed(0)}</strong>
                    </div>
                  </div>
               </div>
             )}

             {/* Citizen mode instruction */}
             {activeTab === 'citizen' && !citizenPoint && (
               <div className="glass-panel" style={{position:'absolute', top:'20px', left:'50%', transform:'translateX(-50%)', zIndex:1000, textAlign:'center', fontSize:'0.75rem', color:'#38bdf8'}}>
                 🖱️ Haga clic en el mapa para seleccionar un punto
               </div>
             )}
          </section>

          {/* SOCIAL STATS TABLE (visible in social tab) */}
          {activeTab === 'social' && socialAnalysis && (
            <section className="glass-panel" style={{ maxHeight: '300px', overflowY: 'auto', padding: '1rem' }}>
              <h3 style={{margin:'0 0 10px 0', fontSize:'0.85rem'}}>Impacto Social por Localidad (Pob. 2026)</h3>
              <div style={{fontSize:'0.65rem', marginBottom:'8px', color:'#10b981'}}>
                Total: {socialAnalysis.summary.total_hab_cubiertos?.toLocaleString()} hab cubiertos / {socialAnalysis.summary.total_poblacion?.toLocaleString()} ({socialAnalysis.summary.pct_poblacion_cubierta}%)
                {' | '}Área: {socialAnalysis.summary.area_cobertura_km2} km²
              </div>
              <table style={{width:'100%', borderCollapse:'collapse', fontSize:'0.65rem'}}>
                <thead>
                  <tr style={{borderBottom:'1px solid #334155', color:'#94a3b8'}}>
                    <th style={{textAlign:'left', padding:'4px'}}>Localidad</th>
                    <th style={{textAlign:'right', padding:'4px'}}>Cob %</th>
                    <th style={{textAlign:'right', padding:'4px'}}>Señal Max</th>
                    <th style={{textAlign:'right', padding:'4px'}}>Población</th>
                    <th style={{textAlign:'right', padding:'4px'}}>Hab. Cubiertos</th>
                  </tr>
                </thead>
                <tbody>
                  {socialAnalysis.stats_by_localidad.map((row, i) => (
                    <tr key={i} style={{borderBottom:'1px solid #1e293b', color: row.cobertura_pct > 50 ? '#10b981' : row.cobertura_pct > 10 ? '#f59e0b' : '#ef4444'}}>
                      <td style={{padding:'4px'}}>{row.localidad}</td>
                      <td style={{textAlign:'right', padding:'4px'}}>{row.cobertura_pct}%</td>
                      <td style={{textAlign:'right', padding:'4px'}}>{row.max_signal} dB</td>
                      <td style={{textAlign:'right', padding:'4px'}}>{row.poblacion?.toLocaleString()}</td>
                      <td style={{textAlign:'right', padding:'4px', fontWeight:'bold'}}>{row.hab_cubiertos?.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* INTERFERENCE PANEL (visible in interference tab) */}
          {activeTab === 'interference' && (
            <section className="glass-panel" style={{ maxHeight: '300px', overflowY: 'auto', padding: '1rem' }}>
              <InterferencePanel 
                station={selectedStationInfo}
                onAnalyze={(results) => setInterferenceData(results)}
              />
            </section>
          )}

          {/* CITIZEN LOOKUP PANEL (visible in citizen tab) */}
          {activeTab === 'citizen' && (
            <section className="glass-panel" style={{ maxHeight: '400px', overflowY: 'auto', padding: '1rem' }}>
                <CitizenLookup 
                  onResults={(signals) => setCitizenSignals(signals)}
                  mapClickPoint={mapClickPoint}
                />
            </section>
          )}

          {activeTab === 'illegal' && (
            <div style={{ height: '400px', overflowY: 'auto', paddingRight: '10px' }}>
                <IllegalDetectionPanel 
                  tx={tx}
                  onDetectionResults={(res, currentMeasurements) => {
                    setDetectedTx(res.detected_tx);
                    setMeasurements(currentMeasurements); 
                    setMapCenter({ lat: res.detected_tx.lat, lon: res.detected_tx.lon });
                  }}
                  onMeasurementsChange={(m) => setMeasurements(m)}
                  mapClickPoint={mapClickPoint}
                />
            </div>
          )}

          {/* PROFILE CHART (Floating or Bottom) */}
          <section className="glass-panel" style={{ height: activeTab === 'sim' ? '250px' : '0px', transition: 'height 0.3s', overflow: 'hidden', display: 'flex', flexDirection: 'column', padding: activeTab === 'sim' ? '1rem' : '0px' }}>
             <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'5px'}}>
               <h3 style={{margin:0, fontSize:'0.85rem'}}>⛰️ Perfil SRTM vs PINN</h3>
               <button onClick={handleRunProfile} disabled={loading} style={{fontSize:'0.65rem', padding:'2px 8px', background:'#3b82f6', border:'none', borderRadius:'4px', color:'white', cursor:'pointer'}}>
                 ACTUALIZAR PERFIL
               </button>
             </div>
             <div style={{flex:1, minHeight: 0}}>
               {profileData ? <ProfileChart data={profileData} /> : <div style={{height:'100%', display:'flex', alignItems:'center', justifyContent:'center', color:'#64748b', fontSize:'0.75rem'}}>Presiona actualizar para ver la física del terreno</div>}
             </div>
          </section>

        </div>
      </main>
    </div>
  );
}

export default App;
