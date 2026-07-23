import React, { useState, useEffect } from 'react';
import { getDepartments, getCities, getStations } from '../api';

const StationSelector = ({ onStationSelect, onCityChange }) => {
  const [departments, setDepartments] = useState([]);
  const [cities, setCities] = useState([]);
  const [stations, setStations] = useState([]);
  const [selectedDept, setSelectedDept] = useState('');
  const [selectedCity, setSelectedCity] = useState('');
  const [selectedStation, setSelectedStation] = useState(null);
  const [loading, setLoading] = useState({ depts: false, cities: false, stations: false });

  // Cargar departamentos al montar
  useEffect(() => {
    const fetchDepts = async () => {
      setLoading(prev => ({ ...prev, depts: true }));
      try {
        const data = await getDepartments();
        setDepartments(data || []);
      } catch (e) {
        console.error('Error cargando departamentos:', e);
      } finally {
        setLoading(prev => ({ ...prev, depts: false }));
      }
    };
    fetchDepts();
  }, []);

  // Cuando cambia el departamento, cargar ciudades
  useEffect(() => {
    if (!selectedDept) {
      setCities([]);
      setStations([]);
      setSelectedCity('');
      setSelectedStation(null);
      return;
    }
    const fetchCities = async () => {
      setLoading(prev => ({ ...prev, cities: true }));
      try {
        const data = await getCities(selectedDept);
        setCities(data || []);
        setSelectedCity('');
        setStations([]);
        setSelectedStation(null);
      } catch (e) {
        console.error('Error cargando ciudades:', e);
      } finally {
        setLoading(prev => ({ ...prev, cities: false }));
      }
    };
    fetchCities();
  }, [selectedDept]);

  // Cuando cambia la ciudad, cargar estaciones
  useEffect(() => {
    if (!selectedCity) {
      setStations([]);
      setSelectedStation(null);
      return;
    }
    const fetchStations = async () => {
      setLoading(prev => ({ ...prev, stations: true }));
      try {
        const data = await getStations({ dept: selectedDept, city: selectedCity });
        setStations(data || []);
        setSelectedStation(null);
        // Notificar cambio de ciudad para centrar mapa
        if (data && data.length > 0) {
          const avgLat = data.reduce((s, st) => s + st.lat, 0) / data.length;
          const avgLon = data.reduce((s, st) => s + st.lon, 0) / data.length;
          if (onCityChange) onCityChange({ lat: avgLat, lon: avgLon, name: selectedCity });
        }
      } catch (e) {
        console.error('Error cargando estaciones:', e);
      } finally {
        setLoading(prev => ({ ...prev, stations: false }));
      }
    };
    fetchStations();
  }, [selectedCity]);

  const handleStationChange = (e) => {
    const stationId = e.target.value;
    if (!stationId) {
      setSelectedStation(null);
      return;
    }
    const station = stations.find(s => String(s.id) === stationId);
    if (station) {
      setSelectedStation(station);
      if (onStationSelect) onStationSelect(station);
    }
  };

  return (
    <div className="card">
      <h3>📡 Selector Nacional</h3>

      {/* Departamento */}
      <div className="input-group" style={{ marginBottom: '8px' }}>
        <label>
          Departamento
          {departments.length > 0 && (
            <span style={{ float: 'right', color: '#3b82f6', fontSize: '0.65rem' }}>
              {departments.length} depts
            </span>
          )}
        </label>
        <select
          value={selectedDept}
          onChange={(e) => setSelectedDept(e.target.value)}
          disabled={loading.depts}
        >
          <option value="">
            {loading.depts ? '⏳ Cargando...' : '— Seleccionar departamento —'}
          </option>
          {departments.map(d => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      {/* Ciudad */}
      <div className="input-group" style={{ marginBottom: '8px' }}>
        <label>
          Ciudad / Municipio
          {cities.length > 0 && (
            <span style={{ float: 'right', color: '#3b82f6', fontSize: '0.65rem' }}>
              {cities.length} ciudades
            </span>
          )}
        </label>
        <select
          value={selectedCity}
          onChange={(e) => setSelectedCity(e.target.value)}
          disabled={!selectedDept || loading.cities}
        >
          <option value="">
            {loading.cities ? '⏳ Cargando...' : !selectedDept ? '— Seleccione departamento primero —' : '— Seleccionar ciudad —'}
          </option>
          {cities.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* Estación */}
      <div className="input-group" style={{ marginBottom: '8px' }}>
        <label>
          Estación
          {stations.length > 0 && (
            <span style={{ float: 'right', color: '#10b981', fontSize: '0.65rem' }}>
              {stations.length} estaciones
            </span>
          )}
        </label>
        <select
          value={selectedStation ? String(selectedStation.id) : ''}
          onChange={handleStationChange}
          disabled={!selectedCity || loading.stations}
        >
          <option value="">
            {loading.stations ? '⏳ Cargando...' : !selectedCity ? '— Seleccione ciudad primero —' : '— Seleccionar estación —'}
          </option>
          {stations.map(s => (
            <option key={s.id} value={String(s.id)}>
              ID {s.id} — {s.frequency_mhz} MHz ({s.pra_kw} kW)
            </option>
          ))}
        </select>
      </div>

      {/* Info de estación seleccionada */}
      {selectedStation && (
        <div className="glass-panel" style={{ marginTop: '8px', padding: '10px', fontSize: '0.7rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#94a3b8' }}>ID</span>
            <strong style={{ color: '#10b981' }}>{selectedStation.id}</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#94a3b8' }}>Frecuencia</span>
            <strong>{selectedStation.frequency_mhz} MHz</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#94a3b8' }}>PRA</span>
            <strong>{selectedStation.pra_kw} kW ({selectedStation.pra_w} W)</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#94a3b8' }}>Clase</span>
            <strong>{selectedStation.clase || '—'}</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#94a3b8' }}>Ubicación</span>
            <strong>{selectedStation.lat?.toFixed(4)}, {selectedStation.lon?.toFixed(4)}</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#94a3b8' }}>Altura</span>
            <strong>{selectedStation.altura_m || '—'} m</strong>
          </div>
        </div>
      )}
    </div>
  );
};

export default StationSelector;
