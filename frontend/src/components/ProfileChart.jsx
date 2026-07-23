import React from 'react';
import Plot from 'react-plotly.js';

const ProfileChart = ({ data }) => {
  if (!data) return null;

  const { distances_km, elevations_m, fspl_dbm, empirical_dbm, pinn_dbm } = data;

  return (
    <Plot
      data={[
        {
          x: distances_km,
          y: elevations_m,
          fill: 'tozeroy',
          type: 'scatter',
          mode: 'lines',
          line: { color: 'saddlebrown' },
          name: 'Terreno (SRTM)',
          yaxis: 'y'
        },
        {
          x: distances_km,
          y: fspl_dbm,
          mode: 'lines',
          type: 'scatter',
          line: { color: '#10b981', width: 2, dash: 'dash' },
          name: 'FSPL (Teórico)',
          yaxis: 'y2'
        },
        {
          x: distances_km,
          y: empirical_dbm,
          mode: 'lines',
          type: 'scatter',
          line: { color: '#3b82f6', width: 2 },
          name: 'Difracción Knife-Edge',
          yaxis: 'y2'
        },
        {
          x: distances_km,
          y: pinn_dbm,
          mode: 'lines',
          type: 'scatter',
          line: { color: '#ef4444', width: 2 },
          name: 'Inferencia PINN (IA)',
          yaxis: 'y2'
        }
      ]}
      layout={{
        autosize: true,
        title: 'Validación Científica: Perfil Topográfico vs Propagación RF',
        xaxis: { title: 'Distancia (km)', gridcolor: '#334155' },
        yaxis: { 
          title: 'Elevación (m.s.n.m)', 
          gridcolor: '#334155',
          side: 'left'
        },
        yaxis2: {
          title: 'Potencia Rx (dBm)',
          overlaying: 'y',
          side: 'right',
          showgrid: false
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#f8fafc' },
        margin: { t: 40, l: 50, r: 50, b: 40 },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.5)' }
      }}
      useResizeHandler={true}
      style={{ width: '100%', height: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
};

export default ProfileChart;
