import React from 'react'
import { normalizeZones } from '../utils/zoneUtils'

const getBadgeClass = (status) => {
  const s = String(status || '').toUpperCase()
  if (s === 'CRITICAL') return 'critical'
  if (s === 'HIGH' || s === 'ELEVATED') return 'high'
  if (s === 'MEDIUM' || s === 'MODERATE' || s === 'WATCH') return 'moderate'
  return 'normal'
}

// Default 4-zone layout for Camera-07 standby state
const DEFAULT_STANDBY_ZONES = [
  { zone_id: 'zone_a', name: 'Zone A (Top-Left)', capacity: 10 },
  { zone_id: 'zone_b', name: 'Zone B (Top-Right)', capacity: 10 },
  { zone_id: 'zone_c', name: 'Zone C (Bottom-Left)', capacity: 10 },
  { zone_id: 'zone_d', name: 'Zone D (Bottom-Right)', capacity: 10 },
]

export default function ZoneTable({ zones, isLive }) {
  const zoneList = normalizeZones(zones)
  const hasLiveZones = zoneList.length > 0

  const displayList = hasLiveZones ? zoneList : DEFAULT_STANDBY_ZONES

  return (
    <section className="card zone-table">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">SPATIAL ZONE INTELLIGENCE</span>
          <h3>Camera-07 Viewport Quadrants (Ramghat)</h3>
        </div>
        <div>
          {hasLiveZones ? (
            <span className="live-tag">
              <i /> LIVE ZONE TELEMETRY
            </span>
          ) : (
            <span className="upload-hint">STANDBY · 4 QUADRANTS CONFIGURED</span>
          )}
        </div>
      </div>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Zone</th>
              <th>Headcount</th>
              <th>Density (EMA)</th>
              <th>Crowd Status</th>
              <th>Flow Velocity</th>
              <th>Congestion</th>
              <th>Behaviour Anomaly</th>
              <th>Risk Level</th>
            </tr>
          </thead>
          <tbody>
            {displayList.map((z) => {
              if (!hasLiveZones) {
                return (
                  <tr key={z.zone_id}>
                    <td>
                      <b>{z.name}</b> <small>({z.zone_id})</small>
                    </td>
                    <td>0 <small>/ {z.capacity}</small></td>
                    <td>0.0%</td>
                    <td>
                      <span className="badge normal">STANDBY</span>
                    </td>
                    <td>STATIONARY · 0.0 px/s</td>
                    <td>
                      <span className="badge normal">LOW (0.0%)</span>
                    </td>
                    <td>
                      <span className="badge normal">NORMAL (0.0%)</span>
                    </td>
                    <td>
                      <span className="badge normal">NORMAL</span>
                    </td>
                  </tr>
                )
              }

              const densityPercent = typeof z.smoothed_density === 'number'
                ? (z.smoothed_density * 100).toFixed(1)
                : typeof z.density_index === 'number'
                ? (z.density_index * 100).toFixed(1)
                : '0.0'

              const rawDensityPercent = typeof z.density_index === 'number'
                ? (z.density_index * 100).toFixed(1)
                : '0.0'

              const congestionPercent = typeof z.congestion_index === 'number'
                ? (z.congestion_index * 100).toFixed(1)
                : '0.0'

              const anomalyPercent = typeof z.anomaly_score === 'number'
                ? (z.anomaly_score * 100).toFixed(1)
                : '0.0'

              const flowSpeedStr = typeof z.flow_speed === 'number'
                ? `${z.flow_speed.toFixed(1)} ${z.flow_speed_unit || 'px/s'}`
                : `0.0 ${z.flow_speed_unit || 'px/s'}`

              return (
                <tr key={z.zone_id}>
                  <td>
                    <b>{z.name}</b> <small>({z.zone_id})</small>
                  </td>
                  <td>
                    <b>{z.current_count ?? 0}</b>{' '}
                    <small>
                      / {z.capacity ?? 10} ({z.active_track_count ?? 0} tracked)
                    </small>
                  </td>
                  <td>
                    <b>{densityPercent}%</b>{' '}
                    <small title="Raw image-space density index">
                      (raw: {rawDensityPercent}%)
                    </small>
                  </td>
                  <td>
                    <span className={`badge ${getBadgeClass(z.crowd_status)}`}>
                      {z.crowd_status || 'LOW'}
                    </span>
                  </td>
                  <td>
                    <span>
                      {z.flow_direction || 'STATIONARY'} · {flowSpeedStr}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${getBadgeClass(z.congestion_status)}`}>
                      {z.congestion_status || 'LOW'} ({congestionPercent}%)
                    </span>
                  </td>
                  <td>
                    <span
                      className={`badge ${getBadgeClass(z.anomaly_status)}`}
                      title={z.anomaly_reason || 'Normal movement patterns'}
                    >
                      {z.anomaly_status || 'NORMAL'} ({anomalyPercent}%)
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${getBadgeClass(z.risk)}`}>
                      {z.risk || 'NORMAL'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
