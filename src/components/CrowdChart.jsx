import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function CrowdChart({
  data,
  currentCount = 0,
  uniqueTracks = 0,
  peakCrowd = 0,
  congestionRisk = 'HIGH',
  isLive = false,
  trendLabel = '↑ 8.4% / hr',
  flowDirection = 'STATIONARY',
  flowSpeed = 0,
  congestionStatus = 'LOW',
  anomalyStatus = 'NORMAL',
  anomalyScore = 0,
  anomalyReason = 'Normal movement patterns',
}) {
  const currentRisk = isLive ? (congestionStatus || 'LOW') : congestionRisk
  const riskClass =
    currentRisk === 'CRITICAL'
      ? 'red-text'
      : currentRisk === 'HIGH'
      ? 'orange-text'
      : currentRisk === 'MEDIUM'
      ? 'yellow-text'
      : 'green-text'

  const currentAnomaly = isLive ? (anomalyStatus || 'NORMAL') : 'NORMAL'
  const anomalyClass =
    currentAnomaly === 'CRITICAL'
      ? 'red-text'
      : currentAnomaly === 'ELEVATED'
      ? 'orange-text'
      : currentAnomaly === 'WATCH'
      ? 'yellow-text'
      : 'green-text'

  return (
    <section className="card chart-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">CROWD INTELLIGENCE</span>
          <h3>Movement, congestion & anomaly overview</h3>
        </div>
        <span className="trend">
          {isLive ? (
            <span className="live-tag">
              <i /> LIVE CAMERA-07
            </span>
          ) : (
            trendLabel
          )}
        </span>
      </div>
      <div className="intel-cards">
        {isLive ? (
          <>
            <div>
              <small>CURRENT CROWD FLOW</small>
              <b>{flowDirection || 'STATIONARY'}</b>
            </div>
            <div>
              <small>AVERAGE SPEED</small>
              <b>{typeof flowSpeed === 'number' ? `${flowSpeed.toFixed(1)} px/s` : '0.0 px/s'}</b>
            </div>
            <div>
              <small>CONGESTION RISK</small>
              <b className={riskClass}>{congestionStatus || 'LOW'}</b>
            </div>
            <div>
              <small>BEHAVIOUR STATUS</small>
              <b className={anomalyClass}>
                {currentAnomaly}{' '}
                <span style={{ fontSize: '11px', fontWeight: 500, opacity: 0.85 }}>
                  ({typeof anomalyScore === 'number' ? `${(anomalyScore * 100).toFixed(0)}%` : '0%'})
                </span>
              </b>
              <small
                style={{
                  marginTop: '3px',
                  fontSize: '8.5px',
                  color: '#8aa0b5',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: 'block',
                }}
                title={anomalyReason}
              >
                {anomalyReason || 'Normal movement patterns'}
              </small>
            </div>
          </>
        ) : (
          <>
            <div>
              <small>CURRENT CROWD FLOW</small>
              <b>→ North</b>
            </div>
            <div>
              <small>AVERAGE SPEED</small>
              <b>1.8 m/s</b>
            </div>
            <div>
              <small>CONGESTION RISK</small>
              <b className={riskClass}>{congestionRisk}</b>
            </div>
            <div>
              <small>BEHAVIOUR STATUS</small>
              <b className="green-text">
                NORMAL{' '}
                <span style={{ fontSize: '11px', fontWeight: 500, opacity: 0.85 }}>
                  (0%)
                </span>
              </b>
              <small
                style={{
                  marginTop: '3px',
                  fontSize: '8.5px',
                  color: '#8aa0b5',
                  display: 'block',
                }}
              >
                Normal movement patterns
              </small>
            </div>
          </>
        )}
      </div>
      <div className="chart-title">
        <span>{isLive ? 'Live detected crowd' : 'Crowd count'}</span>
        <small>
          {isLive
            ? `Current crowd: ${currentCount} · Unique tracks: ${uniqueTracks} · Peak crowd: ${peakCrowd}`
            : 'Last 60 minutes'}
        </small>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="crowdFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#2d8cff" stopOpacity=".35" />
                <stop offset="100%" stopColor="#2d8cff" stopOpacity="0" />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tickLine={false} axisLine={false} />
            <YAxis hide />
            <Tooltip
              contentStyle={{
                background: '#111b29',
                border: '1px solid #2a3a4e',
                borderRadius: 8,
              }}
            />
            <Area
              type="monotone"
              dataKey="crowd"
              stroke="#2d8cff"
              strokeWidth={2.5}
              fill="url(#crowdFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
