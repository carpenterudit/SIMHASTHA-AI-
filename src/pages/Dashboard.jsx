import { useState } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import KpiCard from '../components/KpiCard'
import CctvPanel from '../components/CctvPanel'
import UjjainMap from '../components/UjjainMap'
import CrowdChart from '../components/CrowdChart'
import ZoneTable from '../components/ZoneTable'
import IncidentPanel from '../components/IncidentPanel'
import EmergencyWidget from '../components/EmergencyWidget'
import DroneFleet from '../components/DroneFleet'
import AlertPanel from '../components/AlertPanel'
import { dashboardData } from '../data/mockData'

export default function Dashboard() {
  const [menu, setMenu] = useState(false)
  const [modal, setModal] = useState(false)
  const [activeJob, setActiveJob] = useState(null)
  const [liveChartHistory, setLiveChartHistory] = useState([])

  const handleJobUpdate = job => {
    if (!job) {
      setActiveJob(null)
      setLiveChartHistory([])
      return
    }

    setActiveJob(job)

    if (['processing', 'completed'].includes(job.status) && job.frame) {
      setLiveChartHistory(prev => {
        const last = prev[prev.length - 1]
        if (last && last.frameNum === job.frame) return prev
        const next = [
          ...prev,
          {
            time: `F${job.frame}`,
            crowd: job.current_count,
            frameNum: job.frame,
          },
        ]
        return next.slice(-20)
      })
    }
  }

  const isLive = Boolean(
    activeJob && ['processing', 'completed'].includes(activeJob.status)
  )

  // Dynamic KPIs: Connect real Camera-07 analysis to Current Crowd & Crowd Density
  const kpiItems = dashboardData.kpis.map(item => {
    if (item.label === 'Total crowd') {
      if (!isLive) return item
      return {
        ...item,
        label: 'CURRENT CROWD',
        value: activeJob.current_count.toLocaleString(),
        detail: `Unique tracks: ${activeJob.unique_track_count ?? 0} · Peak: ${activeJob.peak_crowd ?? activeJob.current_count} (Camera-07)`,
        tone: 'blue',
      }
    }

    if (item.label === 'Crowd density') {
      if (!isLive) return item
      const status = activeJob.crowd_status || 'LOW'
      const tone =
        status === 'CRITICAL'
          ? 'red'
          : status === 'HIGH'
          ? 'orange'
          : status === 'MEDIUM'
          ? 'orange'
          : 'green'
      const densityPercent =
        typeof activeJob.density_index === 'number'
          ? (activeJob.density_index * 100).toFixed(1)
          : '0.0'

      return {
        ...item,
        label: 'CROWD DENSITY',
        value: status,
        detail: `Image-space index: ${densityPercent}% (provisional)`,
        tone,
      }
    }

    return item
  })

  // Dynamic Zones: Update Ramghat with real Camera-07 AI metrics
  const zoneItems = dashboardData.zones.map(z => {
    if (z.zone === 'Ramghat' && isLive) {
      const densityPercent =
        typeof activeJob.density_index === 'number'
          ? `${Math.round(activeJob.density_index * 100)}%`
          : '0%'
      const status = activeJob.crowd_status || 'LOW'
      const risk =
        status === 'CRITICAL'
          ? 'Critical'
          : status === 'HIGH'
          ? 'High'
          : status === 'MEDIUM'
          ? 'Moderate'
          : 'Normal'

      return {
        ...z,
        people: `${activeJob.current_count} (Peak: ${activeJob.peak_crowd ?? activeJob.current_count})`,
        density: densityPercent,
        flow: activeJob.flow_direction || z.flow,
        risk,
      }
    }

    return z
  })

  // Crowd chart: Real frame readings when live, falling back to 60-min baseline
  const chartData =
    isLive && liveChartHistory.length >= 2
      ? liveChartHistory
      : dashboardData.chart

  const congestionRisk = isLive ? activeJob.congestion_status || activeJob.crowd_status || 'LOW' : 'HIGH'

  return (
    <div className="app-shell">
      <Header onMenu={() => setMenu(!menu)} />
      <Sidebar open={menu} onClose={() => setMenu(false)} />
      <main>
        <div className="page-intro">
          <div>
            <p className="eyebrow">
              {isLive ? 'OPERATIONAL OVERVIEW · LIVE TELEMETRY' : 'OPERATIONAL OVERVIEW'}
            </p>
            <h1>Command Center Dashboard</h1>
            <span>
              {isLive
                ? 'Simhastha Ujjain · Real-time AI analysis from Camera-07 (Ramghat)'
                : 'Simhastha Ujjain · Live mock operational data'}
            </span>
          </div>
          <button className="refresh">
            {isLive ? (
              <span className="live-tag">
                <i /> AI ANALYSIS ACTIVE
              </span>
            ) : (
              '● Auto-refresh enabled'
            )}
          </button>
        </div>

        <div className="kpi-grid">
          {kpiItems.map(item => (
            <KpiCard item={item} key={item.label} />
          ))}
        </div>

        <div className="primary-grid">
          <CctvPanel onJobUpdate={handleJobUpdate} />
          <UjjainMap />
        </div>

        <div className="content-grid">
          <div>
            <CrowdChart
              data={chartData}
              currentCount={activeJob?.current_count ?? 0}
              uniqueTracks={activeJob?.unique_track_count ?? 0}
              peakCrowd={activeJob?.peak_crowd ?? 0}
              congestionRisk={congestionRisk}
              isLive={isLive}
              flowDirection={activeJob?.flow_direction || 'STATIONARY'}
              flowSpeed={activeJob?.flow_speed ?? 0}
              congestionStatus={activeJob?.congestion_status || 'LOW'}
              anomalyStatus={activeJob?.anomaly_status || 'NORMAL'}
              anomalyScore={activeJob?.anomaly_score ?? 0}
              anomalyReason={activeJob?.anomaly_reason || 'Normal movement patterns'}
            />
            <ZoneTable zones={zoneItems} />
          </div>
          <div className="side-stack">
            <IncidentPanel incidents={dashboardData.incidents} />
            <EmergencyWidget
              onOpen={() => setModal(true)}
              open={modal}
              onClose={() => setModal(false)}
            />
            <DroneFleet drones={dashboardData.drones} />
            <AlertPanel alerts={dashboardData.alerts} />
          </div>
        </div>
      </main>
    </div>
  )
}
