import { useEffect, useRef, useState } from 'react'
import { FileVideo, LoaderCircle, Play, Upload, Wifi } from 'lucide-react'
import { getDetectionStatus, mediaUrl, startDetection, uploadVideo } from '../services/api'

const formatBytes = size => `${(size / 1024 / 1024).toFixed(2)} MB`

// Phase: configurable-detection-settings experiment.
// Options intentionally limited to a small, known-safe set.
const MODEL_OPTIONS = ['yolo11n.pt', 'yolo11s.pt']
const CONFIDENCE_OPTIONS = [0.25, 0.30, 0.40]
const INPUT_SIZE_OPTIONS = [640, 960]

// Must match the existing backend defaults (DEFAULT_MODEL / DEFAULT_CONFIDENCE / DEFAULT_INPUT_SIZE).
const DEFAULT_SETTINGS = { model_name: 'yolo11n.pt', confidence: 0.4, input_size: 640 }

export default function CctvPanel() {
  const [upload, setUpload] = useState(null)
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const pollRef = useRef(null)

  useEffect(() => () => clearInterval(pollRef.current), [])

  const selectFile = async event => {
    const file = event.target.files?.[0]
    if (!file) return
    setError('')
    setJob(null)
    setBusy(true)
    try {
      setUpload(await uploadVideo(file))
    } catch (cause) {
      setUpload(null)
      setError(cause.message)
    } finally {
      setBusy(false)
      event.target.value = ''
    }
  }

  const runAnalysis = async () => {
    if (!upload) return
    setBusy(true)
    setError('')
    try {
      const started = await startDetection({
        upload_id: upload.upload_id,
        confidence: settings.confidence,
        input_size: settings.input_size,
        frame_skip: 1,
        model_name: settings.model_name,
      })
      setJob(started)
      clearInterval(pollRef.current)
      pollRef.current = setInterval(async () => {
        try {
          const status = await getDetectionStatus(started.id)
          setJob(status)
          if (['completed', 'failed'].includes(status.status)) clearInterval(pollRef.current)
        } catch (cause) {
          setError(cause.message)
          clearInterval(pollRef.current)
        }
      }, 700)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setBusy(false)
    }
  }

  const imageUrl = mediaUrl(job?.latest_frame_url)
  const stateLabel = job ? job.status.replace('_', ' ').toUpperCase() : upload ? 'READY' : 'WAITING FOR VIDEO'
  // Settings stay editable until a job exists; once analysis starts they're locked (matches
  // existing one-shot upload -> analyze flow, no mid-job reconfiguration).
  const canConfigure = upload && !job

  return (
    <section className="card cctv">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">LIVE CCTV MONITORING</span>
          <h3>Ramghat South Approach</h3>
        </div>
        <span className="camera-live"><i/> {job ? 'AI ANALYSIS' : 'NO VIDEO'}</span>
      </div>
      <div className="upload-strip">
        <label className="upload-button">
          <Upload size={15}/>
          <span>{busy ? 'UPLOADING…' : 'SELECT VIDEO'}</span>
          <input type="file" accept=".mp4,.avi,.mov,.mkv,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska" onChange={selectFile} disabled={busy}/>
        </label>
        {upload ? (
          <div className="upload-meta">
            <FileVideo size={16}/>
            <span><b>{upload.filename}</b> · {formatBytes(upload.size_bytes)} · {upload.resolution} · {upload.fps} FPS · {upload.duration_seconds}s</span>
            <em>{stateLabel}</em>
          </div>
        ) : (
          <span className="upload-hint">MP4, AVI, MOV, or MKV</span>
        )}
        {canConfigure && (
          <div className="analysis-settings">
            <label>
              <span>MODEL</span>
              <select value={settings.model_name} onChange={e => setSettings(current => ({ ...current, model_name: e.target.value }))} disabled={busy}>
                {MODEL_OPTIONS.map(option => <option key={option} value={option}>{option.replace('.pt', '').toUpperCase()}</option>)}
              </select>
            </label>
            <label>
              <span>CONFIDENCE</span>
              <select value={settings.confidence} onChange={e => setSettings(current => ({ ...current, confidence: Number(e.target.value) }))} disabled={busy}>
                {CONFIDENCE_OPTIONS.map(option => <option key={option} value={option}>{option.toFixed(2)}</option>)}
              </select>
            </label>
            <label>
              <span>INPUT SIZE</span>
              <select value={settings.input_size} onChange={e => setSettings(current => ({ ...current, input_size: Number(e.target.value) }))} disabled={busy}>
                {INPUT_SIZE_OPTIONS.map(option => <option key={option} value={option}>{option}px</option>)}
              </select>
            </label>
          </div>
        )}
        {upload && !job && <button className="analysis-button" onClick={runAnalysis} disabled={busy}><Play size={15}/> START AI ANALYSIS</button>}
      </div>
      {error && <p className="analysis-error">{error}</p>}
      <div className={`video-frame ${imageUrl ? 'real-frame' : ''}`}>
        {imageUrl ? (
          <img src={`${imageUrl}?t=${job.frame}`} alt="YOLO annotated current video frame"/>
        ) : (
          <>
            <div className="scanlines"/>
            <div className="waiting-video">
              {busy ? <LoaderCircle className="spin"/> : <FileVideo size={38}/>}
              <b>{upload ? stateLabel : 'WAITING FOR VIDEO'}</b>
              <span>{upload ? 'Start analysis to run YOLO person detection.' : 'Choose a CCTV/video file to begin real AI analysis.'}</span>
            </div>
          </>
        )}
        <div className="camera-meta">
          <b>CAMERA-07</b>
          <span>ZONE: RAMGHAT</span>
          <span><Wifi size={13}/> {job ? 'AI processing stream' : 'Awaiting video source'}</span>
        </div>
        {job && (
          <div className="video-stats">
            <div><small>CURRENT DETECTED PEOPLE</small><b>{job.current_count}</b></div>
            <div><small>ACTIVE TRACKS</small><b>{typeof job.active_track_count === 'number' ? job.active_track_count : '—'}</b></div>
            <div><small>FPS</small><b>{job.fps || '—'}</b></div>
            <div><small>INFERENCE</small><b>{job.inference_ms ? `${job.inference_ms} ms` : '—'}</b></div>
            <div><small>MODEL / DEVICE</small><b>{job.model_name?.replace('.pt', '').toUpperCase()} · {job.device?.toUpperCase()}</b></div>
            <div><small>CONFIDENCE</small><b>{typeof job.confidence === 'number' ? job.confidence.toFixed(2) : '—'}</b></div>
            <div><small>INPUT SIZE</small><b>{job.input_size ? `${job.input_size}px` : '—'}</b></div>
          </div>
        )}
        <div className="analysis-status">
          {job ? (
            <>
              <span>Frame {job.frame} / {job.total_frames || '—'}</span>
              <span>{job.progress || 0}%</span>
            </>
          ) : (
            <span>Detection mode: person class only · confidence ≥ {settings.confidence.toFixed(2)} · {settings.model_name.replace('.pt', '').toUpperCase()} · {settings.input_size}px</span>
          )}
        </div>
      </div>
    </section>
  )
}
