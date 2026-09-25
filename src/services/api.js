// This adapter keeps mock data and future FastAPI calls separate.
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
export async function checkHealth() { const response = await fetch(`${API_URL}/health`); if (!response.ok) throw new Error('Health check failed'); return response.json() }
async function request(path, options) { const response = await fetch(`${API_URL}${path}`, options); const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.detail || 'Request failed'); return body }
export function uploadVideo(file) { const form = new FormData(); form.append('file', file); return request('/video/upload', { method: 'POST', body: form }) }
export function startDetection(settings) { const form = new FormData(); Object.entries(settings).forEach(([key, value]) => form.append(key, value)); return request('/detection/start', { method: 'POST', body: form }) }
export function getDetectionStatus(jobId) { return request(`/detection/status/${jobId}`) }
export function mediaUrl(path) { return path ? `${API_URL.replace('/api', '')}${path}` : null }
