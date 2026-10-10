/**
 * Static-replay mode (GitHub Pages). Built with VITE_STATIC=1: every API URL is mapped onto pre-recorded files under
 * `<base>/demo-data/` produced by `scripts/export_static_demo.py`, and live runs are replayed from a recorded timeline.
 * In the normal build `u()` is the identity and nothing here is used.
 */
export const STATIC = import.meta.env.VITE_STATIC === '1'
const D = `${import.meta.env.BASE_URL}demo-data`
/** Backend origin when the console is hosted separately (e.g. Vercel -> Hugging Face Space). Empty = same origin. */
export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')

/** WebSocket URL for an API path, honouring API_BASE. */
export function wsUrl(path: string): string {
  if (API_BASE) return API_BASE.replace(/^http/, 'ws') + path
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}${path}`
}

export function u(url: string): string {
  if (!url.startsWith('/api/')) return url
  if (!STATIC) return API_BASE + url
  const [path, query = ''] = url.split('?')
  const q = new URLSearchParams(query)
  const p = path.replace(/^\/api\//, '').split('/')
  if (p[0] === 'stores') {
    if (p.length === 1) return `${D}/stores.json`
    if (p.length === 2) return `${D}/stores/${p[1]}.json`
    if (p[2] === 'floorplan.png') return `${D}/stores/${p[1]}/floorplan.png`
    if (p[2] === 'cameras') return `${D}/stores/${p[1]}/${p[3]}_frame.jpg`
  }
  if (p[0] === 'runs') {
    if (p.length === 1) return `${D}/runs.json`
    const r = `${D}/runs/${p[1]}`
    if (p.length === 2) return `${r}/info.json`
    const name = p[2]
    if (name === 'heatmap') {
      const type = q.get('type') ?? 'occupancy'
      return q.get('format') === 'json' ? `${r}/heatmap_${type}_${q.get('window') ?? 'all'}.json` : `${r}/heatmap_${type}.png`
    }
    if (name === 'insights') return `${r}/insights_${q.get('status') ?? 'active'}.json`
    if (name === 'evidence') return `${r}/evidence/${p[3]}`
    if (name === 'stream.mjpg' || name === 'frame.jpg') return `${r}/frame.jpg`
    if (name.includes('.')) return `${r}/${name}`
    return `${r}/${name}.json`
  }
  return url
}

export const staticVideo = (runId: string) => `${D}/runs/${runId}/video.mp4`
export const staticTimeline = (runId: string) => `${D}/runs/${runId}/timeline.json`

export class StaticDemoError extends Error {
  constructor() {
    super('This is the static replay demo on GitHub Pages — run FlowSight locally (make demo) to start runs or edit config.')
  }
}
