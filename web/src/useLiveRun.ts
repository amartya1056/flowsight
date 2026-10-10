import { useEffect } from 'react'
import { api, type Insight, type Kpis, type LivePayload, type LiveZone, type LoS, type Person } from './api'
import { useApp } from './store'
import { STATIC, staticTimeline } from './static'

/** One WebSocket per selected run; pushes live state (people, zones, KPIs, insight deltas) into the store.
 *  In the static (GitHub Pages) build, the run's recorded timeline is replayed in a loop instead. */
export function useLiveRun(runId: string | null) {
  const applyLive = useApp((s) => s.applyLive)
  const setConnected = useApp((s) => s.setConnected)
  const seedInsights = useApp((s) => s.seedInsights)

  useEffect(() => {
    if (!runId || !STATIC) return
    return replay(runId)
  }, [runId])

  useEffect(() => {
    if (!runId || STATIC) return
    let ws: WebSocket | null = null
    let closed = false
    let retry: number | undefined
    api.insights(runId, 'all').then(seedInsights).catch(() => {})
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/live`)
      ws.onopen = () => setConnected(true)
      ws.onmessage = (e) => {
        const p = JSON.parse(e.data) as LivePayload
        if (p && 'people' in p) applyLive(p)
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = window.setTimeout(connect, 1500)
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      ws?.close()
    }
  }, [runId, applyLive, setConnected, seedInsights])
}

// ---------------------------------------------------------------- replay
interface Frame {
  t: number
  p: [number, number, number, number, number][]
  z: [string, number, LoS, number, number, string][]
  k: Kpis
  i?: Insight[]
}
interface Timeline { step: number; video: boolean; duration: number; frames: Frame[] }

function replay(runId: string): () => void {
  const st = useApp.getState()
  let stopped = false
  let timer: number | undefined
  fetch(staticTimeline(runId))
    .then((r) => r.json() as Promise<Timeline>)
    .then((tl) => {
      if (stopped || !tl.frames.length) return
      st.setConnected(true)
      // start a little before the busiest moment (most active actions) so visitors see the story quickly
      let best = 0, bestN = -1, cur = 0
      tl.frames.forEach((f, k) => {
        if (f.i) cur = f.i.length
        if (cur > bestN) { bestN = cur; best = k }
      })
      let idx = tl.video ? 0 : Math.max(0, best - Math.round(tl.frames.length * 0.08))
      let lastIns = -1
      const show = (k: number) => {
        // carry the most recent insight list forward (it is only stored when it changes)
        let ins = k
        while (ins >= 0 && !tl.frames[ins].i) ins--
        const f = tl.frames[k]
        const people: Person[] = f.p.map(([id, x, y, speed, stationary_s]) => ({ id, x, y, speed, stationary_s }))
        const zones: LiveZone[] = f.z.map(([id, occupancy, los, dwell_live, bn, status]) =>
          ({ id, occupancy, los, density: 0, dwell_live, bottleneck: !!bn, status }))
        useApp.getState().applyLive({ ts: f.t, people, zones, kpis: f.k, insights_delta: [], insights_version: 0,
          finished: false, progress: f.t / Math.max(tl.duration, 1), error: null })
        if (ins !== lastIns) {
          useApp.getState().replaceInsights(ins >= 0 ? tl.frames[ins].i! : [])
          lastIns = ins
        }
      }
      const tick = () => {
        if (stopped) return
        const s = useApp.getState()
        if (tl.video && s.videoClock != null && Date.now() - s.videoClockAt < 1500) {
          // follow the <video> element when it is on screen
          const t = s.videoClock
          let k = Math.min(tl.frames.length - 1, Math.max(0, Math.round(t / tl.step)))
          while (k > 0 && tl.frames[k].t > t) k--
          idx = k
        } else {
          idx = (idx + 1) % tl.frames.length  // loop
        }
        show(idx)
        timer = window.setTimeout(tick, tl.video ? tl.step * 1000 : 250)
      }
      show(idx)
      tick()
    })
    .catch(() => st.setConnected(false))
  return () => {
    stopped = true
    window.clearTimeout(timer)
  }
}
