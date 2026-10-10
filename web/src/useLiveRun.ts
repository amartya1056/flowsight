import { useEffect } from 'react'
import { api, type LivePayload } from './api'
import { useApp } from './store'

/** One WebSocket per selected run; pushes live state (people, zones, KPIs, insight deltas) into the store. */
export function useLiveRun(runId: string | null) {
  const applyLive = useApp((s) => s.applyLive)
  const setConnected = useApp((s) => s.setConnected)
  const seedInsights = useApp((s) => s.seedInsights)

  useEffect(() => {
    if (!runId) return
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
