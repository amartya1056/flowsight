import { create } from 'zustand'
import type { Insight, Kpis, LivePayload, LiveZone, Person } from './api'

interface AppState {
  storeId: string
  runId: string | null
  connected: boolean
  ts: number
  progress: number
  finished: boolean
  error: string | null
  people: Person[]
  zones: Record<string, LiveZone>
  kpis: Kpis | null
  insights: Record<string, Insight>
  setStore: (s: string) => void
  setRun: (r: string | null) => void
  applyLive: (p: LivePayload) => void
  seedInsights: (list: Insight[]) => void
  upsertInsight: (i: Insight) => void
  setConnected: (c: boolean) => void
  replaceInsights: (list: Insight[]) => void
  videoClock: number | null
  videoClockAt: number
  setVideoClock: (t: number) => void
}

const fresh = { people: [], zones: {}, kpis: null, insights: {}, ts: 0, progress: 0, finished: false, error: null }

export const useApp = create<AppState>((set) => ({
  storeId: 'demo_store',
  runId: null,
  connected: false,
  ...fresh,
  setStore: (storeId) => set({ storeId }),
  setRun: (runId) => set({ runId, ...fresh }),
  setConnected: (connected) => set({ connected }),
  applyLive: (p) =>
    set((s) => {
      const insights = { ...s.insights }
      for (const i of p.insights_delta) insights[i.id] = i
      const zones: Record<string, LiveZone> = {}
      for (const z of p.zones) zones[z.id] = z
      return { ts: p.ts, people: p.people, zones, kpis: p.kpis, insights, progress: p.progress, finished: p.finished, error: p.error }
    }),
  seedInsights: (list) =>
    set((s) => {
      const insights = { ...s.insights }
      for (const i of list) if (!insights[i.id] || insights[i.id].updated_ts <= i.updated_ts) insights[i.id] = i
      return { insights }
    }),
  upsertInsight: (i) => set((s) => ({ insights: { ...s.insights, [i.id]: i } })),
  replaceInsights: (list) =>
    set((s) => {
      // keep local acks across replay updates
      const insights: Record<string, Insight> = {}
      for (const i of list) insights[i.id] = s.insights[i.id]?.status === 'acked' ? { ...i, status: 'acked' } : i
      return { insights }
    }),
  videoClock: null,
  videoClockAt: 0,
  setVideoClock: (t) => set({ videoClock: t, videoClockAt: Date.now() }),
}))
