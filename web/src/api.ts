import { STATIC, StaticDemoError, u } from './static'

export type LoS = 'A' | 'B' | 'C' | 'D' | 'E' | 'F'
export type Severity = 'critical' | 'high' | 'medium' | 'low'

export interface Camera { id: string; name: string; source: string; available: boolean; calibrated: boolean; approximate: boolean; frame_url?: string }
export interface StoreSummary { store_id: string; name: string; cameras: Camera[]; scenarios: string[]; multi?: boolean; width_m: number; height_m: number }
export interface ZoneCfg { id: string; name: string; type: string; polygon: number[][] }
export interface StoreDetail extends StoreSummary {
  floorplan: { image: string; width_m: number; height_m: number }
  zones: ZoneCfg[]
  floorplan_url: string
  cameras: (Camera & { calibration?: { image_points: number[][]; floor_points: number[][] } | null })[]
}
export interface RunInfo {
  run_id: string; store_id: string; source: string; speed: string; created: number; finished: boolean; progress: number
  error: string | null; device: string; model: string; calibrated: boolean; ts: number; is_sim: boolean; proc_fps: number
  frame_size: [number, number]
}
export interface Kpis {
  ts: number; elapsed_s: number; people_in_store: number; peak_people: number; visitors_total: number
  avg_zone_dwell_s: number; avg_time_in_store_s: number; avg_path_length_m: number; checkout_wait_s: number | null
  checkout_queue: number | null; active_bottlenecks: number; bottlenecks_total: number; conversion: number | null
  visits_total: number; id_stitches?: number; proc_fps?: number
  staff_on_floor?: number; staff_tracks?: number; staff_visits_purged?: number
}
export interface Person { id: number; x: number; y: number; speed: number; stationary_s: number }
export interface LiveZone { id: string; occupancy: number; los: LoS; density: number; dwell_live: number; bottleneck: boolean; status: string }
export interface Insight {
  id: string; created_ts: number; updated_ts: number; type: string; zone_id: string | null; zone_name: string | null
  severity: Severity; title: string; action: string; owner: string; horizon: string; expected_impact: string
  confidence: number; status: 'open' | 'acked' | 'resolved' | 'suppressed'; priority: number; people_affected: number
  evidence: { metrics: Record<string, unknown>; baseline_ratio: number | null; window: string; snapshot_path: string | null }
}
export interface LivePayload {
  ts: number; people: Person[]; zones: LiveZone[]; insights_delta: Insight[]; insights_version: number; kpis: Kpis
  finished: boolean; progress: number; error: string | null
}
export interface ZoneKpi {
  zone_id: string; name: string; type: string; area_m2: number; visits: number; unique_visitors: number
  median_dwell_s: number; mean_dwell_s: number; p90_dwell_s: number; total_engaged_s: number; stops: number; occupancy: number
  los: LoS; density: number; window_los: LoS; mean_speed: number | null; free_flow_mps: number; exposures: number
  stopping_power: number | null; attention_status: string; attention_reason: string; bottleneck_active: boolean
  dwell_hist: number[]; occupancy_series: [number, number, LoS][]; bottleneck_periods: [number, number][]
  funnel: { exposures: number; entries: number; stops: number; engaged_s: number }
}
export interface FlowSummary {
  transitions: { from: string; to: string; count: number }[]
  sankey: { nodes: { name: string }[]; links: { source: string; target: string; value: number }[] }
  top_paths: { path: string[]; names: string[]; count: number }[]
  entrance_visitors: number; converted: number; conversion: number | null; avg_path_length_m: number
  avg_time_in_store_s: number; tracks: number
}
export interface WhatIf {
  from_zone: string; from_name: string; to_zone: string; to_name: string; from_stopping_power: number | null
  to_stopping_power: number; from_exposures: number; to_exposures: number; current_stops: number
  uplift_low_per_day: number; uplift_high_per_day: number; uplift_mid_per_day: number; label: string
}
export interface AttentionZone {
  zone_id: string; zone_name: string; zone_type: string; exposures: number; entries: number; stops: number
  engaged_s: number; stopping_power: number | null; status: string; reason: string; traffic_share?: number | null
}
export interface RecurringBn { zone_id: string; zone_name: string; count: number; time_lost_s: number; people_affected: number; worst_los: LoS; per_hour: number }
export interface Layout { what_if: WhatIf[]; recurring_bottlenecks: RecurringBn[]; ignored: AttentionZone[]; unvisited: AttentionZone[]; hot: AttentionZone[] }
export interface Bottleneck {
  zone_id: string; zone_name: string; t_start: number; t_confirmed: number; t_end: number | null; duration_s: number | null
  peak_density: number; peak_los: LoS; people_affected: number; time_lost_s: number; reason: string; active: boolean
}
export interface Briefing { mode: 'llm' | 'template'; model: string | null; bullets: string[] }
export interface HeatGrid { type: string; cell_m: number; nx: number; ny: number; max: number; total_person_s: number; grid: number[][] }
export interface CalibResult { ok: boolean; saved: boolean; reprojection_error: Record<string, number>; preview: string | null }

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}
const json = (method: string, body: unknown): RequestInit => {
  if (STATIC) throw new StaticDemoError()
  return jsonInit(method, body)
}
const jsonInit = (method: string, body: unknown): RequestInit => ({
  method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
})

export const api = {
  stores: () => fetch(u('/api/stores')).then(j<StoreSummary[]>),
  store: (id: string) => fetch(u(`/api/stores/${id}`)).then(j<StoreDetail>),
  runs: () => fetch(u('/api/runs')).then(j<RunInfo[]>),
  run: (id: string) => fetch(u(`/api/runs/${id}`)).then(j<RunInfo>),
  startRun: (body: { store_id: string; source: string; speed: string | number; duration_s?: number }) =>
    fetch(u('/api/runs'), json('POST', body)).then(j<RunInfo>),
  stopRun: (id: string) => (STATIC ? Promise.reject(new StaticDemoError()) : fetch(u(`/api/runs/${id}`), { method: 'DELETE' }).then(j<{ ok: boolean }>)),
  zones: (id: string) => fetch(u(`/api/runs/${id}/zones`)).then(j<ZoneKpi[]>),
  flow: (id: string) => fetch(u(`/api/runs/${id}/flow`)).then(j<FlowSummary>),
  layout: (id: string) => fetch(u(`/api/runs/${id}/layout`)).then(j<Layout>),
  bottlenecks: (id: string) => fetch(u(`/api/runs/${id}/bottlenecks`)).then(j<Bottleneck[]>),
  insights: (id: string, status = 'all') => fetch(u(`/api/runs/${id}/insights?status=${status}`)).then(j<Insight[]>),
  ack: (id: string, iid: string) => fetch(u(`/api/runs/${id}/insights/${iid}/ack`), { method: STATIC ? 'GET' : 'POST' }).then(j<Insight>),
  briefing: (id: string) => fetch(u(`/api/runs/${id}/briefing`)).then(j<Briefing>),
  heat: (id: string, type: string, window?: number) =>
    fetch(u(`/api/runs/${id}/heatmap?format=json&type=${type}${window ? `&window=${window}` : ''}`)).then(j<HeatGrid>),
  putZones: (sid: string, zones: ZoneCfg[]) => fetch(u(`/api/stores/${sid}/zones`), json('PUT', { zones })).then(j<{ ok: boolean }>),
  putCalibration: (sid: string, body: { camera_id: string; image_points: number[][]; floor_points: number[][]; save: boolean }) =>
    fetch(u(`/api/stores/${sid}/calibration`), json('PUT', body)).then(j<CalibResult>),
}

export const LOS_COLOR: Record<string, string> = { A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#f97316', E: '#ef4444', F: '#b91c1c' }
export const SEV_COLOR: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#64748b' }
export const STATUS_COLOR: Record<string, string> = { HOT: '#f97316', IGNORED: '#a855f7', UNVISITED: '#64748b', NORMAL: '#334155', BOTTLENECK: '#ef4444' }
export const ZONE_TYPES = ['entrance', 'aisle', 'promo', 'walkway', 'checkout', 'service']

export function fmtDur(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return '–'
  s = Math.round(s)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return `${m}m${String(r).padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}m`
}
export const pct = (v: number | null | undefined, d = 0) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const label = (s: string) => s.replace(/_/g, ' ')
