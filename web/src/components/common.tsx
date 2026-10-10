import type { ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { SEV_COLOR, api, fmtDur, label, pct, type Insight, type Kpis } from '../api'
import { useApp } from '../store'
import { STATIC, u } from '../static'

export function Panel({ title, right, children, className, bodyClass }: { title: ReactNode; right?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={`panel flex min-h-0 flex-col ${className ?? ''}`}>
      <div className="panel-h"><span>{title}</span>{right}</div>
      <div className={`min-h-0 flex-1 ${bodyClass ?? 'p-4'}`}>{children}</div>
    </section>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-[120px] items-center justify-center p-6 text-center text-sm text-ink-400">{children}</div>
}

export function NeedRun({ children }: { children: ReactNode }) {
  const runId = useApp((s) => s.runId)
  if (!runId) return <Empty>Start a run from the top bar (camera or simulator scenario) to see live analytics.</Empty>
  return <>{children}</>
}

export function Stat({ label: l, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'warn' | 'bad' | 'ok' }) {
  const c = tone === 'bad' ? 'text-sev-critical' : tone === 'warn' ? 'text-sev-high' : 'text-ink-100'
  return (
    <div className="panel flex flex-col justify-between px-4 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">{l}</div>
      <div className={`num mt-1 text-2xl font-semibold ${c}`}>{value}</div>
      {sub && <div className="num mt-0.5 text-[11px] text-ink-400">{sub}</div>}
    </div>
  )
}

export function KpiStrip({ k }: { k: Kpis | null }) {
  const wait = k?.checkout_wait_s
  return (
    <div className="grid grid-cols-5 gap-3">
      <Stat label="People in store" value={k?.people_in_store ?? '–'}
        sub={k ? `peak ${k.peak_people} · ${k.visitors_total} tracked` + (k.staff_tracks ? ` · ${k.staff_on_floor ?? 0} staff excluded (${k.staff_visits_purged ?? 0} visits purged)` : '') : ''} />
      <Stat label="Avg zone dwell" value={k ? fmtDur(k.avg_zone_dwell_s) : '–'} sub={k ? `${fmtDur(k.avg_time_in_store_s)} avg time in store` : ''} />
      <Stat label="Checkout wait" value={wait == null ? '–' : fmtDur(wait)} tone={wait != null && wait > 180 ? 'bad' : wait != null && wait > 120 ? 'warn' : undefined}
        sub={k?.checkout_queue != null ? `${k.checkout_queue} in queue now · Little's law` : 'no checkout in view'} />
      <Stat label="Active bottlenecks" value={k?.active_bottlenecks ?? '–'} tone={k && k.active_bottlenecks > 0 ? 'bad' : undefined}
        sub={k ? `${k.bottlenecks_total} episodes so far` : ''} />
      <Stat label="Conversion" value={pct(k?.conversion)} sub="entrance → checkout" />
    </div>
  )
}

const OWNER: Record<string, string> = { floor_staff: 'Floor staff', store_manager: 'Store manager', merchandising: 'Merchandising', layout_planning: 'Layout planning' }

export function InsightCard({ i, compact }: { i: Insight; compact?: boolean }) {
  const { runId, upsertInsight } = useApp()
  const qc = useQueryClient()
  const ratio = i.evidence.baseline_ratio
  const metrics = Object.entries(i.evidence.metrics).filter(([, v]) => typeof v === 'number' || typeof v === 'string').slice(0, 4)
  const ack = async () => {
    if (!runId) return
    if (STATIC) return upsertInsight({ ...i, status: 'acked' })  // replay mode: acknowledge locally
    upsertInsight(await api.ack(runId, i.id))
    qc.invalidateQueries({ queryKey: ['insights'] })
  }
  return (
    <article className={`relative overflow-hidden rounded-lg border bg-ink-850 ${i.status === 'resolved' ? 'border-ink-700 opacity-60' : 'border-ink-600'}`}>
      <div className="absolute inset-y-0 left-0 w-1" style={{ background: SEV_COLOR[i.severity] }} />
      <div className="flex gap-3 p-3 pl-4">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-1.5">
            <span className="chip" style={{ background: SEV_COLOR[i.severity] + '22', color: SEV_COLOR[i.severity] }}>{i.severity}</span>
            <span className="chip bg-ink-700 text-ink-200">{label(i.horizon)}</span>
            <span className="chip bg-ink-800 text-ink-300">{OWNER[i.owner] ?? i.owner}</span>
            {ratio != null && ratio >= 1.2 && <span className="chip bg-sev-high/15 text-sev-high">×{ratio.toFixed(1)} vs normal</span>}
            {i.status === 'acked' && <span className="chip bg-signal/15 text-signal">acked</span>}
            {i.status === 'resolved' && <span className="chip bg-ink-700 text-ink-300">resolved</span>}
          </div>
          <div className="text-sm font-medium leading-snug text-ink-100">{i.action}</div>
          {!compact && <div className="mt-1 text-xs text-ink-400">Impact: {i.expected_impact}</div>}
          {!compact && metrics.length > 0 && (
            <div className="num mt-2 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-ink-300">
              {metrics.map(([k, v]) => (
                <span key={k}>{label(k)}: <span className="text-ink-100">{typeof v === 'number' ? (Math.abs(v) < 1 && v !== 0 ? v.toFixed(2) : Math.round(v * 10) / 10) : String(v)}</span></span>
              ))}
            </div>
          )}
          <div className="mt-2 flex items-center gap-3 text-[11px] text-ink-400">
            <span>conf {pct(i.confidence)}</span>
            <span>{i.evidence.window}</span>
            <span>at t+{fmtDur(i.created_ts)}</span>
            {i.status === 'open' && <button onClick={ack} className="ml-auto rounded border border-ink-600 px-2 py-0.5 text-ink-200 hover:border-signal hover:text-signal">Ack</button>}
          </div>
        </div>
        {!compact && i.evidence.snapshot_path && runId && (
          <a href={u(`/api/runs/${runId}/${i.evidence.snapshot_path}`)} target="_blank" rel="noreferrer" className="shrink-0">
            <img src={u(`/api/runs/${runId}/${i.evidence.snapshot_path}`)} alt="evidence (privacy-blurred)" className="h-20 w-28 rounded border border-ink-700 object-cover" />
          </a>
        )}
      </div>
    </article>
  )
}

export function useActiveInsights() {
  const insights = useApp((s) => s.insights)
  return Object.values(insights)
    .filter((i) => i.status === 'open' || i.status === 'acked')
    .sort((a, b) => b.priority - a.priority)
}
