import { useQuery } from '@tanstack/react-query'
import { LOS_COLOR, api, fmtDur, pct, type AttentionZone } from '../api'
import { Empty, NeedRun, Panel } from '../components/common'
import { useApp } from '../store'

function ZoneList({ items, tone, empty }: { items: AttentionZone[]; tone: string; empty: string }) {
  if (!items.length) return <Empty>{empty}</Empty>
  return (
    <ul className="space-y-2">
      {items.map((z) => (
        <li key={z.zone_id} className="rounded-md border border-ink-700 bg-ink-850 p-3">
          <div className="flex items-center justify-between">
            <span className="font-medium text-ink-100">{z.zone_name}</span>
            <span className="num font-mono text-xs" style={{ color: tone }}>{pct(z.stopping_power)} stop</span>
          </div>
          <div className="mt-1 text-xs text-ink-400">{z.reason}</div>
        </li>
      ))}
    </ul>
  )
}

export default function LayoutAdvisor() {
  const runId = useApp((s) => s.runId)
  const q = useQuery({ queryKey: ['layout', runId], queryFn: () => api.layout(runId!), enabled: !!runId, refetchInterval: 6000 })
  const L = q.data
  const maxHi = Math.max(1, ...(L?.what_if ?? []).map((w) => w.uplift_high_per_day))
  return (
    <NeedRun>
      <div className="flex flex-col gap-4">
        <div className="rounded-lg border border-sev-medium/30 bg-sev-medium/5 px-4 py-2.5 text-xs text-sev-medium">
          What-if figures are <b>estimates</b>: expected engaged visits = exposure of the new location × stopping power (range from
          the promo prior to the target zone's own stopping power) − what the current location converts, scaled to a 12 h store day.
        </div>
        <Panel title="What-if: move under-performing displays to high-attention spots">
          {L?.what_if.length ? (
            <div className="space-y-3">
              {L.what_if.map((w, i) => (
                <div key={i} className="grid grid-cols-[1fr_auto_1fr_260px] items-center gap-4 rounded-lg border border-ink-700 bg-ink-850 p-4">
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">Current placement</div>
                    <div className="text-base font-semibold text-ink-100">{w.from_name}</div>
                    <div className="num text-xs text-ink-400">{w.from_exposures} passed · {pct(w.from_stopping_power)} stopped</div>
                  </div>
                  <svg viewBox="0 0 40 12" className="h-4 w-12 text-signal"><path d="M0 6h34M30 1l6 5-6 5" stroke="currentColor" strokeWidth="2" fill="none" /></svg>
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">Suggested placement</div>
                    <div className="text-base font-semibold text-signal">{w.to_name}</div>
                    <div className="num text-xs text-ink-400">{w.to_exposures} pass · {pct(w.to_stopping_power)} stopping power</div>
                  </div>
                  <div>
                    <div className="num flex items-baseline justify-between">
                      <span className="text-2xl font-semibold text-ink-100">+{Math.round(w.uplift_mid_per_day)}</span>
                      <span className="text-xs text-ink-400">engaged visits / day · est.</span>
                    </div>
                    <div className="relative mt-2 h-2 rounded bg-ink-700">
                      <div className="absolute h-2 rounded bg-signal/60" style={{ left: `${(w.uplift_low_per_day / maxHi) * 100}%`, width: `${Math.max(2, ((w.uplift_high_per_day - w.uplift_low_per_day) / maxHi) * 100)}%` }} />
                    </div>
                    <div className="num mt-1 flex justify-between text-[11px] text-ink-400"><span>{Math.round(w.uplift_low_per_day)}</span><span>range</span><span>{Math.round(w.uplift_high_per_day)}</span></div>
                  </div>
                </div>
              ))}
            </div>
          ) : <Empty>No what-if candidates: needs an IGNORED/UNVISITED promo and a HOT zone in the current window.</Empty>}
        </Panel>
        <div className="grid grid-cols-3 gap-4">
          <Panel title="Recurring bottlenecks" bodyClass="p-0">
            {L?.recurring_bottlenecks.length ? (
              <table className="num w-full text-sm">
                <thead><tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-wider text-ink-400">
                  <th className="px-3 py-2">Zone</th><th className="px-2 text-right">Episodes</th><th className="px-2 text-right">/hr</th><th className="px-2 text-right">Lost</th><th className="px-3 text-right">Worst</th></tr></thead>
                <tbody>
                  {L.recurring_bottlenecks.map((r) => (
                    <tr key={r.zone_id} className="border-b border-ink-800">
                      <td className="px-3 py-2 text-ink-100">{r.zone_name}</td><td className="px-2 text-right">{r.count}</td>
                      <td className="px-2 text-right">{r.per_hour.toFixed(1)}</td><td className="px-2 text-right">{fmtDur(r.time_lost_s)}</td>
                      <td className="px-3 text-right font-semibold" style={{ color: LOS_COLOR[r.worst_los] }}>{r.worst_los}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <Empty>No bottlenecks recorded.</Empty>}
            {runId && <img src={`/api/runs/${runId}/heatmap?type=stationary`} alt="stationary heatmap" className="block w-full border-t border-ink-700" />}
          </Panel>
          <Panel title={<span>Ignored zones <span className="normal-case tracking-normal text-ink-400">· merchandising problem</span></span>}>
            <ZoneList items={L?.ignored ?? []} tone="#a855f7" empty="No ignored zones: passers-by stop at normal rates." />
          </Panel>
          <Panel title={<span>Dead / unvisited zones <span className="normal-case tracking-normal text-ink-400">· layout problem</span></span>}>
            <ZoneList items={L?.unvisited ?? []} tone="#94a3b8" empty="No dead zones: every zone gets traffic." />
            {!!L?.hot.length && (
              <div className="mt-4">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-ink-400">Hot zones · place promotions nearby</div>
                <ZoneList items={L.hot} tone="#f97316" empty="" />
              </div>
            )}
          </Panel>
        </div>
      </div>
    </NeedRun>
  )
}
