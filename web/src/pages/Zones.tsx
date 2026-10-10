import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { LOS_COLOR, STATUS_COLOR, api, fmtDur, pct, type ZoneKpi } from '../api'
import { Empty, NeedRun, Panel } from '../components/common'
import { useApp } from '../store'

type Key = keyof Pick<ZoneKpi, 'name' | 'visits' | 'median_dwell_s' | 'p90_dwell_s' | 'stopping_power' | 'exposures' | 'occupancy'>
const COLS: { k: Key; l: string; num?: boolean }[] = [
  { k: 'name', l: 'Zone' }, { k: 'visits', l: 'Visits', num: true }, { k: 'median_dwell_s', l: 'Median dwell', num: true },
  { k: 'p90_dwell_s', l: 'P90 dwell', num: true }, { k: 'exposures', l: 'Exposures', num: true },
  { k: 'stopping_power', l: 'Stopping power', num: true }, { k: 'occupancy', l: 'Now', num: true },
]

const AX = { axisLine: { lineStyle: { color: '#3a475a' } }, axisLabel: { color: '#8b97a8', fontSize: 10 }, splitLine: { lineStyle: { color: '#1b2430' } } }

function statusOf(z: ZoneKpi) {
  return z.bottleneck_active ? 'BOTTLENECK' : z.attention_status
}

function Drawer({ z, onClose }: { z: ZoneKpi; onClose: () => void }) {
  const bins = useMemo(() => {
    const d = z.dwell_hist
    if (!d.length) return { labels: [] as string[], counts: [] as number[] }
    const edges = [0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, Infinity]
    const counts = edges.slice(0, -1).map((e, i) => d.filter((v) => v >= e && v < edges[i + 1]).length)
    const labels = edges.slice(0, -1).map((e, i) => (edges[i + 1] === Infinity ? `${e}s+` : `${e}–${edges[i + 1]}s`))
    return { labels, counts }
  }, [z.dwell_hist])
  const f = z.funnel
  const funnel = {
    tooltip: { trigger: 'item' },
    series: [{
      type: 'funnel', left: 10, right: 10, top: 10, bottom: 10, sort: 'none', gap: 3, minSize: '18%',
      label: { position: 'inside', color: '#e3e8ef', formatter: '{b}: {c}', fontSize: 11 },
      data: [
        { name: 'Exposures', value: f.exposures, itemStyle: { color: '#1e3a5f' } },
        { name: 'Entries', value: f.entries, itemStyle: { color: '#14796d' } },
        { name: 'Stops (≥3 s engaged)', value: f.stops, itemStyle: { color: '#2dd4bf' } },
      ],
    }],
  }
  const occ = {
    grid: { left: 36, right: 12, top: 16, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value', ...AX, axisLabel: { ...AX.axisLabel, formatter: (v: number) => fmtDur(v) } },
    yAxis: { type: 'value', name: 'people', nameTextStyle: { color: '#8b97a8', fontSize: 10 }, ...AX },
    series: [{
      type: 'line', showSymbol: false, data: z.occupancy_series.map((p) => [p[0], p[1]]), lineStyle: { color: '#2dd4bf', width: 1.5 },
      areaStyle: { color: 'rgba(45,212,191,0.12)' },
      markArea: { itemStyle: { color: 'rgba(239,68,68,0.18)' }, data: z.bottleneck_periods.map(([a, b]) => [{ xAxis: a }, { xAxis: b }]) },
    }],
  }
  const hist = {
    grid: { left: 36, right: 12, top: 16, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: bins.labels, ...AX, axisLabel: { ...AX.axisLabel, rotate: 35 } },
    yAxis: { type: 'value', ...AX },
    series: [{ type: 'bar', data: bins.counts, itemStyle: { color: '#4f8fd9', borderRadius: [3, 3, 0, 0] } }],
  }
  return (
    <div className="fixed inset-y-0 right-0 z-30 flex w-[560px] flex-col border-l border-ink-600 bg-ink-900 shadow-2xl">
      <div className="flex items-center justify-between border-b border-ink-700 px-5 py-3">
        <div>
          <div className="text-base font-semibold">{z.name}</div>
          <div className="text-xs text-ink-400">{z.type} · {z.area_m2} m² · free-flow {z.free_flow_mps} m/s</div>
        </div>
        <button className="btn" onClick={onClose}>Close</button>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-5">
        {z.attention_reason && <div className="rounded-md border border-ink-700 bg-ink-850 p-3 text-sm text-ink-200">{statusOf(z)}: {z.attention_reason}</div>}
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-ink-400">Attention funnel (last 15 min)</div>
          <ReactECharts option={funnel} style={{ height: 170 }} />
          <div className="num text-xs text-ink-300">Stopping power {pct(z.stopping_power)} · engaged {fmtDur(f.engaged_s)} total</div>
        </div>
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-ink-400">Dwell-time distribution ({z.dwell_hist.length} visits)</div>
          {z.dwell_hist.length ? <ReactECharts option={hist} style={{ height: 180 }} /> : <Empty>No completed visits yet.</Empty>}
        </div>
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-ink-400">Occupancy timeline · bottleneck periods shaded</div>
          <ReactECharts option={occ} style={{ height: 180 }} />
        </div>
      </div>
    </div>
  )
}

export default function Zones() {
  const runId = useApp((s) => s.runId)
  const q = useQuery({ queryKey: ['zones', runId], queryFn: () => api.zones(runId!), enabled: !!runId, refetchInterval: 4000 })
  const [sort, setSort] = useState<{ k: Key; d: 1 | -1 }>({ k: 'visits', d: -1 })
  const [sel, setSel] = useState<string | null>(null)
  const rows = useMemo(() => [...(q.data ?? [])].sort((a, b) => {
    const va = a[sort.k] ?? -1, vb = b[sort.k] ?? -1
    return (va > vb ? 1 : va < vb ? -1 : 0) * sort.d
  }), [q.data, sort])
  const selected = q.data?.find((z) => z.zone_id === sel)
  return (
    <NeedRun>
      <Panel title="Zones · click a row for the funnel, dwell histogram and timeline" bodyClass="p-0">
        <table className="num w-full text-sm">
          <thead>
            <tr className="border-b border-ink-700 text-left text-[10px] uppercase tracking-wider text-ink-400">
              {COLS.map((c) => (
                <th key={c.k} className={`cursor-pointer px-4 py-2.5 font-semibold hover:text-ink-100 ${c.num ? 'text-right' : ''}`}
                  onClick={() => setSort((s) => ({ k: c.k, d: s.k === c.k ? (s.d === 1 ? -1 : 1) : -1 }))}>
                  {c.l}{sort.k === c.k ? (sort.d === 1 ? ' ▲' : ' ▼') : ''}
                </th>
              ))}
              <th className="px-4 py-2.5 text-right font-semibold">LoS</th>
              <th className="px-4 py-2.5 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((z) => {
              const st = statusOf(z)
              return (
                <tr key={z.zone_id} onClick={() => setSel(z.zone_id)} className="cursor-pointer border-b border-ink-800 hover:bg-ink-850">
                  <td className="px-4 py-2"><div className="font-medium text-ink-100">{z.name}</div><div className="text-[11px] text-ink-400">{z.type}</div></td>
                  <td className="px-4 text-right">{z.visits}</td>
                  <td className="px-4 text-right">{fmtDur(z.median_dwell_s)}</td>
                  <td className="px-4 text-right">{fmtDur(z.p90_dwell_s)}</td>
                  <td className="px-4 text-right">{z.exposures}</td>
                  <td className="px-4 text-right">{pct(z.stopping_power)}</td>
                  <td className="px-4 text-right">{z.occupancy}</td>
                  <td className="px-4 text-right font-semibold" style={{ color: LOS_COLOR[z.los] }}>{z.los}</td>
                  <td className="px-4">{st !== '-' && <span className="chip" style={{ background: (STATUS_COLOR[st] ?? '#334155') + '33', color: st === 'NORMAL' ? '#8b97a8' : STATUS_COLOR[st] }}>{st}</span>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {!rows.length && <Empty>No zone data yet.</Empty>}
      </Panel>
      {selected && <Drawer z={selected} onClose={() => setSel(null)} />}
    </NeedRun>
  )
}
