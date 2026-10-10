import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { api, fmtDur, pct } from '../api'
import { Empty, NeedRun, Panel, Stat } from '../components/common'
import { useApp } from '../store'

const PALETTE = ['#2dd4bf', '#4f8fd9', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#60a5fa', '#f97316', '#e879f9', '#94a3b8']

export default function Journeys() {
  const runId = useApp((s) => s.runId)
  const q = useQuery({ queryKey: ['flow', runId], queryFn: () => api.flow(runId!), enabled: !!runId, refetchInterval: 5000 })
  const f = q.data
  const zoneNames = Array.from(new Set((f?.sankey.nodes ?? []).map((n) => n.name.split(' · ')[0])))
  const option = f && {
    tooltip: { trigger: 'item', triggerOn: 'mousemove' },
    series: [{
      type: 'sankey', left: 8, right: 140, top: 10, bottom: 10, nodeGap: 10, nodeWidth: 12, draggable: false,
      emphasis: { focus: 'adjacency' }, layoutIterations: 64,
      data: f.sankey.nodes.map((n) => ({ ...n, itemStyle: { color: PALETTE[zoneNames.indexOf(n.name.split(' · ')[0]) % PALETTE.length], borderWidth: 0 } })),
      links: f.sankey.links,
      lineStyle: { color: 'gradient', opacity: 0.35, curveness: 0.5 },
      label: { color: '#e3e8ef', fontSize: 11 },
    }],
  }
  return (
    <NeedRun>
      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-4 gap-3">
          <Stat label="Entrance → checkout" value={pct(f?.conversion)} sub={f ? `${f.converted} of ${f.entrance_visitors} shoppers` : ''} />
          <Stat label="Tracked journeys" value={f?.tracks ?? '–'} />
          <Stat label="Avg path length" value={f ? `${f.avg_path_length_m.toFixed(1)} m` : '–'} sub="floor metres, not pixels" />
          <Stat label="Avg time in store" value={fmtDur(f?.avg_time_in_store_s)} />
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)_360px] gap-4">
          <Panel title="Customer journeys · step-layered Sankey (zone · stop #)">
            {option && f!.sankey.links.length ? <ReactECharts option={option} style={{ height: 560 }} /> : <Empty>Not enough completed journeys yet.</Empty>}
          </Panel>
          <Panel title="Top paths" bodyClass="p-0">
            <ol className="divide-y divide-ink-800">
              {f?.top_paths.map((p, i) => (
                <li key={i} className="flex items-start gap-3 px-4 py-2.5">
                  <span className="num mt-0.5 w-5 text-right font-mono text-xs text-ink-400">{i + 1}</span>
                  <div className="min-w-0 flex-1 text-xs leading-relaxed text-ink-200">{p.names.join(' → ')}</div>
                  <span className="num font-mono text-sm text-signal">{p.count}</span>
                </li>
              ))}
            </ol>
            {!f?.top_paths.length && <Empty>No multi-zone paths yet.</Empty>}
          </Panel>
        </div>
      </div>
    </NeedRun>
  )
}
