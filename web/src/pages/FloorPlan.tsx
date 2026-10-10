import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { LOS_COLOR, api, fmtDur } from '../api'
import FloorCanvas from '../components/FloorCanvas'
import { Empty, NeedRun, Panel } from '../components/common'
import { useApp } from '../store'

const WINDOWS = [{ v: 0, l: 'All' }, { v: 300, l: '5 min' }, { v: 900, l: '15 min' }, { v: 1800, l: '30 min' }, { v: 3600, l: '60 min' }]

export default function FloorPlan() {
  const { runId, storeId, people, zones: live, ts } = useApp()
  const [heatType, setHeatType] = useState<'off' | 'occupancy' | 'stationary'>('occupancy')
  const [wIdx, setWIdx] = useState(0)
  const store = useQuery({ queryKey: ['store', storeId], queryFn: () => api.store(storeId) })
  const win = WINDOWS[wIdx].v
  const heat = useQuery({
    queryKey: ['heat', runId, heatType, win], queryFn: () => api.heat(runId!, heatType, win || undefined),
    enabled: !!runId && heatType !== 'off', refetchInterval: 3000,
  })
  const zk = useQuery({ queryKey: ['zones', runId], queryFn: () => api.zones(runId!), enabled: !!runId, refetchInterval: 5000 })
  const kmap = useMemo(() => Object.fromEntries((zk.data ?? []).map((z) => [z.zone_id, z])), [zk.data])
  return (
    <NeedRun>
      <div className="grid grid-cols-[minmax(0,1fr)_280px] gap-4">
        <Panel title="Floor plan · live positions in metres"
          right={
            <div className="flex items-center gap-2 normal-case tracking-normal">
              {(['off', 'occupancy', 'stationary'] as const).map((t) => (
                <button key={t} onClick={() => setHeatType(t)}
                  className={`rounded px-2 py-1 text-xs ${heatType === t ? 'bg-signal/15 text-signal' : 'text-ink-300 hover:text-ink-100'}`}>
                  {t === 'off' ? 'No heatmap' : t === 'occupancy' ? 'Occupancy heat' : 'Stationary heat'}
                </button>
              ))}
            </div>
          }>
          {store.data ? (
            <FloorCanvas store={store.data} people={people} live={live} heat={heatType === 'off' ? null : heat.data} kpis={kmap} />
          ) : <Empty>Loading floor plan…</Empty>}
          <div className="mt-3 flex items-center gap-4 text-xs text-ink-300">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">Heat window</span>
            <input type="range" min={0} max={WINDOWS.length - 1} value={wIdx} onChange={(e) => setWIdx(+e.target.value)} className="w-56 accent-teal-400" aria-label="Heatmap time window" />
            <span className="num font-mono text-ink-100">{WINDOWS[wIdx].l}</span>
            {heat.data && <span className="num text-ink-400">{Math.round(heat.data.total_person_s)} person-seconds · {heat.data.cell_m} m cells</span>}
          </div>
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel title="Legend">
            <div className="space-y-2 text-xs text-ink-300">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">Zone colour = Fruin level of service</div>
              <div className="grid grid-cols-3 gap-1">
                {Object.entries(LOS_COLOR).map(([k, c]) => (
                  <span key={k} className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ background: c }} />LoS {k}</span>
                ))}
              </div>
              <div className="pt-2 text-[10px] font-semibold uppercase tracking-wider text-ink-400">Shoppers</div>
              <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-teal-300" /> walking</div>
              <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-amber-400" /> stopped / browsing (&lt; 0.3 m/s)</div>
              <div className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm border-2 border-red-500" /> active bottleneck (pulsing)</div>
            </div>
          </Panel>
          <Panel title={`Now · t+${fmtDur(ts)}`} bodyClass="p-0">
            <table className="num w-full text-xs">
              <tbody>
                {store.data?.zones.map((z) => {
                  const l = live[z.id]
                  return (
                    <tr key={z.id} className="border-b border-ink-800">
                      <td className="px-3 py-1.5 text-ink-200">{z.name}</td>
                      <td className="px-2 text-right text-ink-100">{l?.occupancy ?? 0}</td>
                      <td className="px-3 text-right font-semibold" style={{ color: LOS_COLOR[l?.los ?? 'A'] }}>{l?.los ?? 'A'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>
    </NeedRun>
  )
}
