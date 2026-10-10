import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { Empty, InsightCard, KpiStrip, NeedRun, Panel, useActiveInsights } from '../components/common'
import { useApp } from '../store'
import FloorCanvas from '../components/FloorCanvas'
import { STATIC, staticVideo } from '../static'

export default function LiveOps() {
  const { runId, kpis, finished, error, people, zones, storeId, setVideoClock } = useApp()
  const store = useQuery({ queryKey: ['store', storeId], queryFn: () => api.store(storeId), enabled: STATIC })
  const run = useQuery({ queryKey: ['run', runId], queryFn: () => api.run(runId!), enabled: !!runId, refetchInterval: 5000 })
  const active = useActiveInsights()
  const counts = active.reduce<Record<string, number>>((a, i) => ({ ...a, [i.horizon]: (a[i.horizon] ?? 0) + 1 }), {})
  const r = run.data
  return (
    <NeedRun>
      <div className="flex h-full min-h-[640px] flex-col gap-4">
        <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1.55fr)_minmax(380px,1fr)] gap-4">
          <Panel
            title={r?.is_sim ? 'Store view · simulator (top-down)' : r?.source.startsWith('multi') ? 'Multi-camera fusion · global IDs across cameras · heads blurred' : 'Camera · annotated, heads blurred'}
            right={r && (
              <span className="num flex items-center gap-3 font-mono normal-case tracking-normal text-ink-400">
                <span>{r.device}</span><span>{r.model}</span>
                {!r.is_sim && <span className={r.calibrated ? 'text-signal' : 'text-sev-high'}>{r.calibrated ? 'calibrated' : 'UNCALIBRATED'}</span>}
                {r.proc_fps > 0 && <span>{r.proc_fps} fps</span>}
              </span>
            )}
            bodyClass="p-2 flex items-center justify-center bg-ink-950/60"
          >
            {error ? <Empty>Pipeline error: {error}</Empty> : STATIC ? (
              r && !r.is_sim ? (
                <video key={runId} src={staticVideo(runId!)} autoPlay muted loop playsInline controls
                  onTimeUpdate={(e) => setVideoClock(e.currentTarget.currentTime)} className="h-full w-full rounded object-contain" />
              ) : store.data ? (
                <FloorCanvas store={store.data} people={people} live={zones} className="w-full" />
              ) : <Empty>Loading replay…</Empty>
            ) : (
              <img key={runId} src={`/api/runs/${runId}/stream.mjpg`} alt="live annotated stream"
                className="h-full w-full rounded object-contain [image-rendering:auto]" onError={(e) => { (e.target as HTMLImageElement).src = `/api/runs/${runId}/frame.jpg?${Date.now()}` }} />
            )}
          </Panel>
          <Panel title={<span className="flex items-center gap-2">Action feed <span className="chip bg-signal/15 text-signal">{active.length} active</span></span>}
            right={<span className="num normal-case tracking-normal text-ink-400">now {counts.now ?? 0} · today {counts.today ?? 0} · week {counts.this_week ?? 0}</span>}
            bodyClass="overflow-y-auto p-3 space-y-2.5">
            {active.length === 0 ? <Empty>{finished ? 'No open actions.' : 'Watching the floor — actions appear here as soon as analytics have evidence.'}</Empty> :
              active.map((i) => <InsightCard key={i.id} i={i} />)}
          </Panel>
        </div>
        <KpiStrip k={kpis} />
      </div>
    </NeedRun>
  )
}
