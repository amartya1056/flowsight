import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { Empty, InsightCard, KpiStrip, NeedRun, Panel, useActiveInsights } from '../components/common'
import { useApp } from '../store'
import { u } from '../static'

export default function Briefing() {
  const { runId, kpis } = useApp()
  const q = useQuery({ queryKey: ['briefing', runId], queryFn: () => api.briefing(runId!), enabled: !!runId, refetchInterval: 20000 })
  const active = useActiveInsights()
  return (
    <NeedRun>
      <div className="mx-auto flex max-w-5xl flex-col gap-4">
        <Panel title={<span className="flex items-center gap-2">Shift briefing
          {q.data && <span className={`chip ${q.data.mode === 'llm' ? 'bg-signal/15 text-signal' : 'bg-ink-700 text-ink-300'}`}>
            {q.data.mode === 'llm' ? `LLM · ${q.data.model}` : 'deterministic template'}</span>}</span>}
          right={runId && (
            <div className="flex gap-2 normal-case tracking-normal">
              <a className="btn" href={u(`/api/runs/${runId}/report.html`)} target="_blank" rel="noreferrer">Download daily report</a>
              <a className="btn" href={u(`/api/runs/${runId}/export.csv`)}>Zone CSV</a>
              <button className="btn" onClick={() => q.refetch()}>Regenerate</button>
            </div>
          )}>
          {q.data ? (
            <ol className="space-y-3">
              {q.data.bullets.map((b, i) => (
                <li key={i} className="flex gap-3">
                  <span className="num mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink-800 font-mono text-xs text-signal">{i + 1}</span>
                  <span className="text-[15px] leading-relaxed text-ink-100">{b}</span>
                </li>
              ))}
            </ol>
          ) : <Empty>Generating briefing…</Empty>}
          <div className="mt-4 text-[11px] text-ink-400">Numbers come only from FlowSight analytics; the optional LLM is instructed never to invent figures.</div>
        </Panel>
        <KpiStrip k={kpis} />
        <Panel title="Open actions" bodyClass="p-3 space-y-2">
          {active.length ? active.slice(0, 8).map((i) => <InsightCard key={i.id} i={i} compact />) : <Empty>No open actions.</Empty>}
        </Panel>
      </div>
    </NeedRun>
  )
}
