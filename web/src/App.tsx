import { useEffect, useMemo, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, fmtDur, label } from './api'
import { useApp } from './store'
import { useLiveRun } from './useLiveRun'
import { STATIC } from './static'
import LiveOps from './pages/LiveOps'
import FloorPlan from './pages/FloorPlan'
import Zones from './pages/Zones'
import Journeys from './pages/Journeys'
import LayoutAdvisor from './pages/LayoutAdvisor'
import Setup from './pages/Setup'
import Briefing from './pages/Briefing'

const NAV = [
  { to: '/', label: 'Live Ops', icon: 'M3 12h4l3-8 4 16 3-8h4' },
  { to: '/floor', label: 'Floor Plan', icon: 'M3 3h18v18H3zM3 9h18M9 21V9' },
  { to: '/zones', label: 'Zones', icon: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z' },
  { to: '/journeys', label: 'Journeys', icon: 'M4 6c6 0 6 12 16 12M4 18c6 0 6-12 16-12' },
  { to: '/layout', label: 'Layout Advisor', icon: 'M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5' },
  { to: '/briefing', label: 'Shift Briefing', icon: 'M6 3h9l4 4v14H6zM9 12h7M9 16h7M9 8h3' },
  { to: '/setup', label: 'Setup', icon: 'M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v3M12 19v3M2 12h3M19 12h3' },
]

function Icon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  )
}

function Clock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  return <span className="num font-mono text-ink-200">{now.toLocaleTimeString([], { hour12: false })}</span>
}

function TopBar() {
  const qc = useQueryClient()
  const { storeId, runId, setStore, setRun, connected, ts, progress, finished } = useApp()
  const stores = useQuery({ queryKey: ['stores'], queryFn: api.stores })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 4000 })
  const store = stores.data?.find((s) => s.store_id === storeId)
  const sources = useMemo(() => {
    if (!store) return []
    return [
      ...(store.multi ? [{ value: 'multi', label: `Multi-camera fusion · ${store.cameras.length} cams`, disabled: false }] : []),
      ...store.cameras.map((c) => ({ value: c.id, label: `Camera · ${c.name}${c.available ? '' : ' (missing video)'}`, disabled: !c.available })),
      ...store.scenarios.map((s) => ({ value: `sim:${s}`, label: `Simulator · ${label(s)}`, disabled: false })),
    ]
  }, [store])
  const [source, setSource] = useState('')
  const [speed, setSpeed] = useState('4')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (sources.length && !sources.find((s) => s.value === source)) setSource(sources.find((s) => !s.disabled)?.value ?? '')
  }, [sources, source])

  // auto-select the newest run for this store (e.g. the demo runs) when nothing is selected
  useEffect(() => {
    if (runId || !runs.data?.length) return
    const r = runs.data.find((x) => x.store_id === storeId) ?? runs.data[0]
    if (r.store_id !== storeId) setStore(r.store_id)
    setRun(r.run_id)
  }, [runs.data, runId, storeId, setRun, setStore])

  const current = runs.data?.find((r) => r.run_id === runId)
  const start = async () => {
    setBusy(true)
    try {
      const r = await api.startRun({ store_id: storeId, source, speed })
      await qc.invalidateQueries({ queryKey: ['runs'] })
      setRun(r.run_id)
      qc.invalidateQueries({ queryKey: ['runs'] })
    } catch (e) {
      alert(String(e))
    } finally {
      setBusy(false)
    }
  }
  const stop = async () => {
    if (!runId) return
    await api.stopRun(runId)
    qc.invalidateQueries({ queryKey: ['runs'] })
  }

  return (
    <header className="flex h-14 min-w-0 shrink-0 items-center gap-2.5 overflow-hidden border-b border-ink-700 bg-ink-900/80 px-4 backdrop-blur">
      <select className="select max-w-[230px]" value={storeId} onChange={(e) => { setStore(e.target.value); setRun(null) }} aria-label="Store">
        {stores.data?.map((s) => <option key={s.store_id} value={s.store_id}>{s.name}</option>)}
      </select>
      <div className="h-6 w-px bg-ink-700" />
      {!STATIC && <>
      <select className="select max-w-[210px]" value={source} onChange={(e) => setSource(e.target.value)} aria-label="Source">
        {sources.map((s) => <option key={s.value} value={s.value} disabled={s.disabled}>{s.label}</option>)}
      </select>
      <select className="select" value={speed} onChange={(e) => setSpeed(e.target.value)} aria-label="Speed">
        {['1', '2', '4', 'max'].map((s) => <option key={s} value={s}>{s === 'max' ? 'max' : `${s}×`}</option>)}
      </select>
      </>}
      {STATIC ? (
        <a className="whitespace-nowrap rounded border border-sev-medium/50 bg-sev-medium/10 px-2.5 py-1.5 text-xs font-medium text-sev-medium"
          href="https://github.com/amartya1056/flowsight#2-quick-start--run-it-on-localhost" target="_blank" rel="noreferrer"
          title="Recorded runs replayed in your browser. Run FlowSight locally (make demo) for live processing.">
          Static replay demo · run locally for live video
        </a>
      ) : <>
      <button className="btn btn-primary" onClick={start} disabled={busy || !source}>
        <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M7 4l13 8-13 8z" /></svg> Start
      </button>
      <button className="btn" onClick={stop} disabled={!runId || current?.finished}>
        <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><rect x="6" y="6" width="12" height="12" /></svg> Stop
      </button>
      </>}
      <div className="h-6 w-px bg-ink-700" />
      <select className="select max-w-[230px] font-mono text-xs" value={runId ?? ''} aria-label="Run"
        onChange={(e) => {
          const r = runs.data?.find((x) => x.run_id === e.target.value)
          if (r && r.store_id !== storeId) setStore(r.store_id)
          setRun(e.target.value || null)
        }}>
        <option value="">— select run —</option>
        {runs.data?.map((r) => (
          <option key={r.run_id} value={r.run_id}>{r.source} · {r.finished ? 'done' : 'live'} · {r.run_id.slice(-11)}</option>
        ))}
      </select>
      <div className="ml-auto flex shrink-0 items-center gap-3 text-xs">
        {runId && (
          <span className="flex items-center gap-2 text-ink-300">
            <span className={`h-2 w-2 rounded-full ${connected && !finished ? 'bg-signal animate-pulse' : finished ? 'bg-ink-400' : 'bg-sev-high'}`} />
            <span className="num font-mono">t+{fmtDur(ts)}</span>
            {!finished && progress > 0 && <span className="num text-ink-400">{(progress * 100).toFixed(0)}%</span>}
            {finished && <span className="text-ink-400">finished</span>}
          </span>
        )}
        <span className="flex items-center gap-1.5 rounded-full border border-signal-dim/60 bg-signal/5 px-2.5 py-1 text-signal">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2}><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" /></svg>
          Privacy: anonymised
        </span>
        <Clock />
      </div>
    </header>
  )
}

export default function App() {
  const runId = useApp((s) => s.runId)
  useLiveRun(runId)
  return (
    <div className="flex h-full">
      <nav className="flex w-52 shrink-0 flex-col border-r border-ink-700 bg-ink-900">
        <div className="flex h-14 items-center gap-2 border-b border-ink-700 px-4">
          <svg viewBox="0 0 32 32" className="h-7 w-7"><rect width="32" height="32" rx="7" fill="#131a24" /><path d="M6 22 C 12 8, 20 26, 26 10" stroke="#2dd4bf" strokeWidth="3" fill="none" strokeLinecap="round" /></svg>
          <div>
            <div className="text-sm font-bold tracking-wide">FlowSight</div>
            <div className="text-[10px] uppercase tracking-widest text-ink-400">Store Ops Console</div>
          </div>
        </div>
        <div className="flex flex-1 flex-col gap-0.5 p-2">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'}
              className={({ isActive }) => `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${isActive ? 'bg-ink-800 text-signal' : 'text-ink-300 hover:bg-ink-800/60 hover:text-ink-100'}`}>
              <Icon d={n.icon} /> {n.label}
            </NavLink>
          ))}
        </div>
        <div className="border-t border-ink-700 p-3 text-[10px] leading-relaxed text-ink-400">
          Metric floor-plan analytics from existing CCTV. No faces, no identities — anonymous trajectories in metres.
        </div>
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="min-h-0 flex-1 overflow-auto p-4">
          <Routes>
            <Route path="/" element={<LiveOps />} />
            <Route path="/floor" element={<FloorPlan />} />
            <Route path="/zones" element={<Zones />} />
            <Route path="/journeys" element={<Journeys />} />
            <Route path="/layout" element={<LayoutAdvisor />} />
            <Route path="/briefing" element={<Briefing />} />
            <Route path="/setup" element={<Setup />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
