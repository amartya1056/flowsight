import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ZONE_TYPES, api, type CalibResult, type StoreDetail, type ZoneCfg } from '../api'
import { Empty, Panel } from '../components/common'
import { useApp } from '../store'
import { STATIC, u } from '../static'

const TYPE_COLOR: Record<string, string> = { entrance: '#4f8fd9', aisle: '#94a3b8', promo: '#d946ef', walkway: '#64748b', checkout: '#f59e0b', service: '#2dd4bf' }
const snap = (v: number) => Math.round(v * 20) / 20

function svgPoint(svg: SVGSVGElement, e: { clientX: number; clientY: number }) {
  const p = svg.createSVGPoint()
  p.x = e.clientX
  p.y = e.clientY
  const r = p.matrixTransform(svg.getScreenCTM()!.inverse())
  return [r.x, r.y]
}

function ZoneEditor({ store }: { store: StoreDetail }) {
  const qc = useQueryClient()
  const [zones, setZones] = useState<ZoneCfg[]>(store.zones)
  const [sel, setSel] = useState<string | null>(null)
  const [drawing, setDrawing] = useState<number[][] | null>(null)
  const [drag, setDrag] = useState<{ z: string; i: number } | null>(null)
  const [msg, setMsg] = useState('')
  const svg = useRef<SVGSVGElement>(null)
  const W = store.floorplan.width_m, H = store.floorplan.height_m
  useEffect(() => setZones(store.zones), [store])
  const zone = zones.find((z) => z.id === sel)
  const update = (id: string, patch: Partial<ZoneCfg>) => setZones((zs) => zs.map((z) => (z.id === id ? { ...z, ...patch } : z)))

  const onClick = (e: React.MouseEvent) => {
    if (!drawing || !svg.current) return
    const [x, y] = svgPoint(svg.current, e)
    setDrawing([...drawing, [snap(x), snap(y)]])
  }
  const finish = () => {
    if (!drawing || drawing.length < 3) return setDrawing(null)
    const id = `z_new_${Date.now().toString(36).slice(-4)}`
    setZones((zs) => [...zs, { id, name: 'New zone', type: 'aisle', polygon: drawing }])
    setSel(id)
    setDrawing(null)
  }
  const onMove = (e: React.PointerEvent) => {
    if (!drag || !svg.current) return
    const [x, y] = svgPoint(svg.current, e)
    setZones((zs) => zs.map((z) => (z.id !== drag.z ? z : { ...z, polygon: z.polygon.map((p, i) => (i === drag.i ? [snap(Math.min(W, Math.max(0, x))), snap(Math.min(H, Math.max(0, y)))] : p)) })))
  }
  const save = async () => {
    await api.putZones(store.store_id, zones)
    setMsg(`Saved ${zones.length} zones to config/stores/${store.store_id}.yaml — new runs use them.`)
    qc.invalidateQueries({ queryKey: ['store', store.store_id] })
  }
  return (
    <Panel title="Zone editor · floor metres"
      right={<div className="flex gap-2 normal-case tracking-normal">
        {drawing ? <><button className="btn btn-primary" onClick={finish}>Finish polygon ({drawing.length})</button><button className="btn" onClick={() => setDrawing(null)}>Cancel</button></>
          : <button className="btn" onClick={() => { setDrawing([]); setSel(null) }}>+ Draw zone</button>}
        <button className="btn btn-primary" onClick={save} disabled={STATIC} title={STATIC ? 'Saving needs the local backend' : ''}>Save zones</button>
      </div>}>
      <div className="grid grid-cols-[minmax(0,1fr)_260px] gap-4">
        <div>
          <svg ref={svg} viewBox={`0 0 ${W} ${H}`} style={{ maxHeight: 620 }} className={`mx-auto block w-full rounded-md bg-ink-950 ${drawing ? 'cursor-crosshair' : ''}`}
            onClick={onClick} onDoubleClick={finish} onPointerMove={onMove} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
            <image href={u(store.floorplan_url)} x={0} y={0} width={W} height={H} preserveAspectRatio="none" />
            {zones.map((z) => (
              <g key={z.id} onClick={(e) => { if (!drawing) { e.stopPropagation(); setSel(z.id) } }}>
                <polygon points={z.polygon.map((p) => p.join(',')).join(' ')} fill={TYPE_COLOR[z.type] + (z.id === sel ? '55' : '22')}
                  stroke={z.id === sel ? '#2dd4bf' : TYPE_COLOR[z.type]} strokeWidth={z.id === sel ? 0.08 : 0.04} className="cursor-pointer" />
                <text x={Math.min(...z.polygon.map((p) => p[0])) + 0.12} y={Math.min(...z.polygon.map((p) => p[1])) + 0.4} fontSize={Math.max(W, H) / 60} fill="#e3e8ef" pointerEvents="none">{z.name}</text>
              </g>
            ))}
            {zone && zone.polygon.map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r={0.16} fill="#2dd4bf" stroke="#07090d" strokeWidth={0.04} className="cursor-move"
                onPointerDown={(e) => { e.stopPropagation(); (e.target as Element).setPointerCapture(e.pointerId); setDrag({ z: zone.id, i }) }} />
            ))}
            {drawing && drawing.length > 0 && (
              <polyline points={drawing.map((p) => p.join(',')).join(' ')} fill="#2dd4bf22" stroke="#2dd4bf" strokeWidth={0.06} strokeDasharray="0.2 0.1" />
            )}
            {drawing?.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={0.12} fill="#2dd4bf" />)}
          </svg>
          <div className="mt-2 text-xs text-ink-400">{drawing ? 'Click to add vertices (snaps to 5 cm), double-click or “Finish” to close.' : 'Select a zone to edit; drag its vertices.'} {msg && <span className="text-signal">{msg}</span>}</div>
        </div>
        <div className="space-y-3">
          {zone ? (
            <div className="space-y-2 rounded-md border border-ink-700 bg-ink-850 p-3">
              <label className="block text-[10px] font-semibold uppercase tracking-wider text-ink-400">Name
                <input className="input mt-1 w-full" value={zone.name} onChange={(e) => update(zone.id, { name: e.target.value })} /></label>
              <label className="block text-[10px] font-semibold uppercase tracking-wider text-ink-400">Type
                <select className="select mt-1 w-full" value={zone.type} onChange={(e) => update(zone.id, { type: e.target.value })}>
                  {ZONE_TYPES.map((t) => <option key={t}>{t}</option>)}
                </select></label>
              <div className="num font-mono text-[11px] text-ink-400">{zone.id} · {zone.polygon.length} vertices</div>
              <button className="btn w-full justify-center text-sev-critical" onClick={() => { setZones((zs) => zs.filter((z) => z.id !== zone.id)); setSel(null) }}>Delete zone</button>
            </div>
          ) : <div className="rounded-md border border-dashed border-ink-700 p-3 text-xs text-ink-400">No zone selected.</div>}
          <ul className="max-h-80 space-y-1 overflow-y-auto text-sm">
            {zones.map((z) => (
              <li key={z.id}><button onClick={() => setSel(z.id)} className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left ${z.id === sel ? 'bg-ink-800 text-signal' : 'text-ink-200 hover:bg-ink-850'}`}>
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: TYPE_COLOR[z.type] }} />{z.name}<span className="ml-auto text-[10px] text-ink-400">{z.type}</span></button></li>
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  )
}

function Calibration({ store }: { store: StoreDetail }) {
  const cams = store.cameras.filter((c) => c.source !== 'sim')
  const [camId, setCamId] = useState(cams[0]?.id ?? '')
  const cam = cams.find((c) => c.id === camId)
  const [img, setImg] = useState<{ w: number; h: number } | null>(null)
  const [pts, setPts] = useState<{ img: number[]; floor: [string, string] }[]>([])
  const [res, setRes] = useState<CalibResult | null>(null)
  const [err, setErr] = useState('')
  const svg = useRef<SVGSVGElement>(null)
  useEffect(() => {
    const c = cam?.calibration
    setPts(c ? c.image_points.map((p, i) => ({ img: p, floor: [String(c.floor_points[i][0]), String(c.floor_points[i][1])] as [string, string] })) : [])
    setRes(null)
    setImg(null)
  }, [camId, cam?.calibration])
  if (!cam) return <Panel title="Calibration"><Empty>This store has no real cameras (simulator only).</Empty></Panel>
  const frameUrl = u(`/api/stores/${store.store_id}/cameras/${cam.id}/frame.jpg`)
  const run = async (save: boolean) => {
    setErr('')
    try {
      const body = { camera_id: cam.id, image_points: pts.map((p) => p.img), floor_points: pts.map((p) => p.floor.map(Number)), save }
      setRes(await api.putCalibration(store.store_id, body))
    } catch (e) { setErr(String(e)) }
  }
  return (
    <Panel title="Camera calibration · image pixels → floor metres (homography)"
      right={<div className="flex items-center gap-2 normal-case tracking-normal">
        <select className="select" value={camId} onChange={(e) => setCamId(e.target.value)}>{cams.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select>
        <button className="btn" onClick={() => setPts([])}>Clear points</button>
        <button className="btn" disabled={STATIC || pts.length < 4} onClick={() => run(false)}>Compute</button>
        <button className="btn btn-primary" disabled={STATIC || pts.length < 4} onClick={() => run(true)}>Save</button>
      </div>}>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="relative">
            <img src={frameUrl} alt="first frame (heads blurred)" className="w-full rounded" onLoad={(e) => setImg({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
            {img && (
              <svg ref={svg} viewBox={`0 0 ${img.w} ${img.h}`} className="absolute inset-0 h-full w-full cursor-crosshair"
                onClick={(e) => { const [x, y] = svgPoint(svg.current!, e); setPts([...pts, { img: [Math.round(x), Math.round(y)], floor: ['', ''] }]) }}>
                {pts.map((p, i) => (
                  <g key={i}><circle cx={p.img[0]} cy={p.img[1]} r={img.w / 120} fill="#ef4444" stroke="#fff" strokeWidth={img.w / 600} />
                    <text x={p.img[0] + img.w / 90} y={p.img[1] - img.w / 120} fontSize={img.w / 40} fill="#fff" stroke="#000" strokeWidth={img.w / 1200}>{i + 1}</text></g>
                ))}
              </svg>
            )}
          </div>
          <div className="mt-2 text-xs text-ink-400">Click ≥ 4 floor points spread across the view (tile corners, shelf feet), then enter their floor coordinates in metres.</div>
        </div>
        <div className="space-y-3">
          <table className="num w-full text-sm">
            <thead><tr className="text-left text-[10px] uppercase tracking-wider text-ink-400"><th className="py-1">#</th><th>Image px</th><th>Floor X (m)</th><th>Floor Y (m)</th><th /></tr></thead>
            <tbody>
              {pts.map((p, i) => (
                <tr key={i} className="border-t border-ink-800">
                  <td className="py-1 font-mono text-ink-400">{i + 1}</td>
                  <td className="font-mono text-xs text-ink-300">{p.img.join(', ')}</td>
                  {[0, 1].map((k) => (
                    <td key={k}><input className="input w-20" value={p.floor[k]} onChange={(e) => setPts(pts.map((q, j) => (j === i ? { ...q, floor: (k === 0 ? [e.target.value, q.floor[1]] : [q.floor[0], e.target.value]) as [string, string] } : q)))} /></td>
                  ))}
                  <td><button className="text-xs text-ink-400 hover:text-sev-critical" onClick={() => setPts(pts.filter((_, j) => j !== i))}>remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          {err && <div className="text-xs text-sev-critical">{err}</div>}
          {res && (
            <div className="space-y-2">
              <div className="num rounded-md border border-ink-700 bg-ink-850 p-3 font-mono text-xs text-ink-200">
                reprojection error: {res.reprojection_error.floor_m_mean.toFixed(3)} m mean · {res.reprojection_error.floor_m_max.toFixed(3)} m max ·{' '}
                {res.reprojection_error.image_px_mean.toFixed(1)} px mean {res.saved && <span className="text-signal">· saved</span>}
              </div>
              {res.preview && <><div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">Warped preview (camera → floor plan)</div><img src={res.preview} alt="warped preview" style={{ maxHeight: 360 }} className="mx-auto block rounded border border-ink-700 object-contain" /></>}
            </div>
          )}
          {cam.approximate && <div className="text-xs text-sev-medium">Current calibration is marked approximate (hand-estimated for the public test clip).</div>}
        </div>
      </div>
    </Panel>
  )
}

export default function Setup() {
  const storeId = useApp((s) => s.storeId)
  const store = useQuery({ queryKey: ['store', storeId], queryFn: () => api.store(storeId) })
  if (!store.data) return <Empty>Loading store…</Empty>
  return (
    <div className="flex flex-col gap-4">
      <ZoneEditor store={store.data} />
      <Calibration store={store.data} />
    </div>
  )
}
