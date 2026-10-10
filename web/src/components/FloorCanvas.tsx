import { useEffect, useMemo, useRef, useState } from 'react'
import { u } from '../static'
import { LOS_COLOR, fmtDur, pct, type HeatGrid, type LiveZone, type Person, type StoreDetail, type ZoneKpi } from '../api'

interface Props {
  store: StoreDetail
  people?: Person[]
  live?: Record<string, LiveZone>
  heat?: HeatGrid | null
  kpis?: Record<string, ZoneKpi>
  highlight?: string | null
  showZones?: boolean
  className?: string
}

// perceptual "turbo-ish" ramp for heat cells
const RAMP = ['#30123b', '#4662d7', '#36aaf9', '#1ae4b6', '#72fe5e', '#c8ef34', '#faba39', '#f66b19', '#ca2a04']
function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1)
  const i = Math.floor(x), f = x - i
  const a = RAMP[i], b = RAMP[Math.min(i + 1, RAMP.length - 1)]
  const p = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16)
  return [0, 1, 2].map((k) => Math.round(p(a, k) * (1 - f) + p(b, k) * f)) as [number, number, number]
}

export function pointInPoly(x: number, y: number, poly: number[][]) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) inside = !inside
  }
  return inside
}

export default function FloorCanvas({ store, people = [], live = {}, heat, kpis, highlight, showZones = true, className }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  const [width, setWidth] = useState(800)
  const [hover, setHover] = useState<{ zone: string; x: number; y: number } | null>(null)
  const W = store.floorplan.width_m, H = store.floorplan.height_m
  const height = (width * H) / W
  const s = width / W

  useEffect(() => {
    const im = new Image()
    im.onload = () => setImg(im)
    im.src = u(store.floorplan_url)
  }, [store.floorplan_url])

  useEffect(() => {
    if (!wrap.current) return
    const ro = new ResizeObserver((e) => setWidth(Math.max(320, e[0].contentRect.width)))
    ro.observe(wrap.current)
    return () => ro.disconnect()
  }, [])

  // blurred heat layer rendered once per grid
  const heatCanvas = useMemo(() => {
    if (!heat || !heat.grid.length) return null
    const c = document.createElement('canvas')
    c.width = heat.nx
    c.height = heat.ny
    const ctx = c.getContext('2d')!
    const im = ctx.createImageData(heat.nx, heat.ny)
    const vals = heat.grid.flat().filter((v) => v > 0).sort((a, b) => a - b)
    const vmax = vals.length ? vals[Math.floor(vals.length * 0.98)] || vals[vals.length - 1] : 1
    for (let y = 0; y < heat.ny; y++)
      for (let x = 0; x < heat.nx; x++) {
        const v = heat.grid[y][x] / vmax
        const k = (y * heat.nx + x) * 4
        const [r, g, b] = ramp(Math.sqrt(Math.min(1, v)))
        im.data[k] = r; im.data[k + 1] = g; im.data[k + 2] = b
        im.data[k + 3] = v > 0.01 ? Math.min(220, 60 + 200 * Math.sqrt(v)) : 0
      }
    ctx.putImageData(im, 0, 0)
    return c
  }, [heat])

  useEffect(() => {
    let raf = 0
    const draw = (t: number) => {
      const cv = canvas.current
      if (!cv) return
      const dpr = window.devicePixelRatio || 1
      cv.width = width * dpr
      cv.height = height * dpr
      const ctx = cv.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.fillStyle = '#0b0f15'
      ctx.fillRect(0, 0, width, height)
      if (img) ctx.drawImage(img, 0, 0, width, height)
      if (heatCanvas) {
        ctx.save()
        ctx.filter = `blur(${Math.max(2, s * 0.35)}px)`
        ctx.imageSmoothingEnabled = true
        ctx.drawImage(heatCanvas, 0, 0, heatCanvas.width * (heat!.cell_m * s), heatCanvas.height * (heat!.cell_m * s))
        ctx.restore()
      }
      if (showZones)
        for (const z of store.zones) {
          const lz = live[z.id]
          const col = LOS_COLOR[lz?.los ?? 'A']
          ctx.beginPath()
          z.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)))
          ctx.closePath()
          ctx.fillStyle = col + (heat ? '10' : '26')
          ctx.fill()
          const pulse = lz?.bottleneck ? 0.5 + 0.5 * Math.sin(t / 160) : 0
          ctx.lineWidth = lz?.bottleneck ? 2 + 3 * pulse : z.id === highlight ? 3 : 1
          ctx.strokeStyle = lz?.bottleneck ? `rgba(239,68,68,${0.5 + 0.5 * pulse})` : z.id === highlight ? '#2dd4bf' : col + 'aa'
          ctx.stroke()
          const xs = z.polygon.map((p) => p[0]), ys = z.polygon.map((p) => p[1])
          ctx.font = '600 10px "IBM Plex Sans", system-ui'
          ctx.fillStyle = 'rgba(227,232,239,0.85)'
          ctx.fillText(z.name, Math.min(...xs) * s + 4, Math.min(...ys) * s + 12)
          if (lz && lz.occupancy > 0) {
            ctx.fillStyle = col
            ctx.fillText(`${lz.occupancy}`, Math.max(...xs) * s - 14, Math.min(...ys) * s + 12)
          }
        }
      for (const p of people) {
        const still = p.speed < 0.3
        ctx.beginPath()
        ctx.arc(p.x * s, p.y * s, Math.max(3.5, s * 0.16), 0, Math.PI * 2)
        ctx.fillStyle = still ? '#fbbf24' : '#5eead4'
        ctx.fill()
        ctx.lineWidth = 1.2
        ctx.strokeStyle = '#07090d'
        ctx.stroke()
      }
      if (store.zones.some((z) => live[z.id]?.bottleneck)) raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [img, width, height, s, store.zones, live, people, heatCanvas, heat, highlight, showZones])

  const onMove = (e: React.MouseEvent) => {
    const r = canvas.current!.getBoundingClientRect()
    const x = (e.clientX - r.left) / s, y = (e.clientY - r.top) / s
    const z = [...store.zones].reverse().find((z) => pointInPoly(x, y, z.polygon))
    setHover(z ? { zone: z.id, x: e.clientX - r.left, y: e.clientY - r.top } : null)
  }
  const hz = hover ? store.zones.find((z) => z.id === hover.zone) : null
  const hk = hover && kpis ? kpis[hover.zone] : null
  const hl = hover ? live[hover.zone] : null

  return (
    <div ref={wrap} className={`relative ${className ?? ''}`}>
      <canvas ref={canvas} style={{ width, height }} className="block rounded-md" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      {hover && hz && (
        <div className="pointer-events-none absolute z-10 w-56 rounded-md border border-ink-600 bg-ink-850/95 p-3 text-xs shadow-xl"
          style={{ left: Math.min(hover.x + 14, width - 230), top: Math.min(hover.y + 14, height - 150) }}>
          <div className="mb-1 flex items-center justify-between">
            <span className="font-semibold text-ink-100">{hz.name}</span>
            <span className="chip bg-ink-700 text-ink-300">{hz.type}</span>
          </div>
          <div className="num grid grid-cols-2 gap-x-3 gap-y-0.5 text-ink-300">
            <span>Now</span><span className="text-right text-ink-100">{hl?.occupancy ?? 0} people</span>
            <span>LoS</span><span className="text-right" style={{ color: LOS_COLOR[hl?.los ?? 'A'] }}>{hl?.los ?? 'A'}</span>
            {hk && <>
              <span>Visits</span><span className="text-right text-ink-100">{hk.visits}</span>
              <span>Median dwell</span><span className="text-right text-ink-100">{fmtDur(hk.median_dwell_s)}</span>
              <span>Stopping power</span><span className="text-right text-ink-100">{pct(hk.stopping_power)}</span>
              <span>Status</span><span className="text-right text-ink-100">{hk.bottleneck_active ? 'BOTTLENECK' : hk.attention_status}</span>
            </>}
          </div>
        </div>
      )}
    </div>
  )
}
