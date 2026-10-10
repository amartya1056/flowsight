// Named README screenshots of the running console -> docs/screenshots/<name>.png
// Usage: node scripts/readme_shots.mjs <shot> [<shot> ...]   (base URL from $BASE, default http://127.0.0.1:8000)
import { chromium } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const out = resolve(here, '../../docs/screenshots')
mkdirSync(out, { recursive: true })
const base = process.env.BASE ?? 'http://127.0.0.1:8000'
const runs = await (await fetch(`${base}/api/runs`)).json()
const run = (needle) => {
  const r = runs.find((x) => x.run_id.includes(needle))
  if (!r) throw new Error(`no run matching ${needle}`)
  return r.run_id
}

// shot -> [run substring, nav path, extra actions(page), fullPage]
const SHOTS = {
  live_ops_rush: ['demo-rush', '/', null, false],
  live_ops_camera: ['demo-caviar', '/', null, false],
  live_ops_multicam: ['multi', '/', null, false],
  live_ops_staff: ['with_staff', '/', null, false],
  live_ops_blocked: ['blocked_aisle', '/', null, false],
  floor_occupancy: ['demo-rush', '/floor', null, true],
  floor_stationary: ['demo-rush', '/floor', async (p) => { await p.getByText('Stationary heat').click() }, true],
  floor_hover: ['demo-rush', '/floor', async (p) => {
    const c = await p.locator('canvas').first().boundingBox()
    await p.mouse.move(c.x + c.width * 0.53, c.y + c.height * 0.5)
  }, false],
  floor_camera_store: ['demo-caviar', '/floor', null, true],
  zones_table: ['demo-rush', '/zones', null, true],
  zone_drawer: ['demo-rush', '/zones', async (p) => { await p.getByText('Main Walkway East').first().click() }, false],
  zone_drawer_promo: ['ignored_promo', '/zones', async (p) => { await p.getByText('Promo End-cap').first().click() }, false],
  journeys: ['demo-rush', '/journeys', null, true],
  layout_whatif: ['ignored_promo', '/layout', null, true],
  layout_bottlenecks: ['demo-rush', '/layout', null, true],
  layout_blocked: ['blocked_aisle', '/layout', null, true],
  briefing: ['demo-rush', '/briefing', null, true],
  setup_zones: ['demo-rush', '/setup', async (p) => { await p.getByText('Promo End-cap').last().click() }, false],
  setup_calibration: ['demo-caviar', '/setup', async (p) => {
    await p.waitForTimeout(2000)
    await p.getByRole('button', { name: 'Compute' }).click()
    await p.getByText('reprojection error').waitFor({ timeout: 60000 })
    await p.evaluate(() => {
      const h = [...document.querySelectorAll('.panel-h')].find((e) => e.textContent?.includes('Camera calibration'))
      h?.scrollIntoView({ block: 'start' })
    })
  }, false],
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 })
for (const name of process.argv.slice(2)) {
  const [needle, path, act, full] = SHOTS[name]
  await page.goto(base + '/', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(1500)
  await page.selectOption('select[aria-label="Run"]', run(needle))
  await page.waitForTimeout(1500)
  await page.click(`nav a[href="${path}"]`)
  await page.waitForTimeout(3500)
  if (act) { await act(page); await page.waitForTimeout(2500) }
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: full })
  console.log('saved', name)
}
// standalone printable report
const rr = run(process.env.REPORT_RUN ?? 'demo-rush')
if (process.argv.includes('live_ops_rush')) {
  await page.goto(`${base}/api/runs/${rr}/report.html`)
  await page.waitForTimeout(2000)
  await page.screenshot({ path: `${out}/report_html.png`, fullPage: true })
  console.log('saved report_html')
}
await browser.close()
