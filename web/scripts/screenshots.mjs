// Headless screenshots of every console page -> docs/screenshots/*.png
// Usage: node scripts/screenshots.mjs [baseUrl] [runId]
import { chromium } from '@playwright/test'
import { mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const out = resolve(here, '../../docs/screenshots')
mkdirSync(out, { recursive: true })
const base = process.argv[2] ?? 'http://127.0.0.1:8000'
const pages = [
  ['01_live_ops', '/'], ['02_floor_plan', '/floor'], ['03_zones', '/zones'], ['04_journeys', '/journeys'],
  ['05_layout_advisor', '/layout'], ['06_setup', '/setup'], ['07_shift_briefing', '/briefing'],
]

const layoutRun = process.env.LAYOUT_RUN
const pick = async (page, run) => {
  await page.selectOption('select[aria-label="Run"]', run)
  await page.waitForTimeout(3000)
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 })
await page.goto(base + '/', { waitUntil: 'domcontentloaded' })
if (process.argv[3]) await pick(page, process.argv[3])
await page.waitForTimeout(4000)
for (const [name, path] of pages) {
  await page.click(`nav a[href="${path}"]`)
  if (path === '/layout' && layoutRun) await pick(page, layoutRun)
  if (path === '/setup' && process.env.SETUP_RUN) await pick(page, process.env.SETUP_RUN)
  if (path === '/briefing' && process.env.SETUP_RUN && process.argv[3]) await pick(page, process.argv[3])
  await page.waitForTimeout(path === '/' ? 4000 : 3500)
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: path !== '/' })
  console.log('saved', name)
}
// zone drawer
await page.click('nav a[href="/zones"]')
await page.waitForTimeout(2500)
await page.click('tbody tr >> nth=0')
await page.waitForTimeout(2500)
await page.screenshot({ path: `${out}/03b_zone_detail.png` })
console.log('saved 03b_zone_detail')
await browser.close()
