# FlowSight: 2-minute demo script

**Setup (before recording):** `make demo` (or `python scripts/tasks.py demo`) → open http://127.0.0.1:8000.
It auto-starts the `rush_hour` simulator at 4× and the real CAVIAR corridor clip. Let both run for about 2 minutes
before you record, so the action feed has content.

| Time | Click / show | Say |
|---|---|---|
| 0:00 | **Live Ops**, CAVIAR run selected in the top bar | "Ordinary in-store CCTV. FlowSight detects and tracks shoppers, blurs every head on every frame that leaves the box, and projects each person's feet onto the floor plan in **metres**." |
| 0:15 | Point at the zone overlays and the dwell labels on the boxes | "Zones are polygons on the store's floor plan, not pixels. Every number you'll see is physical: seconds, metres, people per square metre." |
| 0:25 | Switch the run selector to the **demo-rush** simulator run | "To prove the numbers are right we built a shopper simulator with ground truth. Here, a rush hour is starting." |
| 0:35 | Right panel: **Action feed** | "This isn't a dashboard, it's a to-do list. *Open another checkout lane, predicted wait 4 minutes, 2.6× normal.* *Clear the main walkway: Level of Service E, 11 shoppers slowed, 96 seconds lost.* Each card has an owner, a horizon, the evidence and the expected impact. Ack it and it leaves the queue; when the condition clears it resolves on its own." |
| 0:55 | **Floor Plan** → toggle *Stationary heat* | "Live positions on the plan. Occupancy heat shows where people are; stationary heat shows where they actually stop. Bottleneck zones pulse red." |
| 1:05 | **Zones** → click *Promo End-cap* | "Our key insight is the attention funnel: exposures → entries → stops. A zone that people walk past but don't stop at is **IGNORED**: that's a merchandising problem. A zone nobody even walks past is **UNVISITED**: that's a layout problem. The fixes are different, so we tell them apart." |
| 1:20 | **Layout Advisor** | "The what-if engine estimates that moving the promo next to Aisle 3, our highest-stopping-power zone, adds about this many engaged visits a day. It's shown as a range and labelled as an estimate. Recurring bottlenecks are listed for the weekly layout review." |
| 1:35 | **Journeys** | "Customer journeys as a Sankey, the top paths, and entrance-to-checkout conversion." |
| 1:45 | **Shift Briefing** → *Download daily report* | "Every shift gets a five-bullet briefing. With an API key an LLM writes it, constrained to our numbers. Without one, a deterministic template writes it. It works fully offline." |
| 1:52 | Show `runs/eval/report.md` (or the README table) | "Validated: visit F1 and dwell error against ground truth across 5 scenarios × 5 seeds, with tracker noise and ID breaks injected. It runs on existing cameras, on the edge, and only anonymous aggregates go to the cloud." |
| 2:00 | end | "FlowSight: from CCTV to actions." |

**Backup plan:** if the real clip can't be processed (no GPU or weights), the simulator runs carry the whole demo:
start `sim:blocked_aisle` and `sim:ignored_promo` from the top bar at 4×.
