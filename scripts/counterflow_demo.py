"""Render an illustration of counter-flow conflict detection (docs/screenshots/counterflow_demo.png).

A synthetic two-way stream of walkers in "Main Walkway @ Aisle 3" is fed through the real AnalyticsHub + ActionEngine
and drawn with the production floor renderer; the raised insight is printed under the plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowsight.actions.engine import ActionEngine  # noqa: E402
from flowsight.analytics.hub import AnalyticsHub  # noqa: E402
from flowsight.config import load_config, load_store  # noqa: E402
from flowsight.render.annotate import Renderer  # noqa: E402
from flowsight.tracking.track_store import TrackState  # noqa: E402


def main():
    cfg, store = load_config(), load_store("demo_store")
    hub = AnalyticsHub(cfg, store)
    eng = ActionEngine(cfg, hub)
    x0, x1, n = 8.7, 11.9, 6
    rng = np.random.default_rng(0)
    for k in range(900):  # 90 s of a balanced two-way stream at ~1.4 m^2/person (LoS C/D)
        t = round(k * 0.1, 3)
        states = []
        for i in range(n):
            d = 1 if i % 2 == 0 else -1
            x = x0 + ((i * 0.55 + 0.7 * t * d) % (x1 - x0))
            y = 5.4 + 1.2 * (i / n) + rng.normal(0, 0.03)
            states.append(TrackState(id=i + 1, ts=t, x=x, y=y, speed=0.7, heading=0.0 if d > 0 else np.pi,
                                     vx=0.7 * d, vy=0.0, age=60, mature=True))
        hub.step(t, states)
        eng.evaluate(t)
    w = hub.density.latest["z_walk_mid"]
    img = Renderer(store, cfg).draw_floor(hub, hub.ts)
    for s in hub.people():  # direction arrows
        x, y = int(s["x"] * 50), int(s["y"] * 50)
    for st in hub.tracks:
        x, y = int(st.x * 50), int(st.y * 50)
        cv2.arrowedLine(img, (x, y), (x + int(np.sign(st.vx) * 22), y), (0, 0, 255) if st.vx < 0 else (255, 200, 0), 2,
                        tipLength=0.4)
    ins = [i for i in eng.active_sorted() if i.type == "counter_flow"]
    pad = np.full((110, img.shape[1], 3), (21, 15, 11), np.uint8)
    lines = [f"Main Walkway @ Aisle 3: counter-flow {w.counterflow:.0%}, direction entropy {w.direction_entropy:.2f}, "
             f"LoS {w.los}, {len(hub.density.conflicts)} conflict episode(s)"]
    if ins:
        lines.append(f"Insight [{ins[0].severity} / {ins[0].horizon}]: " + ins[0].action[:95])
        if len(ins[0].action) > 95:
            lines.append("    " + ins[0].action[95:])
    for j, ln in enumerate(lines):
        cv2.putText(pad, ln, (12, 30 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    out = ROOT / "docs" / "screenshots" / "counterflow_demo.png"
    cv2.imwrite(str(out), np.vstack([img, pad]))
    print("wrote", out, "|", lines)


if __name__ == "__main__":
    main()
