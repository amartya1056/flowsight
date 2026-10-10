"""Record FlowSight runs into static files for the GitHub Pages replay demo.

    python scripts/export_static_demo.py            # -> site/demo-data/

For each demo run we execute the real pipeline (simulator or YOLO on the CAVIAR clips), record a compact live timeline
(people, zone states, KPIs, insight list whenever it changes) and dump every API response the console reads, using the
API's own endpoint functions, so the replay shows exactly what the live app shows.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

import api.main as A  # noqa: E402
from flowsight.config import load_config, load_store, resolve_path  # noqa: E402
from flowsight.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "site" / "demo-data"
DEMO_RUNS = [  # run_id, store, source, duration (s), record every (s), seed
    ("demo-rush-hour", "demo_store", "sim:rush_hour", 1500, 2.0, 0),
    ("demo-ignored-promo", "demo_store", "sim:ignored_promo", 1800, 3.0, 2),
    ("demo-with-staff", "demo_store", "sim:with_staff", 2700, 4.0, 0),
    ("caviar-camera", "caviar_corridor", "cam1", None, 0.5, 0),
    ("caviar-multicam", "caviar_multicam", "multi", None, 0.5, 0),
]
HEAT_WINDOWS = [None, 300, 900, 1800, 3600]


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(A._jsonable(obj), separators=(",", ":"), default=str), encoding="utf-8")


def record(run_id, store_id, source, duration, every, seed) -> dict:
    cfg = load_config()
    st = load_store(store_id)
    run_dir = ROOT / "runs" / f"static-{run_id}"
    shutil.rmtree(run_dir, ignore_errors=True)
    p = Pipeline(cfg, st, source, run_dir, speed="max", duration_s=duration, live=True, seed=seed)
    frames, last = [], {"t": -1e9, "v": -1}
    orig = p._step_analytics

    def rec(ts, states):
        orig(ts, states)
        if ts - last["t"] < every - 1e-9:
            return
        last["t"] = ts
        h = p.hub
        with p.state_lock:
            f = {"t": round(ts, 2),
                 "p": [[q["id"], round(q["x"], 2), round(q["y"], 2), round(q["speed"], 2), q["stationary_s"]]
                       for q in h.people()],
                 "z": [[z["id"], z["occupancy"], z["los"], z["dwell_live"], int(z["bottleneck"]), z["status"]]
                       for z in h.live_zones()],
                 "k": A._jsonable({k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.kpis().items()
                                   if k != "fusion"})}
            if p.engine.version != last["v"]:
                last["v"] = p.engine.version
                f["i"] = [i.model_dump() for i in p.engine.active_sorted()]
        frames.append(f)

    p._step_analytics = rec
    s = p.run()
    if s.get("error"):
        raise RuntimeError(f"{run_id}: {s['error']}")
    print(f"  {run_id}: {len(frames)} frames, ts={p.hub.ts:.0f}s, insights={len(p.engine.all)}")

    # register under a friendly id and dump every endpoint the console reads
    h = A.RunHandle(p, store_id, source, "max")
    p.run_id = run_id
    A.RUNS[run_id] = h
    d = OUT / "runs" / run_id
    dump(d / "timeline.json", {"step": every, "video": not p.is_sim, "duration": p.hub.ts or 0, "frames": frames})
    info = h.info()
    info["finished"] = False  # replays loop forever
    dump(d / "info.json", info)
    for name, fn in [("kpis", A.kpis), ("zones", A.zones), ("flow", A.flow), ("bottlenecks", A.bottlenecks),
                     ("attention", A.attention), ("queue", A.queue), ("layout", A.layout), ("briefing", A.briefing)]:
        dump(d / f"{name}.json", fn(run_id))
    dump(d / "insights_all.json", A.insights(run_id, "all"))
    # the replay's insight list comes from the timeline; "active" = what was active at the end of the run
    dump(d / "insights_active.json", A.insights(run_id, "active"))
    for typ in ("occupancy", "stationary"):
        for w in HEAT_WINDOWS:
            dump(d / f"heatmap_{typ}_{w or 'all'}.json", A.heatmap(run_id, typ, w, "json"))
        (d / f"heatmap_{typ}.png").write_bytes(A.heatmap(run_id, typ, None, "png").body)
    (d / "report.html").write_text(A.report_html(run_id), encoding="utf-8")
    (d / "export.csv").write_bytes(A.export_csv(run_id).body)
    if p.latest_jpeg:
        (d / "frame.jpg").write_bytes(p.latest_jpeg)
    if (run_dir / "evidence").exists():
        shutil.copytree(run_dir / "evidence", d / "evidence", dirs_exist_ok=True)
    if not p.is_sim and (run_dir / "annotated.mp4").exists():
        shutil.copy(run_dir / "annotated.mp4", d / "video.mp4")
    p.db.close()
    return info


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    infos = []
    for spec in DEMO_RUNS:
        infos.append(record(*spec))
    dump(OUT / "runs.json", infos)
    stores = A.stores()
    dump(OUT / "stores.json", stores)
    for s in stores:
        sid = s["store_id"]
        dump(OUT / "stores" / f"{sid}.json", A.store_detail(sid))
        st = load_store(sid)
        shutil.copy(resolve_path(st.floorplan.image), OUT / "stores" / sid / "floorplan.png") \
            if (OUT / "stores" / sid).mkdir(parents=True, exist_ok=True) is None else None
        for c in st.cameras:
            if c.source != "sim" and resolve_path(c.source).exists():
                (OUT / "stores" / sid / f"{c.id}_frame.jpg").write_bytes(A.camera_frame(sid, c.id).body)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
