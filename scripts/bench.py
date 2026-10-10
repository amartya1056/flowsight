"""Benchmarks on the current device -> docs/benchmarks.md

1. detector+tracker FPS for yolo11n / yolo11s at imgsz 640 / 960 on a real clip
2. analytics-only throughput vs people per frame (synthetic tracks through the full hub + action engine)
3. cameras-per-edge-box estimate at the configured analytics rate
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowsight.config import load_config, load_store, resolve_path  # noqa: E402


def bench_detector(cfg, video: Path, frames: int) -> list[dict]:
    import cv2

    from flowsight.perception.detector_tracker import DetectorTracker

    cap = cv2.VideoCapture(str(video))
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    out = []
    for model in ("yolo11n.pt", "yolo11s.pt"):
        for imgsz in (640, 960):
            det = DetectorTracker(cfg.runtime, model_name=model, imgsz=imgsz)
            for f in imgs[:10]:
                det.update(f)  # warm-up
            det = DetectorTracker(cfg.runtime, model_name=model, imgsz=imgsz)
            n_people = []
            t0 = time.perf_counter()
            for f in imgs:
                n_people.append(len(det.update(f)))
            dt = time.perf_counter() - t0
            out.append({"model": model, "imgsz": imgsz, "device": det.device, "half": det.half,
                        "fps": len(imgs) / dt, "ms": 1000 * dt / len(imgs), "people": float(np.mean(n_people))})
            print(out[-1], flush=True)
    return out


def bench_analytics(cfg, people_counts=(5, 20, 50, 100, 200), seconds: float = 60.0) -> list[dict]:
    from flowsight.actions.engine import ActionEngine
    from flowsight.analytics.hub import AnalyticsHub
    from flowsight.tracking.track_store import TrackStore

    store = load_store("demo_store")
    W, H = store.floorplan.width_m, store.floorplan.height_m
    out = []
    for n in people_counts:
        rng = np.random.default_rng(n)
        hub = AnalyticsHub(cfg, store)
        eng = ActionEngine(cfg, hub)
        ts = TrackStore(cfg.tracking)
        pos = rng.uniform([0, 0], [W, H], (n, 2))
        vel = rng.normal(0, 0.6, (n, 2))
        steps = int(seconds * cfg.runtime.target_fps)
        dt = 1 / cfg.runtime.target_fps
        t0 = time.perf_counter()
        for k in range(steps):
            pos = np.clip(pos + vel * dt, 0, [W, H])
            t = (k + 1) * dt
            states = ts.update(t, [(i, pos[i, 0] + rng.normal(0, .1), pos[i, 1] + rng.normal(0, .1)) for i in range(n)])
            hub.step(t, states, ts.pop_expired())
            eng.evaluate(t)
        el = time.perf_counter() - t0
        out.append({"people": n, "frames_per_s": steps / el, "ms_per_frame": 1000 * el / steps})
        print(out[-1], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--video", default="data/videos/caviar_corridor.mp4")
    ap.add_argument("--onnx", action="store_true", help="only run the CPU PyTorch-vs-ONNX section")
    a = ap.parse_args()
    cfg = load_config()
    video = resolve_path(a.video)
    det = bench_detector(cfg, video, 60 if a.quick else 300) if video.exists() else []
    dense = resolve_path("data/videos/mot20_01.mp4")
    det_dense = bench_detector(cfg, dense, 60 if a.quick else 200) if dense.exists() and not a.quick else []
    ana = bench_analytics(cfg, (5, 50) if a.quick else (5, 20, 50, 100, 200), 20 if a.quick else 60)

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    fps_t = cfg.runtime.target_fps
    L = ["# FlowSight benchmarks", "",
         f"Host: {platform.processor() or platform.machine()} · {platform.system()} {platform.release()} · GPU: {gpu} · "
         f"torch {torch.__version__}. Analytics target rate: {fps_t:.0f} FPS per camera.", ""]
    for title, rows in (("Detector + ByteTrack — CAVIAR corridor (384×288, sparse)", det),
                        ("Detector + ByteTrack — MOT20-01 (1920×1080, dense crowd)", det_dense)):
        if not rows:
            continue
        L += [f"## {title}", "", "| Model | imgsz | Device | FP16 | FPS | ms/frame | people/frame |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r['model']} | {r['imgsz']} | {r['device']} | {r['half']} | {r['fps']:.1f} | {r['ms']:.1f} | {r['people']:.1f} |" for r in rows]
        L.append("")
    L += ["## Analytics only (TrackStore + all analytics + action engine, no detector)", "",
          "| People in frame | Frames/s | ms/frame | Cameras per core @10 FPS |", "|---|---|---|---|"]
    L += [f"| {r['people']} | {r['frames_per_s']:.0f} | {r['ms_per_frame']:.2f} | {r['frames_per_s'] / fps_t:.0f} |" for r in ana]
    if det:
        best = max((r for r in det if r["model"] == cfg.runtime.model_gpu), key=lambda r: r["imgsz"], default=det[0])
        a20 = next((r for r in ana if r["people"] >= 20), ana[-1])
        cams_det = best["fps"] / fps_t
        cams_ana = a20["frames_per_s"] / fps_t
        L += ["", "## Cameras per edge box (estimate)", "",
              f"* Detector ({best['model']} @ {best['imgsz']}, {best['device']}): {best['fps']:.0f} FPS → **~{cams_det:.0f} cameras** "
              f"at {fps_t:.0f} FPS each (sequential, single stream; batching multiple cameras raises this further).",
              f"* Analytics (~20 people/frame): {a20['frames_per_s']:.0f} FPS on one CPU core → ~{cams_ana:.0f} cameras per core.",
              f"* The detector is the bottleneck: one edge box with this GPU ≈ **{int(min(cams_det, cams_ana * 4))} cameras** at "
              f"{fps_t:.0f} FPS analytics. Running at 5 FPS (enough for dwell/flow) doubles that.",
              "", "Analytics cost is independent of image resolution — it only sees floor-metre points."]
    out = ROOT / "docs" / "benchmarks.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__" and "--onnx" not in sys.argv:
    main()


def bench_cpu_onnx(video: Path, frames: int = 120) -> list[dict]:
    """PyTorch-CPU vs ONNX-Runtime-CPU at 640 (the edge-box-without-GPU case), with a detection-parity check."""
    import cv2

    from flowsight.config import load_config
    from flowsight.perception.detector_tracker import DetectorTracker

    cfg = load_config().with_overrides(runtime={"device": "cpu"})
    cap = cv2.VideoCapture(str(video))
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    rows, ref = [], {}
    for stem in ("yolo11n", "yolo11s"):
        for fmt, name in (("pytorch", f"{stem}.pt"), ("onnx", f"{stem}_640.onnx")):
            det = DetectorTracker(cfg.runtime, model_name=name, imgsz=640)
            for f in imgs[:5]:
                det.detect(f)
            counts = []
            t0 = time.perf_counter()
            for f in imgs:
                counts.append(len(det.detect(f)))
            dt = time.perf_counter() - t0
            if fmt == "pytorch":
                ref[stem] = counts
            agree = float(np.mean(np.array(counts) == np.array(ref[stem]))) if stem in ref else 1.0
            rows.append({"model": stem, "format": fmt, "fps": len(imgs) / dt, "ms": 1000 * dt / len(imgs),
                         "people": float(np.mean(counts)), "count_agreement": agree})
            print(rows[-1], flush=True)
    return rows


def write_cpu_section(rows: list[dict]) -> None:
    out = ROOT / "docs" / "benchmarks.md"
    text = out.read_text(encoding="utf-8") if out.exists() else "# FlowSight benchmarks\n"
    start, end = "<!-- cpu-onnx:start -->", "<!-- cpu-onnx:end -->"
    if start in text:
        text = text[:text.index(start)] + text[text.index(end) + len(end):]
    sec = [start, "## CPU-only edge box: PyTorch vs ONNX Runtime (detector only, imgsz 640, CAVIAR)", "",
           "| Model | Runtime | FPS | ms/frame | people/frame | same #detections as PyTorch |", "|---|---|---|---|---|---|"]
    sec += [f"| {r['model']} | {r['format']} | {r['fps']:.1f} | {r['ms']:.1f} | {r['people']:.2f} | "
            f"{r['count_agreement']:.0%} |" for r in rows]
    by = {(r["model"], r["format"]): r for r in rows}
    sp = [f"{m}: ×{by[(m, 'onnx')]['fps'] / by[(m, 'pytorch')]['fps']:.2f}" for m in ("yolo11n", "yolo11s")
          if (m, "onnx") in by and (m, "pytorch") in by]
    sec += ["", f"ONNX Runtime speed-up over PyTorch on CPU — {', '.join(sp)}. At 5 FPS analytics (enough for dwell "
            f"and flow) this CPU (all cores) with yolo11n ONNX serves ~{by[('yolo11n', 'onnx')]['fps'] / 5:.0f} cameras' "
            "worth of detection. Export: `python scripts/export_onnx.py`; use via `runtime.model_cpu: yolo11n_640.onnx`.",
            end, ""]
    out.write_text(text.rstrip() + "\n\n" + "\n".join(sec), encoding="utf-8")
    print("wrote CPU/ONNX section to", out)


if __name__ == "__main__" and "--onnx" in sys.argv:
    write_cpu_section(bench_cpu_onnx(ROOT / "data" / "videos" / "caviar_corridor.mp4"))
