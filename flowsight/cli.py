"""FlowSight CLI: process | sim | eval | bench | calibrate | device."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from flowsight.config import ROOT, load_config, load_store, save_store

app = typer.Typer(add_completion=False, help="FlowSight — retail dwell-time & bottleneck intelligence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _print_summary(s: dict) -> None:
    k = s["kpis"]
    typer.echo(json.dumps({
        "run_id": s["run_id"], "device": s["device"], "model": s["model"], "calibrated": s["calibrated"],
        "proc_fps": s["proc_fps"], "wall_s": s["wall_s"], "visitors": k["visitors_total"],
        "peak_people": k["peak_people"], "visits": k["visits_total"], "bottlenecks": k["bottlenecks_total"],
        "id_stitches": s["id_stitches"], "tracks_created": s["tracks_created"],
        "insights_total": s["insights_total"], "error": s["error"], "db_counts": s.get("db_counts")}, indent=2))


@app.command()
def device():
    """Print the compute device that would be used."""
    from flowsight.perception.detector_tracker import select_device

    d, half = select_device(load_config().runtime.device)
    typer.echo(f"device={d} half={half}")


@app.command()
def process(store: str = typer.Option("demo_store"), camera: str = typer.Option("cam1"),
            out: Optional[Path] = typer.Option(None), max_seconds: Optional[float] = None,
            model: Optional[str] = None, no_video: bool = False):
    """Run the full pipeline on a camera source (file/RTSP/webcam) and write outputs to runs/<id>."""
    from flowsight.pipeline import Pipeline, default_run_dir, new_run_id

    cfg, st = load_config(), load_store(store)
    run_dir = out or default_run_dir(new_run_id(f"{store}-{camera}"))
    p = Pipeline(cfg, st, camera, run_dir, speed="max", duration_s=max_seconds, write_video=not no_video,
                 model_name=model)
    s = p.run()
    _print_summary(s)
    typer.echo(f"outputs in {run_dir}")


@app.command()
def sim(scenario: str = typer.Option("baseline"), store: str = "demo_store", seed: int = 0,
        duration: float = 900.0, out: Optional[Path] = None):
    """Run a simulator scenario through the analytics + action engine."""
    from flowsight.pipeline import Pipeline, default_run_dir, new_run_id

    cfg, st = load_config(), load_store(store)
    run_dir = out or default_run_dir(new_run_id(f"sim-{scenario}"))
    p = Pipeline(cfg, st, f"sim:{scenario}", run_dir, speed="max", duration_s=duration, seed=seed)
    s = p.run()
    _print_summary(s)
    for ins in p.engine.active_sorted():
        typer.echo(f"  [{ins.severity:8s}|{ins.horizon:9s}] {ins.action}")
    typer.echo(f"outputs in {run_dir}")


@app.command("eval")
def eval_cmd(seeds: int = 5, duration: float = 1800.0, workers: int = 4,
             scenarios: str = "baseline,rush_hour,blocked_aisle,ignored_promo,dead_aisle"):
    """Accuracy of analytics vs simulator ground truth (writes runs/eval/report.md + CSV)."""
    from flowsight.eval.evaluate import run_eval

    run_eval([s.strip() for s in scenarios.split(",") if s.strip()], seeds=seeds, duration=duration, workers=workers)


@app.command()
def bench(quick: bool = False):
    """Benchmark detector+tracker and analytics throughput on this device."""
    import runpy
    import sys

    sys.argv = ["bench.py"] + (["--quick"] if quick else [])
    runpy.run_path(str(ROOT / "scripts" / "bench.py"), run_name="__main__")


@app.command()
def calibrate(store: str = typer.Option("demo_store"), camera: str = typer.Option("cam1")):
    """Click >= 4 image points on the first frame, type their floor coordinates (m); saves to the store yaml."""
    import cv2

    from flowsight.config import Calibration
    from flowsight.geometry.calibration import Homography
    from flowsight.io.video_source import VideoSource

    st = load_store(store)
    cam = st.camera(camera)
    frame = VideoSource(cam.source).read_first()
    pts: list[list[float]] = []
    win = "FlowSight calibrate: click points, press ENTER when done (ESC to abort)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([float(x), float(y)])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        cv2.imshow(win, frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 13:
            break
        if k == 27:
            cv2.destroyAllWindows()
            raise typer.Exit(1)
    cv2.destroyAllWindows()
    if len(pts) < 4:
        typer.echo("need at least 4 points")
        raise typer.Exit(1)
    floor = []
    for i, p in enumerate(pts, 1):
        s = typer.prompt(f"floor X,Y (metres) for point {i} at image {p}")
        floor.append([float(v) for v in s.split(",")])
    H = Homography(pts, floor)
    typer.echo(f"reprojection error: {H.reprojection_error()}")
    cam.calibration = Calibration(image_points=pts, floor_points=floor)
    cam.approximate = False
    typer.echo(f"saved to {save_store(st)}")


if __name__ == "__main__":
    app()
