"""FlowSight API — FastAPI app. A background thread runs each pipeline; endpoints read its live state."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flowsight.actions.briefing import generate_briefing
from flowsight.actions.engine import fmt_dur, what_if
from flowsight.analytics.heatmap import heatmap_on_floorplan
from flowsight.config import (ROOT, Calibration, ZoneCfg, list_stores, load_config, load_store, resolve_path,
                              save_store)
from flowsight.geometry.calibration import Homography
from flowsight.pipeline import Pipeline, new_run_id
from flowsight.render.annotate import make_floorplan_image

log = logging.getLogger("flowsight.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CFG = load_config()
app = FastAPI(title="FlowSight API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CFG.api.cors_origins or ["*"], allow_methods=["*"],
                   allow_headers=["*"])


class RunHandle:
    def __init__(self, pipeline: Pipeline, store_id: str, source: str, speed):
        self.p = pipeline
        self.store_id, self.source, self.speed = store_id, source, speed
        self.stop = threading.Event()
        self.created = time.time()
        self.thread = threading.Thread(target=self.p.run, args=(self.stop,), daemon=True, name=f"run-{pipeline.run_id}")

    def info(self) -> dict:
        p = self.p
        return {"run_id": p.run_id, "store_id": self.store_id, "source": self.source, "speed": str(self.speed),
                "created": self.created, "finished": p.finished, "progress": round(p.progress, 4), "error": p.error,
                "device": p.device, "model": p.model, "calibrated": p.calibrated, "ts": p.hub.ts or 0.0,
                "is_sim": p.is_sim, "proc_fps": round(p.proc_fps, 1), "frame_size": p.frame_size}


RUNS: dict[str, RunHandle] = {}
_FRAME_CACHE: dict[tuple[str, str], bytes] = {}


def _run(run_id: str) -> RunHandle:
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, f"run {run_id} not found")
    return h


def _store_or_404(store_id: str):
    try:
        return load_store(store_id)
    except FileNotFoundError:
        raise HTTPException(404, f"store {store_id} not found")


def start_run(store_id: str, source: str, speed="1", duration_s: float | None = None, seed: int = 0,
              run_id: str | None = None) -> RunHandle:
    store = _store_or_404(store_id)
    if not source.startswith("sim"):
        cam = store.camera(source)
        if not cam.source.startswith(("rtsp://", "http://", "https://")) and not cam.source.isdigit() \
                and not resolve_path(cam.source).exists():
            raise HTTPException(400, f"video for {source} not found: {cam.source} (run `make data`)")
    sp = "max" if str(speed) == "max" else float(speed)
    rid = run_id or new_run_id(f"{store_id}-{source.replace(':', '-')}")
    p = Pipeline(CFG, store, source, ROOT / "runs" / rid, speed=sp, duration_s=duration_s, live=True, seed=seed)
    h = RunHandle(p, store_id, source, speed)
    RUNS[rid] = h
    h.thread.start()
    log.info("started run %s (%s, %s, speed=%s)", rid, store_id, source, speed)
    return h


# ------------------------------------------------------------------ stores
@app.get("/api/health")
def health():
    from flowsight.perception.detector_tracker import select_device

    d, half = select_device(CFG.runtime.device)
    return {"ok": True, "device": d, "half": half, "runs": len(RUNS),
            "llm_briefing": bool(os.environ.get("ANTHROPIC_API_KEY"))}


def _store_summary(st) -> dict:
    cams = []
    for c in st.cameras:
        avail = c.source == "sim" or c.source.startswith(("rtsp://", "http")) or c.source.isdigit() or \
            resolve_path(c.source).exists()
        cams.append({"id": c.id, "name": c.name or c.id, "source": c.source, "available": avail,
                     "calibrated": c.calibration is not None, "approximate": c.approximate})
    return {"store_id": st.store_id, "name": st.name or st.store_id, "cameras": [c for c in cams if c["source"] != "sim"],
            "scenarios": list((st.sim or {}).get("scenarios", {}).keys()),
            "width_m": st.floorplan.width_m, "height_m": st.floorplan.height_m}


@app.get("/api/stores")
def stores():
    return [_store_summary(load_store(s)) for s in list_stores()]


@app.get("/api/stores/{store_id}")
def store_detail(store_id: str):
    st = _store_or_404(store_id)
    d = st.model_dump(exclude={"sim"})
    d.update(_store_summary(st))
    d["floorplan_url"] = f"/api/stores/{store_id}/floorplan.png"
    for c in d["cameras"]:
        c["frame_url"] = f"/api/stores/{store_id}/cameras/{c['id']}/frame.jpg"
    return d


@app.get("/api/stores/{store_id}/floorplan.png")
def floorplan(store_id: str):
    st = _store_or_404(store_id)
    p = resolve_path(st.floorplan.image)
    if p.exists():
        return FileResponse(p, media_type="image/png")
    ok, buf = cv2.imencode(".png", make_floorplan_image(st, CFG.render.floor_px_per_m))
    return Response(buf.tobytes(), media_type="image/png")


def _camera_frame(store_id: str, cam_id: str) -> np.ndarray:
    from flowsight.io.video_source import VideoSource
    from flowsight.privacy.anonymize import blur_heads

    st = _store_or_404(store_id)
    cam = st.camera(cam_id)
    frame = VideoSource(cam.source).read_first()
    try:  # privacy: blur heads on the calibration frame too
        from flowsight.perception.detector_tracker import DetectorTracker

        boxes = DetectorTracker(CFG.runtime).detect(frame)
        blur_heads(frame, boxes, CFG.privacy.head_fraction)
    except Exception as e:  # no weights offline -> blur the whole upper body region band as a fallback
        log.warning("detector unavailable for calibration frame (%s); applying global blur", e)
        frame = cv2.GaussianBlur(frame, (0, 0), 3)
    return frame


@app.get("/api/stores/{store_id}/cameras/{cam_id}/frame.jpg")
def camera_frame(store_id: str, cam_id: str):
    key = (store_id, cam_id)
    if key not in _FRAME_CACHE:
        try:
            frame = _camera_frame(store_id, cam_id)
        except (KeyError, FileNotFoundError, RuntimeError) as e:
            raise HTTPException(404, str(e))
        _FRAME_CACHE[key] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    return Response(_FRAME_CACHE[key], media_type="image/jpeg")


class ZonesIn(BaseModel):
    zones: list[ZoneCfg]


@app.put("/api/stores/{store_id}/zones")
def put_zones(store_id: str, body: ZonesIn):
    st = _store_or_404(store_id)
    st.zones = body.zones
    save_store(st)
    return {"ok": True, "zones": len(st.zones)}


class CalibIn(BaseModel):
    camera_id: str
    image_points: list[list[float]]
    floor_points: list[list[float]]
    save: bool = True


@app.put("/api/stores/{store_id}/calibration")
def put_calibration(store_id: str, body: CalibIn):
    st = _store_or_404(store_id)
    try:
        H = Homography(body.image_points, body.floor_points)
    except ValueError as e:
        raise HTTPException(400, str(e))
    err = H.reprojection_error()
    preview = None
    try:
        frame = cv2.imdecode(np.frombuffer(camera_frame(store_id, body.camera_id).body, np.uint8), cv2.IMREAD_COLOR)
        ppm = CFG.render.floor_px_per_m
        W, Hh = int(st.floorplan.width_m * ppm), int(st.floorplan.height_m * ppm)
        S = np.array([[ppm, 0, 0], [0, ppm, 0], [0, 0, 1]], float)
        warped = cv2.warpPerspective(frame, S @ H.H, (W, Hh))
        fp = make_floorplan_image(st, ppm)
        blend = cv2.addWeighted(fp, 0.45, warped, 0.75, 0)
        preview = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", blend)[1].tobytes()).decode()
    except HTTPException:
        pass
    if body.save:
        cam = st.camera(body.camera_id)
        cam.calibration = Calibration(image_points=body.image_points, floor_points=body.floor_points)
        save_store(st)
    return {"ok": True, "saved": body.save, "reprojection_error": err, "preview": preview}


# -------------------------------------------------------------------- runs
class RunIn(BaseModel):
    store_id: str = "demo_store"
    source: str = "sim:rush_hour"
    speed: str | float = 1
    duration_s: Optional[float] = None
    seed: int = 0


@app.post("/api/runs")
def create_run(body: RunIn):
    h = start_run(body.store_id, body.source, body.speed, body.duration_s, body.seed)
    return h.info()


@app.get("/api/runs")
def list_runs():
    return sorted([h.info() for h in RUNS.values()], key=lambda r: -r["created"])


@app.get("/api/runs/{run_id}")
def run_info(run_id: str):
    return _run(run_id).info()


@app.delete("/api/runs/{run_id}")
def stop_run(run_id: str):
    h = _run(run_id)
    h.stop.set()
    h.thread.join(timeout=10)
    return {"ok": True, "finished": h.p.finished}


@app.get("/api/runs/{run_id}/frame.jpg")
def latest_frame(run_id: str):
    h = _run(run_id)
    if h.p.latest_jpeg is None:
        raise HTTPException(404, "no frame yet")
    return Response(h.p.latest_jpeg, media_type="image/jpeg")


@app.get("/api/runs/{run_id}/stream.mjpg")
async def stream(run_id: str, max_frames: int = Query(0, ge=0)):
    h = _run(run_id)

    async def gen():
        last = None
        sent = 0
        t_wait = time.monotonic()
        while True:
            jpg = h.p.latest_jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                sent += 1
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + \
                    b"\r\n\r\n" + jpg + b"\r\n"
                if max_frames and sent >= max_frames:
                    return
            if h.p.finished and last is not None and last is h.p.latest_jpeg:
                return
            if jpg is None and time.monotonic() - t_wait > 30:
                return
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache"})


@app.websocket("/api/runs/{run_id}/live")
async def live(ws: WebSocket, run_id: str):
    await ws.accept()
    h = RUNS.get(run_id)
    if h is None:
        await ws.send_json({"error": f"run {run_id} not found"})
        await ws.close()
        return
    version = 0
    try:
        while True:
            payload = await asyncio.to_thread(h.p.live_payload, version)
            version = payload["insights_version"]
            await ws.send_json(_jsonable(payload))
            if h.p.finished:
                await asyncio.sleep(2.0)
            else:
                await asyncio.sleep(CFG.api.live_push_ms / 1000)
    except (WebSocketDisconnect, RuntimeError):
        return


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, set):
        return list(o)
    return o


def _locked(h: RunHandle, fn):
    with h.p.state_lock:
        return _jsonable(fn(h.p))


@app.get("/api/runs/{run_id}/kpis")
def kpis(run_id: str):
    return _locked(_run(run_id), lambda p: p.hub.kpis())


@app.get("/api/runs/{run_id}/zones")
def zones(run_id: str):
    def f(p):
        k = p.hub.zone_kpis()
        hist = {}
        for v in p.hub.visits.visits:
            hist.setdefault(v.zone_id, []).append(round(v.dwell_s, 1))
        occ = {}
        for w in p.hub.density.windows[-len(p.hub.zone_ids) * 360:]:
            occ.setdefault(w.zone_id, []).append([w.window_start, round(w.occupancy, 2), w.los])
        bns = {}
        for b in p.hub.density.bottlenecks:
            bns.setdefault(b.zone_id, []).append([b.t_start, b.t_end if b.t_end is not None else p.hub.ts])
        for zid, d in k.items():
            d["dwell_hist"] = hist.get(zid, [])[-2000:]
            d["occupancy_series"] = occ.get(zid, [])
            d["bottleneck_periods"] = bns.get(zid, [])
            a = (p.hub.attention_latest or {}).get(zid, {})
            d["funnel"] = {"exposures": a.get("exposures", 0), "entries": a.get("entries", 0),
                           "stops": a.get("stops", 0), "engaged_s": round(a.get("engaged_s", 0.0), 1)}
        return list(k.values())
    return _locked(_run(run_id), f)


@app.get("/api/runs/{run_id}/heatmap")
def heatmap(run_id: str, type: str = "occupancy", window: Optional[float] = None, format: str = "png"):
    h = _run(run_id)
    if type not in ("occupancy", "stationary"):
        raise HTTPException(400, "type must be occupancy|stationary")
    with h.p.state_lock:
        g = h.p.hub.heat.grid(type, window_s=window, now=h.p.hub.ts if window else None)
    if format == "json":
        return {"type": type, "window_s": window, "cell_m": h.p.hub.heat.cell, "nx": h.p.hub.heat.nx,
                "ny": h.p.hub.heat.ny, "max": float(g.max()), "total_person_s": float(g.sum()),
                "grid": np.round(g, 2).tolist()}
    img = heatmap_on_floorplan(g, h.p.floorplan, CFG.geometry.heatmap_blur_sigma_cells)
    return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png")


@app.get("/api/runs/{run_id}/flow")
def flow(run_id: str):
    return _locked(_run(run_id), lambda p: p.hub.flow.summary(p.hub.visits.visits + p.hub.visits.open_visits()))


@app.get("/api/runs/{run_id}/bottlenecks")
def bottlenecks(run_id: str):
    return _locked(_run(run_id), lambda p: p.hub.bottlenecks())


def _attention(p):
    att = p.hub.attention_latest or p.hub.compute_attention()
    rows = [dict(r, zone_name=p.hub.zname[z], zone_type=p.hub.ztype[z]) for z, r in att.items()]
    wi = what_if(att, p.hub.zname, p.hub.ztype, CFG.attention.window_min * 60, CFG.actions)
    return {"window_min": CFG.attention.window_min, "zones": rows, "what_if": wi}


@app.get("/api/runs/{run_id}/attention")
def attention(run_id: str):
    return _locked(_run(run_id), _attention)


@app.get("/api/runs/{run_id}/queue")
def queue(run_id: str):
    def f(p):
        out = []
        for st in p.hub.queue_status():
            q = p.hub.queues[st["zone_id"]]
            st["windows"] = [w.as_dict() for w in q.windows[-60:]]
            out.append(st)
        return out
    return _locked(_run(run_id), f)


@app.get("/api/runs/{run_id}/insights")
def insights(run_id: str, status: str = "active"):
    def f(p):
        items = p.engine.active_sorted() if status == "active" else p.engine.history()
        return [i.model_dump() for i in items]
    return _locked(_run(run_id), f)


@app.post("/api/runs/{run_id}/insights/{insight_id}/ack")
def ack(run_id: str, insight_id: str):
    h = _run(run_id)
    with h.p.state_lock:
        ins = h.p.engine.ack(insight_id)
    if ins is None:
        raise HTTPException(404, "insight not found")
    return ins.model_dump()


@app.get("/api/runs/{run_id}/evidence/{name}")
def evidence(run_id: str, name: str):
    h = _run(run_id)
    p = (h.p.run_dir / "evidence" / Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/runs/{run_id}/layout")
def layout(run_id: str):
    def f(p):
        a = _attention(p)
        rec: dict[str, dict] = {}
        for b in p.hub.density.bottlenecks:
            r = rec.setdefault(b.zone_id, {"zone_id": b.zone_id, "zone_name": p.hub.zname[b.zone_id], "count": 0,
                                           "time_lost_s": 0.0, "people_affected": 0, "worst_los": "A"})
            r["count"] += 1
            r["time_lost_s"] += b.time_lost_s
            r["people_affected"] += len(b.people)
            r["worst_los"] = max(r["worst_los"], b.peak_los)
        elapsed_h = max((p.hub.ts or 0) / 3600.0, 1e-6)
        for r in rec.values():
            r["per_hour"] = r["count"] / elapsed_h
        return {"what_if": a["what_if"], "recurring_bottlenecks": sorted(rec.values(), key=lambda r: -r["count"]),
                "ignored": [z for z in a["zones"] if z["status"] == "IGNORED"],
                "unvisited": [z for z in a["zones"] if z["status"] == "UNVISITED"],
                "hot": [z for z in a["zones"] if z["status"] == "HOT"]}
    return _locked(_run(run_id), f)


def _briefing(p):
    with p.state_lock:
        k = _jsonable(p.hub.kpis())
        ins = [i.model_dump() for i in p.engine.active_sorted()]
        zs = _jsonable(list(p.hub.zone_kpis().values()))
    return generate_briefing(CFG.briefing, k, ins, zs)


@app.get("/api/runs/{run_id}/briefing")
def briefing(run_id: str):
    return _jsonable(_briefing(_run(run_id).p))


@app.get("/api/runs/{run_id}/export.csv")
def export_csv(run_id: str):
    h = _run(run_id)
    with h.p.state_lock:
        rows = h.p.hub.zone_kpis()
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows.values()])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="flowsight_{run_id}_zones.csv"'})


@app.get("/api/runs/{run_id}/report.html", response_class=HTMLResponse)
def report_html(run_id: str):
    from jinja2 import Template

    h = _run(run_id)
    p = h.p
    b = _briefing(p)
    with p.state_lock:
        k = _jsonable(p.hub.kpis())
        zs = _jsonable(list(p.hub.zone_kpis().values()))
        ins = [i.model_dump() for i in p.engine.history()]
        bns = _jsonable(p.hub.bottlenecks())
        heat = {t: heatmap_on_floorplan(p.hub.heat.grid(t), p.floorplan, CFG.geometry.heatmap_blur_sigma_cells)
                for t in ("occupancy", "stationary")}
    imgs = {t: "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im)[1].tobytes()).decode()
            for t, im in heat.items()}
    tpl = Template((Path(__file__).parent / "report_template.html").read_text(encoding="utf-8"))
    return tpl.render(run=h.info(), k=k, zones=zs, insights=ins, bottlenecks=bns, briefing=b, imgs=imgs,
                      fmt_dur=fmt_dur, generated=time.strftime("%Y-%m-%d %H:%M"))


# ------------------------------------------------------------- demo + web
@app.on_event("startup")
def _demo_autostart():
    if os.environ.get("FLOWSIGHT_DEMO") != "1":
        return
    try:
        start_run("demo_store", "sim:rush_hour", speed=4, duration_s=3600, run_id=new_run_id("demo-rush"))
    except Exception as e:
        log.error("demo sim failed to start: %s", e)
    weights = resolve_path(CFG.runtime.models_dir) / CFG.runtime.model_gpu
    try:
        st = load_store("caviar_corridor")
        if weights.exists() or (resolve_path(CFG.runtime.models_dir) / CFG.runtime.model_cpu).exists():
            if resolve_path(st.camera("cam1").source).exists():
                start_run("caviar_corridor", "cam1", speed=1, run_id=new_run_id("demo-caviar"))
        else:
            log.warning("YOLO weights missing offline; demo runs the simulator only")
    except Exception as e:
        log.warning("real clip not started: %s", e)


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / full_path
        if full_path and f.exists() and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
