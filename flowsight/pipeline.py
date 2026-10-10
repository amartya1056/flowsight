"""Pipeline: source -> perception -> floor projection -> tracks -> analytics -> actions -> render/storage.

Two front-ends feed the *same* analytics hub:
  * video  (file / RTSP / webcam): YOLO11 + ByteTrack -> foot point -> homography -> TrackStore
  * sim:<scenario>: simulator trajectories (with realistic noise, dropouts and ID breaks) -> TrackStore
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from flowsight.actions.engine import ActionEngine
from flowsight.analytics.heatmap import heatmap_on_camera, heatmap_on_floorplan
from flowsight.analytics.hub import AnalyticsHub
from flowsight.config import ROOT, AppConfig, StoreConfig, resolve_path
from flowsight.geometry.calibration import build_mapping
from flowsight.privacy.anonymize import blur_heads
from flowsight.render.annotate import Renderer, make_floorplan_image
from flowsight.storage.db import RunDB
from flowsight.tracking.track_store import TrackStore

log = logging.getLogger(__name__)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


class Pipeline:
    def __init__(self, cfg: AppConfig, store: StoreConfig, source: str, run_dir: str | Path,
                 speed: float | str = "max", duration_s: float | None = None, write_video: bool = True,
                 live: bool = False, seed: int = 0, sim_overrides: dict | None = None,
                 model_name: str | None = None, store_db: bool = True):
        self.cfg, self.store, self.source = cfg, store, source
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name
        self.speed = speed
        self.duration_s = duration_s
        self.write_video = write_video
        self.live = live
        self.seed = seed
        self.is_sim = source.startswith("sim")
        self.is_multi = source.startswith("multi")
        self.db = RunDB(self.run_dir / "flowsight.duckdb", cfg.storage.flush_every_s) if store_db else None
        self.hub = AnalyticsHub(cfg, store, self.db, cam_id=source)
        self.engine = ActionEngine(cfg, self.hub, snapshot_fn=self._snapshot, on_change=self._on_insights)
        self.floorplan = self._load_floorplan()
        self.lock = threading.Lock()
        self.state_lock = threading.RLock()  # guards hub/engine between pipeline thread and API readers
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.finished = False
        self.error: str | None = None
        self.progress = 0.0
        self.proc_fps = 0.0
        self.device = "n/a"
        self.model = "n/a"
        self.calibrated = True
        self.frame_size = (0, 0)
        self._last_render = 0.0
        self._sample_saved = False
        self._ins_dirty: list = []
        self.started_wall = time.time()

        if self.is_sim:
            from flowsight.sim.simulator import Simulator

            scenario = source.split(":", 1)[1] if ":" in source else "baseline"
            self.sim = Simulator(store, cfg, scenario=scenario, seed=seed, overrides=sim_overrides or {})
            self.tracks = TrackStore(cfg.tracking)
            self.renderer = Renderer(store, cfg, None, self.floorplan)
            self.device, self.model = "simulator", f"sim:{scenario}"
        elif self.is_multi:
            self._init_multi(source, model_name)
        else:
            from flowsight.io.video_source import VideoSource
            from flowsight.perception.detector_tracker import DetectorTracker
            from flowsight.perception.footpoint import FootpointEstimator

            cam = store.camera(source)
            self.cam = cam
            self.video = VideoSource(cam.source, cfg.runtime.target_fps, max_seconds=duration_s)
            self.frame_size = (self.video.info.width, self.video.info.height)
            self.mapping = build_mapping(cam, cfg.geometry.assumed_person_height_m, self.video.info.width)
            self.calibrated = bool(getattr(self.mapping, "calibrated", False))
            self.det = DetectorTracker(cfg.runtime, model_name=model_name)
            self.device, self.model = self.det.device, self.det.model_name
            self.foot = FootpointEstimator()
            self.tracks = TrackStore(cfg.tracking, metric=True)  # both Homography and AutoScale output metres
            self.renderer = Renderer(store, cfg, self.mapping, self.floorplan)
        if self.db is not None:
            self.db.add("runs", {"run_id": self.run_id, "store_id": store.store_id, "source": source,
                                 "started_at": time.time(), "ended_at": None, "device": self.device,
                                 "model": self.model, "calibrated": self.calibrated,
                                 "meta": json.dumps({"speed": str(speed), "seed": seed})})

    def _init_multi(self, source: str, model_name: str | None) -> None:
        """source = 'multi' (all calibrated cameras) or 'multi:cam1,cam2'."""
        from flowsight.io.video_source import VideoSource
        from flowsight.perception.detector_tracker import DetectorTracker
        from flowsight.perception.footpoint import FootpointEstimator
        from flowsight.tracking.fusion import CrossCameraFuser

        ids = source.split(":", 1)[1].split(",") if ":" in source else \
            [c.id for c in self.store.cameras if c.calibration is not None and c.source != "sim"]
        cams = [self.store.camera(i) for i in ids]
        min_k = min(c.time_offset_frames for c in cams)
        self.mc = []
        for c in cams:  # camera i's frame f_i = f_ref + k_i  ->  skip (k_i - min_k) frames to align clocks
            vs = VideoSource(c.source, self.cfg.runtime.target_fps, max_seconds=self.duration_s,
                             start_frame=c.time_offset_frames - min_k)
            mp = build_mapping(c, self.cfg.geometry.assumed_person_height_m, vs.info.width)
            self.mc.append({"cam": c, "video": vs, "mapping": mp, "det": DetectorTracker(self.cfg.runtime, model_name=model_name),
                            "foot": FootpointEstimator(), "renderer": Renderer(self.store, self.cfg, mp, self.floorplan)})
        self.fuser = CrossCameraFuser(ids, gate_m=self.cfg.tracking.fusion_gate_m)
        self.tracks = TrackStore(self.cfg.tracking, metric=True)
        self.renderer = Renderer(self.store, self.cfg, None, self.floorplan)
        self.device = self.mc[0]["det"].device
        self.model = f"{self.mc[0]['det'].model_name} x{len(self.mc)} cams"
        self.frame_size = (sum(m["video"].info.width for m in self.mc), self.mc[0]["video"].info.height)

    # --------------------------------------------------------------- helpers
    def _load_floorplan(self) -> np.ndarray:
        p = resolve_path(self.store.floorplan.image)
        img = cv2.imread(str(p)) if p.exists() else None
        if img is None:
            img = make_floorplan_image(self.store, self.cfg.render.floor_px_per_m)
        return img

    def _snapshot(self, ins) -> str | None:
        if self.cfg.privacy.store_frames == "none" or ins.zone_id is None:
            return None
        d = self.run_dir / "evidence"
        d.mkdir(exist_ok=True)
        if self.is_sim or self.is_multi or self.latest_frame is None or not self.calibrated:
            img = self.renderer.draw_floor(self.hub, self.hub.ts or 0.0)
            img = self.renderer.highlight_zone(img, ins.zone_id, camera=False)
        else:
            img = self.renderer.highlight_zone(self.latest_frame, ins.zone_id, camera=True)  # already blurred
        name = f"{ins.id}.jpg"
        cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f"evidence/{name}"

    def _on_insights(self, changed):
        if self.db is not None:
            self.db.upsert_insights([{**i.model_dump(), "evidence": i.evidence.model_dump()} for i in changed])

    def _publish(self, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.render.jpeg_quality])
        if ok:
            with self.lock:
                self.latest_jpeg = buf.tobytes()

    def _pace(self, ts: float, wall0: float) -> None:
        if self.speed in ("max", None, 0):
            return
        target = wall0 + ts / float(self.speed)
        d = target - time.monotonic()
        if d > 0:
            time.sleep(min(d, 0.5))

    # ------------------------------------------------------------------ run
    def run(self, stop_event: threading.Event | None = None) -> dict:
        try:
            if self.is_sim:
                self._run_sim(stop_event)
            elif self.is_multi:
                self._run_multi(stop_event)
            else:
                self._run_video(stop_event)
        except Exception as e:  # keep the API alive; surface the error
            log.exception("pipeline failed")
            self.error = f"{type(e).__name__}: {e}"
        finally:
            summary = self._finish()
            self.finished = True
        return summary

    def _step_analytics(self, ts: float, states) -> None:
        expired = self.tracks.pop_expired()
        self.hub.extra.update({"id_stitches": self.tracks.n_stitches, "tracks_created": self.tracks.n_tracks_created,
                               "proc_fps": round(self.proc_fps, 1)})
        with self.state_lock:
            self.hub.step(ts, states, expired)
            self.engine.evaluate(ts)
        if self.db is not None:
            self.db.maybe_flush()

    def _run_sim(self, stop_event) -> None:
        dur = self.duration_s or self.cfg.sim.duration_s
        wall0 = time.monotonic()
        t_fps, n_fps = time.monotonic(), 0
        render_dt = 0.1 if self.live else None
        while self.sim.t < dur:
            if stop_event is not None and stop_event.is_set():
                break
            ts, obs = self.sim.step()
            states = self.tracks.update(ts, obs)
            self._step_analytics(ts, states)
            n_fps += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.proc_fps = n_fps / (now - t_fps)
                t_fps, n_fps = now, 0
            if render_dt is not None and now - self._last_render >= render_dt:
                self._last_render = now
                with self.state_lock:
                    img = self.renderer.draw_floor(self.hub, ts, title=f"{self.source}  t={ts:6.0f}s")
                self.latest_frame = img
                self._publish(img)
            self.progress = min(1.0, ts / dur)
            self._pace(ts, wall0)
        img = self.renderer.draw_floor(self.hub, self.hub.ts or 0.0, title=f"{self.source} (finished)")
        self.latest_frame = img
        self._publish(img)

    def _run_video(self, stop_event) -> None:
        cfg = self.cfg
        writer = None
        out_path = self.run_dir / "annotated.mp4"
        wall0 = time.monotonic()
        t_fps, n_fps = time.monotonic(), 0
        n_total = self.video.info.n_frames
        H = self.video.info.height
        for idx, ts, frame in self.video:
            if stop_event is not None and stop_event.is_set():
                break
            dets = self.det.update(frame)
            obs, feet, boxes = [], {}, {}
            for d in dets:
                fx, fy, trunc = self.foot(d.track_id, d.bbox_xyxy, H)
                if not self.calibrated:
                    self.mapping.add_box(d.bbox_xyxy, trunc)
                feet[d.track_id] = (fx, fy, trunc)
                boxes[d.track_id] = d.bbox_xyxy
            if feet:
                raw_ids = list(feet)
                floor = self.mapping.img_to_floor(np.array([[feet[r][0], feet[r][1]] for r in raw_ids]))
                obs = [(r, float(floor[i, 0]), float(floor[i, 1]), feet[r][2]) for i, r in enumerate(raw_ids)]
            states = self.tracks.update(ts, obs)
            self._step_analytics(ts, states)

            # privacy first: blur heads on the frame before anything renders or leaves the pipeline
            vis = frame.copy()
            if cfg.privacy.blur_heads:
                blur_heads(vis, list(boxes.values()), cfg.privacy.head_fraction)
            stable = {}
            for r, b in boxes.items():
                tid = self.tracks.raw_to_id.get(r)
                if tid is not None:
                    stable[tid] = (b, feet[r][:2])
                    if cfg.tracking.staff_uniform_hsv:
                        with self.state_lock:
                            self.hub.observe_staff_color(tid, frame, b)  # raw frame: colour only, torso only
            self.latest_frame = vis.copy()
            with self.state_lock:
                ann = self.renderer.draw_camera(vis, stable, self.hub, ts)
                self._hud(ann, ts)
            if self.write_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.runtime.target_fps,
                                             (ann.shape[1], ann.shape[0]))
                writer.write(ann)
            if not self._sample_saved and ts > 2.0 and len(boxes) > 0:
                cv2.imwrite(str(self.run_dir / "sample.jpg"), ann)
                self._sample_saved = True
            self._publish(ann)
            n_fps += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.proc_fps = n_fps / (now - t_fps)
                t_fps, n_fps = now, 0
            if n_total > 0:
                self.progress = min(1.0, idx / n_total)
            self._pace(ts, wall0)
        if writer is not None:
            writer.release()
            self._transcode(out_path)

    def _run_multi(self, stop_event) -> None:
        cfg = self.cfg
        writer = None
        out_path = self.run_dir / "annotated.mp4"
        wall0 = time.monotonic()
        t_fps, n_fps = time.monotonic(), 0
        iters = [iter(m["video"]) for m in self.mc]
        n_total = min(m["video"].info.n_frames for m in self.mc)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            items = [next(it, None) for it in iters]
            if any(x is None for x in items):
                break
            ts = items[0][1]
            obs_by_cam, per_cam = {}, []
            for m, (idx, _, frame) in zip(self.mc, items):
                H = frame.shape[0]
                dets = m["det"].update(frame)
                feet, boxes = {}, {}
                for d in dets:
                    feet[d.track_id] = m["foot"](d.track_id, d.bbox_xyxy, H)
                    boxes[d.track_id] = d.bbox_xyxy
                obs = []
                if feet:
                    raw = list(feet)
                    fl = m["mapping"].img_to_floor(np.array([[feet[r][0], feet[r][1]] for r in raw]))
                    obs = [(r, float(fl[i, 0]), float(fl[i, 1])) for i, r in enumerate(raw)]
                obs_by_cam[m["cam"].id] = obs
                per_cam.append((m, frame, feet, boxes))
            fused = self.fuser.fuse(obs_by_cam)
            states = self.tracks.update(ts, [(f[0], f[1], f[2]) for f in fused])
            self.hub.extra.update({"fusion": self.fuser.stats.summary()})
            self._step_analytics(ts, states)

            panels = []
            for m, frame, feet, boxes in per_cam:
                vis = frame.copy()
                if cfg.privacy.blur_heads:
                    blur_heads(vis, list(boxes.values()), cfg.privacy.head_fraction)
                stable = {}
                for r, b in boxes.items():
                    key = self.fuser.key_of.get((m["cam"].id, r))
                    tid = self.tracks.raw_to_id.get(key) if key is not None else None
                    if tid is not None:
                        stable[tid] = (b, feet[r][:2])
                with self.state_lock:
                    ann = m["renderer"].draw_camera(vis, stable, self.hub, ts)
                cv2.putText(ann, m["cam"].name or m["cam"].id, (6, ann.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(ann)
            h = panels[0].shape[0]
            with self.state_lock:
                floor = self.renderer.draw_floor(self.hub, ts)
            floor = cv2.resize(floor, (max(1, int(floor.shape[1] * h / floor.shape[0])), h))
            comp = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels] + [floor])
            self._hud(comp, ts)
            self.latest_frame = comp
            if self.write_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.runtime.target_fps,
                                             (comp.shape[1], comp.shape[0]))
                writer.write(comp)
            if not self._sample_saved and ts > 2.0 and any(p[3] for p in per_cam):
                cv2.imwrite(str(self.run_dir / "sample.jpg"), comp)
                self._sample_saved = True
            self._publish(comp)
            n_fps += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                self.proc_fps = n_fps / (now - t_fps)
                t_fps, n_fps = now, 0
            if n_total > 0:
                self.progress = min(1.0, items[0][0] / n_total)
            self._pace(ts, wall0)
        if writer is not None:
            writer.release()
            self._transcode(out_path)

    def _hud(self, img, ts):
        k = self.hub.kpis()
        txt = (f"t={ts:6.1f}s  people={k['people_in_store']}  fps={self.proc_fps:4.1f}  "
               f"bottlenecks={k['active_bottlenecks']}  {'CAL' if self.calibrated else 'UNCALIBRATED'}  privacy:ON")
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (25, 25, 25), -1)
        cv2.putText(img, txt, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _transcode(path: Path) -> None:
        """Re-encode to H.264 so browsers can play it (best effort; keeps mp4v file if ffmpeg missing)."""
        ff = shutil.which("ffmpeg")
        if not ff:
            return
        tmp = path.with_name(path.stem + "_h264.mp4")
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "veryfast", "-crf", "26", str(tmp)], capture_output=True)
        if r.returncode == 0 and tmp.exists():
            tmp.replace(path)

    # --------------------------------------------------------------- finish
    def _finish(self) -> dict:
        with self.state_lock:
            return self._finish_locked()

    def _finish_locked(self) -> dict:
        ts = self.hub.ts or 0.0
        expired = self.tracks.flush_all()
        self.hub.step(ts + 1e-3, [], expired) if self.hub.ts is not None else None
        self.hub.finalize(ts)
        self.engine.evaluate(ts, force=True)
        # outputs: heatmaps on the floor plan (+ camera view when calibrated)
        for kind in ("occupancy", "stationary"):
            g = self.hub.heat.grid(kind)
            cv2.imwrite(str(self.run_dir / f"heatmap_{kind}.png"),
                        heatmap_on_floorplan(g, self.floorplan, self.cfg.geometry.heatmap_blur_sigma_cells))
            if not self.is_sim and not self.is_multi and self.calibrated and self.latest_frame is not None:
                cv2.imwrite(str(self.run_dir / f"heatmap_{kind}_camera.jpg"),
                            heatmap_on_camera(g, self.latest_frame, self.mapping.floor_to_img_matrix(),
                                              self.store.floorplan.width_m, self.store.floorplan.height_m,
                                              self.cfg.geometry.heatmap_blur_sigma_cells))
        summary = {
            "run_id": self.run_id, "store_id": self.store.store_id, "source": self.source,
            "device": self.device, "model": self.model, "calibrated": self.calibrated,
            "error": self.error, "kpis": self.hub.kpis(),
            "id_stitches": self.tracks.n_stitches, "tracks_created": self.tracks.n_tracks_created,
            "proc_fps": round(self.proc_fps, 2), "wall_s": round(time.time() - self.started_wall, 1),
            "active_insights": len(self.engine.active), "insights_total": len(self.engine.all),
        }
        if self.is_multi:
            summary["fusion"] = self.fuser.stats.summary()
        if self.db is not None:
            self.db.flush()
            self.db.con.execute("UPDATE runs SET ended_at = ? WHERE run_id = ?", [time.time(), self.run_id])
            summary["db_counts"] = self.db.counts()
            try:
                self.db.export_parquet(self.run_dir / "parquet")
            except Exception as e:
                log.warning("parquet export failed: %s", e)
        if self.error is None:
            self.progress = 1.0
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return summary

    # ---------------------------------------------------------- live payload
    def live_payload(self, since_version: int = 0) -> dict:
        h = self.hub
        with self.state_lock:
            return {"ts": h.ts or 0.0, "people": h.people(), "zones": h.live_zones(),
                    "insights_delta": self.engine.delta_since(since_version), "insights_version": self.engine.version,
                    "kpis": h.kpis(), "finished": self.finished, "progress": round(self.progress, 4),
                    "error": self.error}


def default_run_dir(run_id: str | None = None) -> Path:
    return ROOT / "runs" / (run_id or new_run_id())
