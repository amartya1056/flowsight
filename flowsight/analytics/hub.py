"""AnalyticsHub — wires every analytics module together.

It consumes *floor-coordinate track states with timestamps* and nothing else, which is exactly why the
simulator can validate it: video and simulator feed the same hub.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from flowsight.analytics.attention import AttentionAnalyzer
from flowsight.analytics.baseline import RollingBaseline
from flowsight.analytics.density import DensityAnalyzer
from flowsight.analytics.flow import FlowAnalyzer
from flowsight.analytics.heatmap import Heatmap
from flowsight.analytics.queue import QueueAnalyzer
from flowsight.analytics.visits import VisitTracker
from flowsight.geometry.zones import ZoneSet
from flowsight.tracking.staff import StaffClassifier


class AnalyticsHub:
    def __init__(self, cfg, store, db=None, t_start: float = 0.0, cam_id: str = "cam1"):
        self.cfg, self.store, self.db, self.cam_id = cfg, store, db, cam_id
        self.t_start = t_start
        self.zones = ZoneSet(store.zones, cfg.visits.exposure_buffer_m)
        self.zone_ids = self.zones.ids
        self.ztype = {z.id: z.type for z in self.zones}
        self.zname = {z.id: z.name for z in self.zones}
        self.visits = VisitTracker(self.zone_ids, cfg.visits, on_visit=self._on_visit, on_entry=self._on_entry)
        self.heat = Heatmap(store.floorplan.width_m, store.floorplan.height_m, cfg.geometry.heatmap_cell_m,
                            engaged_speed=cfg.visits.engaged_speed_mps)
        self.flow = FlowAnalyzer(self.zones, cfg.flow)
        self.density = DensityAnalyzer(self.zones, cfg.density, on_window=self._on_window,
                                       on_bottleneck=self._on_bottleneck)
        self.attention = AttentionAnalyzer(self.zones, cfg.attention)
        self.density.on_conflict = self._on_conflict
        self.queues = {z.id: QueueAnalyzer(z.id, cfg.queue, t_start, on_window=self._on_queue_window)
                       for z in self.zones if z.type == cfg.queue.service_zone_type}
        self.overflow = {z.id: [o for o in z.overflow if o in self.ztype] for z in store.zones if z.overflow}
        self.baseline = RollingBaseline(cfg.baseline.history_windows, cfg.baseline.min_samples, cfg.baseline.min_std)
        self.ts: float | None = None
        self.n_frames = 0
        self.tracks: list = []
        self.inside = np.zeros((0, len(self.zone_ids)), bool)
        self.stationary_since: dict[int, float] = {}
        self.trails: dict[int, deque] = {}
        self._last_pt: dict[int, float] = {}
        self._pt_period = 1.0 / max(cfg.storage.track_point_hz, 1e-3)
        self.attention_latest: dict = {}
        self._last_attention = -1e9
        self._last_attention_db = -1e9
        self.peak_people = 0
        self.extra: dict = {}  # pipeline-level counters (stitches, fps, ...)
        self.staff = StaffClassifier(cfg.tracking)
        self.staff_on_floor = 0
        self.staff_visits_purged = 0

    # ------------------------------------------------------------ callbacks
    def _on_visit(self, v):
        if self.db is not None:
            self.db.add("visits", {k: getattr(v, k) for k in
                                   ("track_id", "zone_id", "t_enter", "t_exit", "dwell_s", "engaged_s", "is_stop", "mean_speed")})
        self.baseline.add(v.zone_id, "dwell", v.dwell_s)
        q = self.queues.get(v.zone_id)
        if q is not None:
            q.on_exit(v.t_exit, v.dwell_s)

    def _on_entry(self, tid, zid, t):
        q = self.queues.get(zid)
        if q is not None:
            q.on_entry(t)

    def _on_window(self, w):
        if self.db is not None:
            self.db.add("zone_windows", w.as_dict())
        self.baseline.add(w.zone_id, "occupancy", w.occupancy)
        self.baseline.add(w.zone_id, "density", w.density)

    def _on_bottleneck(self, kind, b):
        if kind == "end" and self.db is not None:
            self.db.add("bottlenecks", {k: v for k, v in b.as_dict().items() if k != "active"})

    def _on_conflict(self, c):
        if self.db is not None:
            self.db.add("conflicts", {k: v for k, v in c.as_dict().items() if k != "active"})

    def conflicts(self) -> list[dict]:
        now = self.ts or 0.0
        return [dict(c.as_dict(now), zone_name=self.zname[c.zone_id]) for c in self.density.conflicts]

    def _on_queue_window(self, w):
        if self.db is not None:
            self.db.add("queue_windows", w.as_dict())
        if w.predicted_wait_s is not None:
            self.baseline.add(w.zone_id, "queue_wait", w.predicted_wait_s)
        self.baseline.add(w.zone_id, "queue_len", w.queue_len)

    # ----------------------------------------------------------------- step
    def step(self, ts: float, tracks, expired=()) -> None:
        dt = 0.0 if self.ts is None else min(max(ts - self.ts, 0.0), 1.0)
        if self.ts is None and self.cfg.runtime.target_fps > 0:
            dt = 1.0 / self.cfg.runtime.target_fps
        self.ts = ts
        self.n_frames += 1
        mature = [t for t in tracks if t.mature]
        if self.cfg.tracking.staff_filter:
            for t in mature:
                if self.staff.check_presence(t.id, t.age):
                    self.mark_staff(t.id)
            self.staff_on_floor = sum(1 for t in mature if self.staff.is_staff(t.id))
            mature = [t for t in mature if not self.staff.is_staff(t.id)]
        self.tracks = mature
        self.peak_people = max(self.peak_people, len(mature))
        ids = [t.id for t in mature]
        xs = np.array([t.x for t in mature], float)
        ys = np.array([t.y for t in mature], float)
        sp = np.array([t.speed for t in mature], float)
        inside = self.zones.membership(xs, ys)
        exposed = self.zones.exposure_membership(xs, ys)
        self.inside = inside

        self.visits.update(ts, ids, sp, inside, exposed)
        self.heat.update(ts, dt, xs, ys, sp)
        self.density.update(ts, dt, ids, sp, inside, vx=np.array([t.vx for t in mature], float),
                            vy=np.array([t.vy for t in mature], float))
        thr = self.cfg.visits.engaged_speed_mps
        slow = sp < thr
        for j, zid in enumerate(self.zone_ids):
            q = self.queues.get(zid)
            if q is not None:
                over = 0
                for oz in self.overflow.get(zid, ()):
                    k = self.zone_ids.index(oz)
                    over += int((inside[:, k] & slow & ~inside[:, j]).sum()) if len(ids) else 0
                q.sample(ts, int((inside[:, j] & slow).sum()) if len(ids) else 0, over)

        trail_n = int(self.cfg.render.trail_s * max(self.cfg.runtime.target_fps, 1)) + 1
        for t in mature:
            if t.speed < thr:
                self.stationary_since.setdefault(t.id, ts)
            else:
                self.stationary_since.pop(t.id, None)
            self.flow.observe_track(t.id, ts, getattr(t, "path_len", 0.0))
            tr = self.trails.get(t.id)
            if tr is None:
                tr = self.trails[t.id] = deque(maxlen=trail_n)
            tr.append((t.x, t.y))

        for tr in expired:
            self.visits.close_track(tr.id)
            self.stationary_since.pop(tr.id, None)
            self.trails.pop(tr.id, None)
            self._last_pt.pop(tr.id, None)
            self.flow.observe_track(tr.id, tr.last_ts, tr.path_len)

        if self.db is not None:
            self.db.add("frames", {"ts": ts, "n_people": len(mature)})
            for t in mature:
                if ts - self._last_pt.get(t.id, -1e9) >= self._pt_period - 1e-9:
                    self._last_pt[t.id] = ts
                    self.db.add("track_points", {"track_id": t.id, "ts": ts, "x": t.x, "y": t.y,
                                                 "speed": t.speed, "cam": self.cam_id})
        if ts - self._last_attention >= self.cfg.actions.eval_every_s:
            self._last_attention = ts
            self.attention_latest = self.compute_attention()
            if self.db is not None and ts - self._last_attention_db >= 60:
                self._last_attention_db = ts
                self.db.add_many("attention_windows", [
                    {"ts": ts, **{k: r.get(k) for k in ("zone_id", "exposures", "entries", "stops", "engaged_s",
                                                         "stopping_power", "status")}}
                    for r in self.attention_latest.values()])

    def mark_staff(self, tid: int) -> None:
        """A track was identified as staff: purge it from shopper analytics retroactively."""
        self.staff_visits_purged += self.visits.purge_track(tid)
        for d in (self.flow.track_first, self.flow.track_last, self.flow.track_len, self.trails,
                  self.stationary_since, self._last_pt):
            d.pop(tid, None)
        if self.db is not None:
            self.db.delete_track(tid)

    def observe_staff_color(self, tid: int, frame, bbox) -> None:
        if self.cfg.tracking.staff_filter and self.staff.observe_color(tid, frame, bbox):
            self.mark_staff(tid)

    def compute_attention(self, window_s: float | None = None) -> dict:
        now = self.ts or 0.0
        return self.attention.compute(now, self.visits.exposures, self.visits.entries, self.visits.visits,
                                      self.visits.open_visits(), window_s)

    def finalize(self, ts: float | None = None) -> None:
        ts = self.ts if ts is None else ts
        self.visits.finalize()
        self.density.finalize(ts or 0.0)
        for q in self.queues.values():
            q.finalize()
        self.attention_latest = self.compute_attention()
        if self.db is not None:
            self.db.add_many("attention_windows", [
                {"ts": ts, **{k: r.get(k) for k in ("zone_id", "exposures", "entries", "stops", "engaged_s",
                                                     "stopping_power", "status")}}
                for r in self.attention_latest.values()])
            self.db.flush()

    # ------------------------------------------------------------- read API
    def occupancy(self) -> dict[str, int]:
        if self.inside.size == 0:
            return {z: 0 for z in self.zone_ids}
        return {z: int(self.inside[:, j].sum()) for j, z in enumerate(self.zone_ids)}

    def zone_kpis(self) -> dict:
        k = self.visits.zone_kpis(self.occupancy())
        att = self.attention_latest or {}
        for zid, d in k.items():
            inst = self.density.instant.get(zid, {})
            lw = self.density.latest.get(zid)
            a = att.get(zid, {})
            d.update({
                "zone_id": zid, "name": self.zname[zid], "type": self.ztype[zid],
                "area_m2": round(self.zones[zid].area, 2),
                "los": inst.get("los", "A"), "density": round(inst.get("density", 0.0), 3),
                "window_los": lw.los if lw else "A", "mean_speed": lw.mean_speed if lw else None,
                "free_flow_mps": round(self.density.free_flow[zid], 2),
                "exposures": a.get("exposures", 0), "stopping_power": a.get("stopping_power"),
                "attention_status": a.get("status", "-"), "attention_reason": a.get("reason", ""),
                "bottleneck_active": self.density.active.get(zid) is not None,
            })
        return k

    def kpis(self) -> dict:
        now = self.ts or 0.0
        flow = self.flow.summary(self.visits.visits + self.visits.open_visits())
        dw = [v.dwell_s for v in self.visits.visits if self.ztype.get(v.zone_id) not in ("walkway",)]
        waits = [q.status(now) for q in self.queues.values()]
        wait = max((w["predicted_wait_s"] or 0.0) for w in waits) if waits else None
        return {
            "ts": now, "elapsed_s": now - self.t_start,
            "people_in_store": len(self.tracks), "peak_people": self.peak_people,
            "visitors_total": flow["tracks"],
            "avg_zone_dwell_s": float(np.mean(dw)) if dw else 0.0,
            "avg_time_in_store_s": flow["avg_time_in_store_s"],
            "avg_path_length_m": flow["avg_path_length_m"],
            "checkout_wait_s": wait,
            "checkout_queue": sum(w["queue_len_now"] for w in waits) if waits else None,
            "active_bottlenecks": len(self.density.active_bottlenecks()),
            "bottlenecks_total": len(self.density.bottlenecks),
            "conversion": flow["conversion"],
            "visits_total": len(self.visits.visits),
            "staff_on_floor": self.staff_on_floor, "staff_tracks": len(self.staff.staff),
            "staff_visits_purged": self.staff_visits_purged,
            **self.extra,
        }

    def people(self) -> list[dict]:
        now = self.ts or 0.0
        return [{"id": t.id, "x": round(t.x, 3), "y": round(t.y, 3), "speed": round(t.speed, 3),
                 "stationary_s": round(now - self.stationary_since[t.id], 1) if t.id in self.stationary_since else 0.0}
                for t in self.tracks]

    def live_zones(self) -> list[dict]:
        occ = self.occupancy()
        opened = self.visits.open_visits()
        out = []
        for zid in self.zone_ids:
            inst = self.density.instant[zid]
            live = [v.dwell_s for v in opened if v.zone_id == zid]
            out.append({"id": zid, "occupancy": occ[zid], "los": inst["los"], "density": round(inst["density"], 3),
                        "dwell_live": round(float(np.mean(live)), 1) if live else 0.0,
                        "bottleneck": self.density.active.get(zid) is not None,
                        "status": (self.attention_latest.get(zid) or {}).get("status", "-")})
        return out

    def bottlenecks(self) -> list[dict]:
        now = self.ts or 0.0
        return [dict(b.as_dict(now), zone_name=self.zname[b.zone_id]) for b in self.density.bottlenecks]

    def queue_status(self) -> list[dict]:
        now = self.ts or 0.0
        return [dict(q.status(now), zone_name=self.zname[z]) for z, q in self.queues.items()]
