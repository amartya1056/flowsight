"""Density, Fruin Level-of-Service, free-flow speed baselines and bottleneck detection.

Per zone and per `window_s` window we compute: mean occupancy, density (people/m²), area per person ->
Fruin LoS A..F, people-weighted mean speed, and flow (zone entries per minute).

Counter-flow: for walkers (speed > moving_speed_mps) we histogram headings into 8 bins (normalised Shannon
*direction entropy*) and project velocities on the zone's main axis; counterflow = min(n+, n-) / (n+ + n-),
0 = one-way, 0.5 = perfectly two-way. A *conflict* episode = counterflow >= counterflow_min_ratio at LoS >=
counterflow_min_los for >= bottleneck_min_duration_s.

Each zone learns its own *free-flow speed* (p75 of individual speeds observed while the zone is at LoS
A/B). A **bottleneck** is declared when, for >= bottleneck_min_duration_s, a walkway/entrance/aisle zone
is at LoS D or worse, OR its mean speed falls below speed_drop_ratio x free-flow while holding >= 3
people. For each bottleneck we record peak density, people affected and *time lost*:
Σ dt·max(0, 1 − v/v_free) over everyone in the zone — i.e. actual traversal time minus free-flow time.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

LOS_ORDER = ["A", "B", "C", "D", "E", "F"]


def fruin_los(area_per_person: float, thresholds: dict[str, float]) -> str:
    for los in ["A", "B", "C", "D", "E"]:
        if area_per_person >= thresholds[los]:
            return los
    return "F"


def los_rank(los: str) -> int:
    return LOS_ORDER.index(los)


@dataclass
class ZoneWindow:
    zone_id: str
    window_start: float
    window_end: float
    occupancy: float
    density: float
    los: str
    mean_speed: float | None
    flow_per_min: float
    free_flow_mps: float
    direction_entropy: float | None = None
    counterflow: float | None = None

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Bottleneck:
    zone_id: str
    t_start: float
    t_confirmed: float
    t_end: float | None = None
    peak_density: float = 0.0
    peak_los: str = "A"
    people: set = field(default_factory=set)
    time_lost_s: float = 0.0
    mean_speed_min: float | None = None
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.t_end is None

    def as_dict(self, now: float | None = None) -> dict:
        end = self.t_end if self.t_end is not None else now
        return {"zone_id": self.zone_id, "t_start": self.t_start, "t_confirmed": self.t_confirmed,
                "t_end": self.t_end, "duration_s": (end - self.t_start) if end is not None else None,
                "peak_density": round(self.peak_density, 3), "peak_los": self.peak_los,
                "people_affected": len(self.people), "time_lost_s": round(self.time_lost_s, 1),
                "reason": self.reason, "active": self.active}


@dataclass
class Conflict:
    zone_id: str
    t_start: float
    t_end: float | None = None
    peak_counterflow: float = 0.0
    peak_los: str = "A"
    people: set = field(default_factory=set)

    @property
    def active(self) -> bool:
        return self.t_end is None

    def as_dict(self, now: float | None = None) -> dict:
        end = self.t_end if self.t_end is not None else now
        return {"zone_id": self.zone_id, "t_start": self.t_start, "t_end": self.t_end,
                "duration_s": (end - self.t_start) if end is not None else None,
                "peak_counterflow": round(self.peak_counterflow, 3), "peak_los": self.peak_los,
                "people_affected": len(self.people), "active": self.active}


def zone_axis(polygon) -> np.ndarray:
    """Unit vector along the long side of the zone's minimum rotated rectangle."""
    pts = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    n = np.linalg.norm(e)
    return e / n if n > 0 else np.array([1.0, 0.0])


@dataclass
class _Acc:
    frames: int = 0
    occ_sum: float = 0.0
    spd_sum: float = 0.0
    spd_n: int = 0
    entries: int = 0
    lost: float = 0.0
    ids: set = field(default_factory=set)
    speeds: list = field(default_factory=list)
    hbins: np.ndarray = field(default_factory=lambda: np.zeros(8))
    n_pos: int = 0
    n_neg: int = 0
    mover_ids: set = field(default_factory=set)


class DensityAnalyzer:
    def __init__(self, zones, cfg, on_window: Callable | None = None, on_bottleneck: Callable | None = None):
        self.cfg = cfg
        self.zones = list(zones)
        self.ids = [z.id for z in self.zones]
        self.area = {z.id: max(z.area, 1e-6) for z in self.zones}
        self.ztype = {z.id: z.type for z in self.zones}
        self.on_window, self.on_bottleneck = on_window, on_bottleneck
        self.window_idx: int | None = None
        self.acc = {z: _Acc() for z in self.ids}
        self.prev_inside: dict[str, set] = {z: set() for z in self.ids}
        self.base_samples = {z: deque(maxlen=5000) for z in self.ids}
        self.free_flow = {z: cfg.default_free_flow_mps for z in self.ids}
        self.cand_start: dict[str, float | None] = {z: None for z in self.ids}
        self.cand_acc: dict[str, Bottleneck | None] = {z: None for z in self.ids}
        self.active: dict[str, Bottleneck | None] = {z: None for z in self.ids}
        self.clear_count: dict[str, int] = {z: 0 for z in self.ids}
        self.bottlenecks: list[Bottleneck] = []
        self.windows: list[ZoneWindow] = []
        self.latest: dict[str, ZoneWindow] = {}
        self.instant: dict[str, dict] = {z: {"occupancy": 0, "los": "A", "density": 0.0} for z in self.ids}
        self.axis = {z.id: zone_axis(z.polygon) for z in self.zones}
        self.conflicts: list[Conflict] = []
        self.conflict_active: dict[str, Conflict | None] = {z: None for z in self.ids}
        self.conflict_cand: dict[str, Conflict | None] = {z: None for z in self.ids}
        self.on_conflict = None

    # ------------------------------------------------------------------ update
    def update(self, ts: float, dt: float, track_ids, speeds, inside: np.ndarray, vx=None, vy=None) -> None:
        w = int(math.floor(ts / self.cfg.window_s + 1e-9))
        if self.window_idx is None:
            self.window_idx = w
        while w > self.window_idx:
            self._close_window(self.window_idx)
            self.window_idx += 1
        speeds = np.asarray(speeds, float)
        for j, zid in enumerate(self.ids):
            col = inside[:, j] if inside.size else np.zeros(0, bool)
            n = int(col.sum())
            a = self.acc[zid]
            a.frames += 1
            a.occ_sum += n
            now_ids = {track_ids[i] for i in np.flatnonzero(col)}
            a.entries += len(now_ids - self.prev_inside[zid])
            self.prev_inside[zid] = now_ids
            app = self.area[zid] / n if n else float("inf")
            self.instant[zid] = {"occupancy": n, "density": n / self.area[zid],
                                 "los": fruin_los(app, self.cfg.fruin_los_m2_per_person)}
            if n:
                s = speeds[col]
                a.spd_sum += float(s.sum())
                a.spd_n += n
                a.ids |= now_ids
                a.speeds.extend(s.tolist())
                ff = self.free_flow[zid]
                a.lost += float(np.maximum(0.0, 1.0 - s / max(ff, 1e-3)).sum()) * dt
                if vx is not None:
                    mv = col & (speeds > self.cfg.moving_speed_mps)
                    if mv.any():
                        vxs, vys = np.asarray(vx)[mv], np.asarray(vy)[mv]
                        ang = np.arctan2(vys, vxs)
                        a.hbins += np.bincount(((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8, minlength=8)
                        proj = vxs * self.axis[zid][0] + vys * self.axis[zid][1]
                        a.n_pos += int((proj > 0).sum())
                        a.n_neg += int((proj < 0).sum())
                        a.mover_ids |= {track_ids[i] for i in np.flatnonzero(mv)}

    def _close_window(self, w: int) -> None:
        ws = w * self.cfg.window_s
        we = ws + self.cfg.window_s
        for zid in self.ids:
            a = self.acc[zid]
            occ = a.occ_sum / a.frames if a.frames else 0.0
            dens = occ / self.area[zid]
            app = self.area[zid] / occ if occ > 0 else float("inf")
            los = fruin_los(app, self.cfg.fruin_los_m2_per_person)
            mspd = (a.spd_sum / a.spd_n) if a.spd_n else None
            nm = a.n_pos + a.n_neg
            ent = cf = None
            if nm >= self.cfg.counterflow_min_samples:
                p = a.hbins / a.hbins.sum()
                p = p[p > 0]
                ent = float(-(p * np.log(p)).sum() / np.log(8))
                cf = min(a.n_pos, a.n_neg) / nm
            win = ZoneWindow(zid, ws, we, occ, dens, los, mspd, a.entries * 60.0 / self.cfg.window_s,
                             self.free_flow[zid], ent, cf)
            self.windows.append(win)
            self.latest[zid] = win
            if self.on_window:
                self.on_window(win)
            # learn free-flow speed from uncongested windows
            if los in ("A", "B") and a.speeds:
                self.base_samples[zid].extend(a.speeds)
                if len(self.base_samples[zid]) >= self.cfg.min_baseline_samples:
                    self.free_flow[zid] = max(float(np.percentile(self.base_samples[zid], 75)), 0.2)
            self._bottleneck_step(zid, win, a)
            self._conflict_step(zid, win, a)
            self.acc[zid] = _Acc()

    def _condition(self, zid: str, win: ZoneWindow) -> tuple[bool, str]:
        if self.ztype[zid] not in self.cfg.bottleneck_zone_types:
            return False, ""
        if los_rank(win.los) >= los_rank("D"):
            return True, f"density LoS {win.los}"
        crowded_enough = (self.ztype[zid] != "aisle"  # in aisles, slow alone is just browsing
                          or los_rank(win.los) >= los_rank(self.cfg.aisle_speed_rule_min_los))
        if (win.mean_speed is not None and win.occupancy >= self.cfg.min_occupancy_for_speed_rule
                and crowded_enough and win.mean_speed < self.cfg.speed_drop_ratio * self.free_flow[zid]):
            return True, f"speed {win.mean_speed:.2f} m/s vs free-flow {self.free_flow[zid]:.2f}"
        return False, ""

    def _bottleneck_step(self, zid: str, win: ZoneWindow, a: _Acc) -> None:
        cond, reason = self._condition(zid, win)
        if not cond and self.active[zid] is not None:
            # hysteresis: an active bottleneck ends only after N consecutive clear windows
            self.clear_count[zid] += 1
            if self.clear_count[zid] < self.cfg.bottleneck_end_windows:
                return
        if cond:
            self.clear_count[zid] = 0
            if self.cand_start[zid] is None:
                self.cand_start[zid] = win.window_start
                self.cand_acc[zid] = Bottleneck(zid, win.window_start, t_confirmed=-1.0)
            b = self.active[zid] or self.cand_acc[zid]
            b.peak_density = max(b.peak_density, win.density)
            if los_rank(win.los) > los_rank(b.peak_los):
                b.peak_los = win.los
            b.people |= a.ids
            b.time_lost_s += a.lost
            if win.mean_speed is not None:
                b.mean_speed_min = win.mean_speed if b.mean_speed_min is None else min(b.mean_speed_min, win.mean_speed)
            b.reason = b.reason or reason
            if self.active[zid] is None and win.window_end - self.cand_start[zid] >= self.cfg.bottleneck_min_duration_s - 1e-9:
                b.t_confirmed = win.window_end
                self.active[zid] = b
                self.bottlenecks.append(b)
                if self.on_bottleneck:
                    self.on_bottleneck("start", b)
        else:
            b = self.active[zid]
            if b is not None:
                b.t_end = win.window_end - self.clear_count[zid] * self.cfg.window_s
                self.clear_count[zid] = 0
                if self.on_bottleneck:
                    self.on_bottleneck("end", b)
            self.active[zid] = None
            self.cand_start[zid] = None
            self.cand_acc[zid] = None

    def _conflict_step(self, zid: str, win: ZoneWindow, a: _Acc) -> None:
        cond = (self.ztype[zid] in self.cfg.counterflow_zone_types and win.counterflow is not None
                and win.counterflow >= self.cfg.counterflow_min_ratio
                and los_rank(win.los) >= los_rank(self.cfg.counterflow_min_los))
        act = self.conflict_active[zid]
        if cond:
            c = act or self.conflict_cand[zid]
            if c is None:
                c = self.conflict_cand[zid] = Conflict(zid, win.window_start)
            c.peak_counterflow = max(c.peak_counterflow, win.counterflow)
            if los_rank(win.los) > los_rank(c.peak_los):
                c.peak_los = win.los
            c.people |= a.mover_ids
            if act is None and win.window_end - c.t_start >= self.cfg.counterflow_min_duration_s - 1e-9:
                self.conflict_active[zid] = c
                self.conflict_cand[zid] = None
                self.conflicts.append(c)
        else:
            self.conflict_cand[zid] = None
            if act is not None:
                act.t_end = win.window_start
                self.conflict_active[zid] = None
                if self.on_conflict:
                    self.on_conflict(act)

    def finalize(self, ts: float) -> None:
        if self.window_idx is not None:
            self._close_window(self.window_idx)
            self.window_idx += 1
        for zid, c in self.conflict_active.items():
            if c is not None:
                c.t_end = ts
                if self.on_conflict:
                    self.on_conflict(c)
                self.conflict_active[zid] = None
        for zid, b in self.active.items():
            if b is not None:
                b.t_end = ts
                if self.on_bottleneck:
                    self.on_bottleneck("end", b)
                self.active[zid] = None

    def active_bottlenecks(self) -> list[Bottleneck]:
        return [b for b in self.active.values() if b is not None]
