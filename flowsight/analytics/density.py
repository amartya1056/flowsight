"""Density, Fruin Level-of-Service, free-flow speed baselines and bottleneck detection.

Per zone and per `window_s` window we compute: mean occupancy, density (people/m²), area per person ->
Fruin LoS A..F, people-weighted mean speed, and flow (zone entries per minute).

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
class _Acc:
    frames: int = 0
    occ_sum: float = 0.0
    spd_sum: float = 0.0
    spd_n: int = 0
    entries: int = 0
    lost: float = 0.0
    ids: set = field(default_factory=set)
    speeds: list = field(default_factory=list)


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
        self.bottlenecks: list[Bottleneck] = []
        self.windows: list[ZoneWindow] = []
        self.latest: dict[str, ZoneWindow] = {}
        self.instant: dict[str, dict] = {z: {"occupancy": 0, "los": "A", "density": 0.0} for z in self.ids}

    # ------------------------------------------------------------------ update
    def update(self, ts: float, dt: float, track_ids, speeds, inside: np.ndarray) -> None:
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
            win = ZoneWindow(zid, ws, we, occ, dens, los, mspd, a.entries * 60.0 / self.cfg.window_s,
                             self.free_flow[zid])
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
            self.acc[zid] = _Acc()

    def _condition(self, zid: str, win: ZoneWindow) -> tuple[bool, str]:
        if self.ztype[zid] not in self.cfg.bottleneck_zone_types:
            return False, ""
        if los_rank(win.los) >= los_rank("D"):
            return True, f"density LoS {win.los}"
        if (win.mean_speed is not None and win.occupancy >= self.cfg.min_occupancy_for_speed_rule
                and win.mean_speed < self.cfg.speed_drop_ratio * self.free_flow[zid]):
            return True, f"speed {win.mean_speed:.2f} m/s vs free-flow {self.free_flow[zid]:.2f}"
        return False, ""

    def _bottleneck_step(self, zid: str, win: ZoneWindow, a: _Acc) -> None:
        cond, reason = self._condition(zid, win)
        if cond:
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
                b.t_end = win.window_start
                if self.on_bottleneck:
                    self.on_bottleneck("end", b)
            self.active[zid] = None
            self.cand_start[zid] = None
            self.cand_acc[zid] = None

    def finalize(self, ts: float) -> None:
        if self.window_idx is not None:
            self._close_window(self.window_idx)
            self.window_idx += 1
        for zid, b in self.active.items():
            if b is not None:
                b.t_end = ts
                if self.on_bottleneck:
                    self.on_bottleneck("end", b)
                self.active[zid] = None

    def active_bottlenecks(self) -> list[Bottleneck]:
        return [b for b in self.active.values() if b is not None]
