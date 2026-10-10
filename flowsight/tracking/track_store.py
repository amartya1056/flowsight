"""Per-track state: smoothing, speed/heading, maturity, and ID stitching.

Coordinates are whatever the caller feeds in — floor metres after calibration (normal case) or image
pixels when uncalibrated. Stable IDs are assigned here; the raw tracker IDs are never exposed.

ID stitching: when a raw tracker ID appears that we have never seen, we look for a *lost* track
(not observed this frame, last seen <= stitch_max_gap_s ago) whose constant-velocity prediction is
within stitch_max_dist of the new point and whose heading is compatible. If found, the new raw ID is
mapped onto the old stable ID, so dwell/visits continue instead of being split into two people.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from flowsight.config import TrackingCfg


@dataclass
class TrackState:
    id: int
    ts: float
    x: float
    y: float
    speed: float
    heading: float
    vx: float
    vy: float
    age: float
    mature: bool
    truncated: bool = False
    path_len: float = 0.0


@dataclass
class Track:
    id: int
    first_ts: float
    last_ts: float
    sx: float
    sy: float
    hist: deque = field(default_factory=lambda: deque(maxlen=64))
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    path_len: float = 0.0
    raw_ids: list = field(default_factory=list)
    truncated: bool = False

    @property
    def age(self) -> float:
        return self.last_ts - self.first_ts


class TrackStore:
    def __init__(self, cfg: TrackingCfg, max_stitch_dist: float | None = None, id_offset: int = 0,
                 metric: bool = True):
        self.cfg = cfg
        self.alpha = cfg.ema_alpha
        self.max_dist = cfg.stitch_max_dist_m if max_stitch_dist is None else max_stitch_dist
        self.tracks: dict[int, Track] = {}
        self.raw_to_id: dict = {}
        self._next_id = 1 + id_offset
        self.n_stitches = 0
        self.n_tracks_created = 0
        self._expired: list[Track] = []
        # |v| from noisy positions is biased upwards (a standing person "moves" ~0.13 m/s). Noise of the EMA-smoothed
        # position per axis: s = sigma*sqrt(a/(2-a)); displacement over the speed window ~ sqrt(2)*s; per-axis speed
        # noise sv = sqrt(2)*s/window, so E|v_noise|^2 = 2*sv^2. We subtract that energy (scale is metres only).
        s = cfg.position_noise_m * math.sqrt(self.alpha / (2 - self.alpha))
        sv = math.sqrt(2) * s / max(cfg.speed_window_s, 1e-3)
        self._noise_v2 = 2 * sv * sv if metric else 0.0

    # ------------------------------------------------------------------ main API
    def update(self, ts: float, observations: list[tuple]) -> list[TrackState]:
        """observations: iterable of (raw_id, x, y) or (raw_id, x, y, truncated). Returns states updated now."""
        seen_now: set[int] = set()
        out: list[TrackState] = []
        new_obs = []
        for ob in observations:
            raw_id, x, y = ob[0], float(ob[1]), float(ob[2])
            trunc = bool(ob[3]) if len(ob) > 3 else False
            tid = self.raw_to_id.get(raw_id)
            if tid is not None and tid in self.tracks and tid not in seen_now:
                seen_now.add(tid)
                out.append(self._step(self.tracks[tid], ts, x, y, trunc))
            else:
                new_obs.append((raw_id, x, y, trunc))

        for raw_id, x, y, trunc in new_obs:
            tid = self._find_stitch(ts, x, y, exclude=seen_now)
            if tid is not None:
                tr = self.tracks[tid]
                # retire old raw ids of this track so they cannot double-map
                for r in tr.raw_ids:
                    self.raw_to_id.pop(r, None)
                tr.raw_ids = [raw_id]
                self.raw_to_id[raw_id] = tid
                self.n_stitches += 1
            else:
                tid = self._next_id
                self._next_id += 1
                self.n_tracks_created += 1
                tr = Track(id=tid, first_ts=ts, last_ts=ts, sx=x, sy=y, raw_ids=[raw_id])
                tr.hist.append((ts, x, y))
                self.tracks[tid] = tr
                self.raw_to_id[raw_id] = tid
                seen_now.add(tid)
                out.append(self._state(tr, trunc))
                continue
            seen_now.add(tid)
            out.append(self._step(tr, ts, x, y, trunc))

        # expire tracks that have been lost for longer than the stitch window
        for tid in [t for t, tr in self.tracks.items() if ts - tr.last_ts > self.cfg.stitch_max_gap_s]:
            tr = self.tracks.pop(tid)
            for r in tr.raw_ids:
                if self.raw_to_id.get(r) == tid:
                    self.raw_to_id.pop(r, None)
            self._expired.append(tr)
        return out

    def pop_expired(self) -> list[Track]:
        ex, self._expired = self._expired, []
        return ex

    def flush_all(self) -> list[Track]:
        """End of stream: expire everything."""
        ex = list(self.tracks.values()) + self.pop_expired()
        self.tracks.clear()
        self.raw_to_id.clear()
        return ex

    # ---------------------------------------------------------------- internals
    def _find_stitch(self, ts: float, x: float, y: float, exclude: set[int]) -> int | None:
        best, best_d = None, self.max_dist
        for tid, tr in self.tracks.items():
            if tid in exclude:
                continue
            gap = ts - tr.last_ts
            if gap <= 0 or gap > self.cfg.stitch_max_gap_s:
                continue
            g = min(gap, 1.0)  # don't extrapolate a constant velocity too far
            px, py = tr.sx + tr.vx * g, tr.sy + tr.vy * g
            d = math.hypot(x - px, y - py)
            if d > best_d:
                continue
            # heading compatibility: if the old track was clearly moving, the new point must not be behind it
            dx, dy = x - tr.sx, y - tr.sy
            disp = math.hypot(dx, dy)
            if tr.speed > 0.3 and disp > 0.3 * self.max_dist:
                cosang = (dx * math.cos(tr.heading) + dy * math.sin(tr.heading)) / disp
                if cosang < -0.2:
                    continue
            best, best_d = tid, d
        return best

    def _step(self, tr: Track, ts: float, x: float, y: float, trunc: bool) -> TrackState:
        a = self.alpha
        nsx = a * x + (1 - a) * tr.sx
        nsy = a * y + (1 - a) * tr.sy
        tr.path_len += math.hypot(nsx - tr.sx, nsy - tr.sy)
        tr.sx, tr.sy = nsx, nsy
        tr.last_ts = ts
        tr.truncated = trunc
        tr.hist.append((ts, nsx, nsy))
        # velocity = displacement of the smoothed position over ~speed_window_s
        w = self.cfg.speed_window_s
        ref = tr.hist[0]
        for h in tr.hist:
            if ts - h[0] <= w + 1e-9:
                ref = h
                break
        dt = ts - ref[0]
        if dt > 1e-6:
            tr.vx, tr.vy = (nsx - ref[1]) / dt, (nsy - ref[2]) / dt
            raw2 = tr.vx * tr.vx + tr.vy * tr.vy
            tr.speed = math.sqrt(max(0.0, raw2 - self._noise_v2))  # de-biased speed magnitude
            if tr.speed > 1e-3:
                tr.heading = math.atan2(tr.vy, tr.vx)
        return self._state(tr, trunc)

    def _state(self, tr: Track, trunc: bool) -> TrackState:
        return TrackState(id=tr.id, ts=tr.last_ts, x=tr.sx, y=tr.sy, speed=tr.speed, heading=tr.heading,
                          vx=tr.vx, vy=tr.vy, age=tr.age, mature=tr.age >= self.cfg.min_track_age_s,
                          truncated=trunc, path_len=tr.path_len)
