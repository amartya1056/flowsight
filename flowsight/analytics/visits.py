"""Zone visits & dwell — a debounced per-(track, zone) state machine.

States: OUTSIDE -> ENTERING -> INSIDE -> EXITING -> OUTSIDE
  * ENTERING becomes INSIDE (a confirmed *visit*) once the track has been in the zone for enter_min_s
    (measured from the first inside sample; short excursions out of the zone within the grace period
    don't reset it).
  * A visit ends only after the track has been outside for exit_grace_s. This debounces boundary
    jitter: someone standing on a zone edge does not generate dozens of visits.
  * Entry time = first inside timestamp; exit time = last inside timestamp; dwell = exit - entry.
  * engaged_s = time inside with speed < engaged_speed_mps; is_stop = engaged_s >= stop_min_s.

The same machine (with enter_min_s = 0, no engagement) is reused for *exposure* passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

OUTSIDE, ENTERING, INSIDE, EXITING = "OUTSIDE", "ENTERING", "INSIDE", "EXITING"


@dataclass
class Visit:
    track_id: int
    zone_id: str
    t_enter: float
    t_exit: float
    dwell_s: float
    engaged_s: float
    is_stop: bool
    mean_speed: float
    open: bool = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class _Presence:
    first_in: float
    last_in: float
    confirmed: bool = False
    engaged: float = 0.0
    speed_sum: float = 0.0
    n: int = 0
    last_inside_sample: bool = True

    def state(self) -> str:
        if not self.confirmed:
            return ENTERING
        return INSIDE if self.last_inside_sample else EXITING


class ZonePresence:
    """Debounced presence of tracks in zones. Generic: used for visits and for exposures."""

    def __init__(self, zone_ids: list[str], enter_min_s: float, exit_grace_s: float,
                 engaged_speed: float | None = None, stop_min_s: float = 3.0,
                 on_open: Callable | None = None, on_close: Callable | None = None):
        self.zone_ids = list(zone_ids)
        self._zidx = {z: i for i, z in enumerate(self.zone_ids)}
        self.enter_min_s = enter_min_s
        self.exit_grace_s = exit_grace_s
        self.engaged_speed = engaged_speed
        self.stop_min_s = stop_min_s
        self.on_open = on_open
        self.on_close = on_close
        self.states: dict[tuple[int, str], _Presence] = {}
        self._by_track: dict[int, set[str]] = {}
        self._last_ts: dict[int, float] = {}

    # ------------------------------------------------------------------ update
    def update(self, ts: float, track_ids, speeds, inside: np.ndarray) -> None:
        """inside: bool [n_tracks, n_zones] aligned with self.zone_ids."""
        for i, tid in enumerate(track_ids):
            prev = self._last_ts.get(tid)
            dt = 0.0 if prev is None else min(max(ts - prev, 0.0), 1.0)
            self._last_ts[tid] = ts
            spd = float(speeds[i])
            row = inside[i]
            active = self._by_track.get(tid)
            cand = set(np.flatnonzero(row).tolist())
            if active:
                cand |= {self._zidx[z] for z in active}
            for j in cand:
                self._sample(tid, self.zone_ids[j], ts, dt, spd, bool(row[j]))

    def _sample(self, tid: int, zid: str, ts: float, dt: float, spd: float, is_in: bool) -> None:
        key = (tid, zid)
        st = self.states.get(key)
        if is_in:
            if st is None:
                st = _Presence(first_in=ts, last_in=ts)
                self.states[key] = st
                self._by_track.setdefault(tid, set()).add(zid)
            elif self.engaged_speed is not None and spd < self.engaged_speed:
                st.engaged += dt
            st.last_in = ts
            st.last_inside_sample = True
            st.speed_sum += spd
            st.n += 1
            if not st.confirmed and st.last_in - st.first_in >= self.enter_min_s - 1e-9:
                st.confirmed = True
                if self.on_open:
                    self.on_open(tid, zid, st.first_in)
        elif st is not None:
            st.last_inside_sample = False
            if ts - st.last_in >= self.exit_grace_s - 1e-9:
                self._close(key)

    def _close(self, key) -> Visit | None:
        st = self.states.pop(key)
        tid, zid = key
        s = self._by_track.get(tid)
        if s:
            s.discard(zid)
            if not s:
                self._by_track.pop(tid, None)
        if not st.confirmed:
            return None
        v = self._as_visit(tid, zid, st, open_=False)
        if self.on_close:
            self.on_close(v)
        return v

    def _as_visit(self, tid, zid, st: _Presence, open_: bool) -> Visit:
        dwell = st.last_in - st.first_in
        eng = min(st.engaged, dwell) if self.engaged_speed is not None else 0.0
        return Visit(tid, zid, st.first_in, st.last_in, dwell, eng, eng >= self.stop_min_s - 1e-9,
                     st.speed_sum / max(st.n, 1), open=open_)

    # ------------------------------------------------------------- lifecycle
    def close_track(self, tid: int) -> None:
        for zid in list(self._by_track.get(tid, ())):
            self._close((tid, zid))
        self._last_ts.pop(tid, None)

    def finalize(self) -> None:
        for key in list(self.states):
            self._close(key)

    def forget_track(self, tid: int) -> None:
        """Drop all state of a track without emitting visits (used when a track turns out to be staff)."""
        for zid in list(self._by_track.get(tid, ())):
            self.states.pop((tid, zid), None)
        self._by_track.pop(tid, None)
        self._last_ts.pop(tid, None)

    def open_items(self) -> list[Visit]:
        return [self._as_visit(t, z, st, open_=True) for (t, z), st in self.states.items() if st.confirmed]

    def state_of(self, tid: int, zid: str) -> str:
        st = self.states.get((tid, zid))
        return OUTSIDE if st is None else st.state()


class VisitTracker:
    """Visits + exposures + per-zone KPIs."""

    def __init__(self, zone_ids: list[str], cfg, on_visit: Callable | None = None,
                 on_entry: Callable | None = None, on_exposure: Callable | None = None):
        self.zone_ids = list(zone_ids)
        self.cfg = cfg
        self.visits: list[Visit] = []
        self.entries: list[tuple[float, int, str]] = []  # (t_enter, track, zone)
        self.exposures: list[tuple[float, int, str]] = []
        self._on_visit, self._on_entry, self._on_exposure = on_visit, on_entry, on_exposure
        self.presence = ZonePresence(zone_ids, cfg.enter_min_s, cfg.exit_grace_s, cfg.engaged_speed_mps,
                                     cfg.stop_min_s, on_open=self._opened, on_close=self._closed)
        self.exposure = ZonePresence(zone_ids, 0.0, cfg.exit_grace_s, on_open=self._exposed)

    def _opened(self, tid, zid, t):
        self.entries.append((t, tid, zid))
        if self._on_entry:
            self._on_entry(tid, zid, t)

    def _closed(self, v: Visit):
        self.visits.append(v)
        if self._on_visit:
            self._on_visit(v)

    def _exposed(self, tid, zid, t):
        self.exposures.append((t, tid, zid))
        if self._on_exposure:
            self._on_exposure(tid, zid, t)

    def update(self, ts, track_ids, speeds, inside, exposed) -> None:
        self.presence.update(ts, track_ids, speeds, inside)
        self.exposure.update(ts, track_ids, speeds, exposed)

    def close_track(self, tid: int) -> None:
        self.presence.close_track(tid)
        self.exposure.close_track(tid)

    def finalize(self) -> None:
        self.presence.finalize()
        self.exposure.finalize()

    def purge_track(self, tid: int) -> int:
        """Retroactively remove every visit/entry/exposure of a track. Returns the number of visits removed."""
        n = sum(1 for v in self.visits if v.track_id == tid)
        self.visits = [v for v in self.visits if v.track_id != tid]
        self.entries = [e for e in self.entries if e[1] != tid]
        self.exposures = [e for e in self.exposures if e[1] != tid]
        self.presence.forget_track(tid)
        self.exposure.forget_track(tid)
        return n

    def open_visits(self) -> list[Visit]:
        return self.presence.open_items()

    def zone_kpis(self, occupancy: dict[str, int] | None = None, since: float | None = None) -> dict[str, dict]:
        occupancy = occupancy or {}
        opened = self.open_visits()
        out = {}
        for zid in self.zone_ids:
            done = [v for v in self.visits if v.zone_id == zid and (since is None or v.t_enter >= since)]
            live = [v for v in opened if v.zone_id == zid and (since is None or v.t_enter >= since)]
            d = np.array([v.dwell_s for v in done]) if done else np.array([])
            out[zid] = {
                "visits": len(done) + len(live),
                "completed_visits": len(done),
                "unique_visitors": len({v.track_id for v in done + live}),
                "median_dwell_s": float(np.median(d)) if d.size else 0.0,
                "mean_dwell_s": float(d.mean()) if d.size else 0.0,
                "p90_dwell_s": float(np.percentile(d, 90)) if d.size else 0.0,
                "total_engaged_s": float(sum(v.engaged_s for v in done + live)),
                "stops": int(sum(v.is_stop for v in done + live)),
                "occupancy": int(occupancy.get(zid, 0)),
                "live_dwell_s": [round(v.dwell_s, 1) for v in live],
            }
        return out
