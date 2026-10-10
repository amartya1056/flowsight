"""Attention funnel & dead-zone classification — the key differentiator.

Funnel per zone over a rolling window, counted in UNIQUE SHOPPERS (tracks):  exposures -> entries -> stops.
  * exposure  = a track passed through the zone or its exposure buffer (could have seen it)
  * entry     = a track with a confirmed visit to the zone
  * stop      = a track with a visit of >= stop_min_s engaged (slow) time
  * stopping power = stopping shoppers / exposed shoppers
Counting shoppers rather than passes keeps zones next to busy walkways (passed 4-9x per shopper at peak) comparable
with destination zones off the main path.

Classification (merchandisable zones only: aisle / promo / service):
  * UNVISITED — exposure < unvisited_traffic_share x median exposure: nobody even walks past.
                That is a LAYOUT / WAYFINDING problem.
  * IGNORED   — normal exposure but stopping power < ignored_stopping_power: people walk past and
                don't stop. That is a MERCHANDISING / DISPLAY problem.
  * HOT       — top-decile stopping power: an opportunity to place promotions nearby.
"""
from __future__ import annotations

import numpy as np

UNVISITED, IGNORED, HOT, NORMAL, NA = "UNVISITED", "IGNORED", "HOT", "NORMAL", "-"


class AttentionAnalyzer:
    def __init__(self, zones, cfg):
        self.cfg = cfg
        self.zones = list(zones)
        self.merch = {z.id for z in self.zones if z.type in cfg.merch_zone_types}

    def compute(self, now: float, exposures, entries, visits, open_visits, window_s: float | None = None) -> dict:
        window_s = self.cfg.window_min * 60.0 if window_s is None else window_s
        t0 = now - window_s
        res = {z.id: {"zone_id": z.id, "exposures": 0, "entries": 0, "stops": 0, "engaged_s": 0.0}
               for z in self.zones}
        exp_ids = {z.id: set() for z in self.zones}
        ent_ids = {z.id: set() for z in self.zones}
        stop_ids = {z.id: set() for z in self.zones}
        for t, tid, zid in exposures:
            if t >= t0:
                exp_ids[zid].add(tid)
        for t, tid, zid in entries:
            if t >= t0:
                ent_ids[zid].add(tid)
        for v in list(visits) + list(open_visits):
            if v.t_enter >= t0:
                res[v.zone_id]["engaged_s"] += v.engaged_s
                if v.is_stop:
                    stop_ids[v.zone_id].add(v.track_id)
        for zid, r in res.items():
            r["exposures"] = len(exp_ids[zid] | ent_ids[zid])
            r["entries"] = len(ent_ids[zid])
            r["stops"] = len(stop_ids[zid])

        for r in res.values():
            e = r["exposures"]
            r["stopping_power"] = (r["stops"] / e) if e else None
            r["entry_rate"] = (r["entries"] / e) if e else None
            r["status"] = NA if r["zone_id"] not in self.merch else NORMAL
            r["reason"] = ""

        merch = [res[z] for z in self.merch]
        if not merch:
            return res
        med = float(np.median([r["exposures"] for r in merch]))
        r_ok = [r for r in merch if r["exposures"] >= self.cfg.min_exposures]
        sps = [r["stopping_power"] for r in r_ok if r["stopping_power"] is not None]
        hot_thr = float(np.quantile(sps, self.cfg.hot_quantile)) if len(sps) >= 3 else None
        for r in merch:
            r["median_exposure"] = med
            r["traffic_share"] = (r["exposures"] / med) if med > 0 else None
            if med >= self.cfg.min_exposures and r["exposures"] < self.cfg.unvisited_traffic_share * med:
                r["status"] = UNVISITED
                r["reason"] = (f"{r['exposures']} exposures = {r['exposures'] / med:.0%} of the median zone "
                               f"({med:.0f}) — shoppers do not walk past")
            elif r["exposures"] >= self.cfg.min_exposures and r["stopping_power"] is not None \
                    and r["stopping_power"] < self.cfg.ignored_stopping_power:
                r["status"] = IGNORED
                r["reason"] = (f"{r['exposures']} shoppers passed, only {r['stopping_power']:.0%} stopped")
            elif hot_thr is not None and r["exposures"] >= self.cfg.min_exposures \
                    and r["stopping_power"] is not None and r["stopping_power"] >= max(hot_thr, self.cfg.hot_min_stopping_power):
                r["status"] = HOT
                r["reason"] = f"stopping power {r['stopping_power']:.0%} (top decile)"
        return res
