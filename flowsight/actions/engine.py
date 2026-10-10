"""Action engine: analytics state -> prioritised, deduplicated Insight objects ("what to do now").

Each rule yields candidates keyed by (rule, zone). The engine:
  * dedups — an active key is updated in place, never duplicated
  * auto-resolves — when a key's condition clears, the insight is resolved and its key cools down
  * cooldown — a resolved key cannot re-fire for cooldown_s (no flapping / spam)
  * caps the active list at max_active, ordered by priority
priority = severity_weight x confidence x (1 + ln(1 + people_affected))
"""
from __future__ import annotations

import math
import uuid

import numpy as np
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
SEV_W = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}


class Evidence(BaseModel):
    metrics: dict = Field(default_factory=dict)
    baseline_ratio: Optional[float] = None
    window: str = ""
    snapshot_path: Optional[str] = None


class Insight(BaseModel):
    id: str
    created_ts: float
    updated_ts: float
    type: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    severity: Severity
    title: str
    evidence: Evidence
    action: str
    owner: Literal["floor_staff", "store_manager", "merchandising", "layout_planning"]
    horizon: Literal["now", "today", "this_week"]
    expected_impact: str
    confidence: float
    status: Literal["open", "acked", "resolved", "suppressed"] = "open"
    priority: float = 0.0
    people_affected: int = 0
    resolved_ts: Optional[float] = None


def fmt_dur(s: float | None) -> str:
    if s is None or not math.isfinite(s):
        return "n/a"
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    m, r = divmod(s, 60)
    if m < 60:
        return f"{m}m{r:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _cand(type_, zone_id, severity, title, action, owner, horizon, impact, confidence, people=0,
          metrics=None, ratio=None, window=""):
    return {"type": type_, "zone_id": zone_id, "severity": severity, "title": title, "action": action,
            "owner": owner, "horizon": horizon, "expected_impact": impact,
            "confidence": float(max(0.05, min(0.99, confidence))), "people_affected": int(people),
            "evidence": {"metrics": metrics or {}, "baseline_ratio": ratio, "window": window}}


# ------------------------------------------------------------------- what-if
def what_if(attention: dict, zname: dict, ztype: dict, window_s: float, cfg) -> list[dict]:
    """Expected engaged visits if an under-performing promo/display moved from zone A to zone B.

    expected(B) = exposure(B) x stopping power; range uses the promo prior (low) and B's own
    stopping power (high); minus what A currently converts. Scaled to a store day. ESTIMATE.
    """
    day = cfg.store_open_hours * 3600.0 / max(window_s, 60.0)
    hot = [r for r in attention.values() if r.get("status") == "HOT"]
    weak = [r for r in attention.values()
            if ztype.get(r["zone_id"]) == "promo" and r.get("status") in ("IGNORED", "UNVISITED")]
    out = []
    for a in weak:
        for b in hot:
            if a["zone_id"] == b["zone_id"]:
                continue
            exp_b = b["exposures"]
            sp_b = b["stopping_power"] or 0.0
            prior = cfg.promo_stopping_power_prior
            lo = exp_b * min(prior, sp_b) - a["stops"]
            hi = exp_b * max(prior, sp_b) - a["stops"]
            out.append({
                "from_zone": a["zone_id"], "from_name": zname[a["zone_id"]],
                "to_zone": b["zone_id"], "to_name": zname[b["zone_id"]],
                "from_stopping_power": a["stopping_power"], "to_stopping_power": sp_b,
                "from_exposures": a["exposures"], "to_exposures": exp_b, "current_stops": a["stops"],
                "uplift_low_per_day": max(0.0, lo * day), "uplift_high_per_day": max(0.0, hi * day),
                "uplift_mid_per_day": max(0.0, 0.5 * (lo + hi) * day), "label": "estimate",
            })
    out.sort(key=lambda r: -r["uplift_mid_per_day"])
    return out


class ActionEngine:
    def __init__(self, cfg, hub, snapshot_fn: Callable | None = None, on_change: Callable | None = None):
        self.cfg = cfg  # AppConfig
        self.acfg = cfg.actions
        self.hub = hub
        self.snapshot_fn = snapshot_fn
        self.on_change = on_change
        self.active: dict[str, Insight] = {}
        self.all: dict[str, Insight] = {}
        self.cooldown_until: dict[str, float] = {}
        self.last_true: dict[str, float] = {}
        self.version = 0
        self.changed_ids: dict[str, int] = {}
        self._last_eval = -1e9

    # ---------------------------------------------------------------- rules
    def candidates(self, now: float) -> list[dict]:
        h, a = self.hub, self.acfg
        zname, ztype = h.zname, h.ztype
        out: list[dict] = []

        # 1. checkout queue
        target = self.cfg.queue.max_wait_target_s
        for q in h.queue_status():
            W_little, L = q["predicted_wait_s"], q["queue_len"]
            W_new, over = q.get("wait_new_arrival_s"), q.get("overflow_len", 0)
            W = max([w for w in (W_little, W_new) if w is not None], default=None)
            rising = q["rising_streak"] >= self.cfg.queue.rising_windows
            if L >= 2 and ((W is not None and W > target) or rising):
                sc = h.baseline.score(q["zone_id"], "queue_wait", W or 0.0, exclude_last=6) if W else None
                ratio = sc["ratio"] if sc and sc["ratio"] else None
                rtxt = f" ({ratio:.1f}× normal)" if ratio and ratio >= 1.2 else ""
                sev = "critical" if W and W > 1.5 * target else "high"
                why = f"predicted wait {fmt_dur(W)}{rtxt}" if W else "queue rising"
                if over:
                    why += f", queue spills {over} people into the walkway"
                L = L + over
                out.append(_cand(
                    "checkout_queue", q["zone_id"], sev, f"Checkout queue building at {q['zone_name']}",
                    f"Open another checkout lane — {why}, {L:.0f} in queue.",
                    "floor_staff", "now",
                    f"Halving the queue cuts predicted wait to ≈{fmt_dur((W or 0) / 2)} for {L:.0f} waiting shoppers",
                    0.55 + min(0.4, q["arrival_rate_pm"] / 10), people=L,
                    metrics={"predicted_wait_s": W, "little_wait_s": W_little, "queue_len": L, "overflow": over, "arrival_rate_pm": q["arrival_rate_pm"],
                             "service_rate_pm": q["service_rate_pm"], "rising_windows": q["rising_streak"],
                             "observed_wait_s": q["observed_wait_s"]},
                    ratio=ratio, window=f"last {fmt_dur(self.cfg.queue.rate_horizon_s)}"))

        # 2. active bottlenecks
        for b in h.density.active_bottlenecks():
            zt = ztype[b.zone_id]
            dur = now - b.t_start
            sev = "critical" if b.peak_los in ("E", "F") or b.time_lost_s > 120 else ("high" if zt != "aisle" else "medium")
            verb = "Clear" if zt in ("walkway", "entrance") else "Check"
            out.append(_cand(
                "bottleneck", b.zone_id, sev, f"Congestion in {zname[b.zone_id]}",
                f"{verb} {zname[b.zone_id]} — "
                + (f"density LoS {b.peak_los}" if b.peak_los in ("D", "E", "F") or b.mean_speed_min is None
                   else f"flow slowed to {b.mean_speed_min:.1f} m/s (LoS {b.peak_los})")
                + f" for {fmt_dur(dur)}, "
                f"{len(b.people)} shoppers slowed, {b.time_lost_s:.0f} s lost.",
                "floor_staff", "now",
                f"Restoring free flow saves ≈{b.time_lost_s / max(dur, 1) * 60:.0f} shopper-seconds per minute",
                0.6 + min(0.35, dur / 300), people=len(b.people),
                metrics={"peak_los": b.peak_los, "peak_density": round(b.peak_density, 2), "duration_s": round(dur),
                         "people_affected": len(b.people), "time_lost_s": round(b.time_lost_s),
                         "reason": b.reason},
                window=f"since {fmt_dur(b.t_start)}"))

        # 2b. counter-flow conflicts (two-way streams in a crowded walkway)
        recent_cf: dict[str, int] = {}
        for c in h.density.conflicts:
            if c.t_start >= now - 3600:
                recent_cf[c.zone_id] = recent_cf.get(c.zone_id, 0) + 1
        for c in [c for c in h.density.conflict_active.values() if c is not None]:
            dur = now - c.t_start
            n_hr = recent_cf.get(c.zone_id, 1)
            if dur < 60 and n_hr < 2:  # layout advice only for sustained or recurring conflicts
                continue
            out.append(_cand(
                "counter_flow", c.zone_id, "medium", f"Two-way flow conflict in {zname[c.zone_id]}",
                f"Two-way conflict in {zname[c.zone_id]}: {c.peak_counterflow:.0%} of walkers move against the main "
                f"flow at LoS {c.peak_los} for {fmt_dur(dur)}" + (f" ({n_hr}× this hour)" if n_hr >= 2 else "")
                + ". Add one-way guidance or a centre divider.",
                "layout_planning", "this_week",
                "Separating opposing streams restores walking speed and reduces bumps", 0.5 + min(0.4, dur / 300),
                people=len(c.people), metrics={"counterflow": round(c.peak_counterflow, 2), "peak_los": c.peak_los,
                                               "duration_s": round(dur), "people_affected": len(c.people)},
                window=f"since {fmt_dur(c.t_start)}"))

        # 3. recurring bottlenecks
        recent: dict[str, list] = {}
        for b in h.density.bottlenecks:
            if b.t_start >= now - 3600:
                recent.setdefault(b.zone_id, []).append(b)
        for zid, bs in recent.items():
            if len(bs) >= a.recurring_bottleneck_count:
                lost = sum(b.time_lost_s for b in bs)
                out.append(_cand(
                    "recurring_bottleneck", zid, "medium", f"Recurring congestion in {zname[zid]}",
                    f"Widen or reposition the display obstructing {zname[zid]} — recurs {len(bs)}×/hr at peak.",
                    "layout_planning", "this_week",
                    f"Removes ≈{lost / 60:.0f} shopper-minutes of delay per peak hour",
                    0.5 + 0.1 * len(bs), people=len(set().union(*[b.people for b in bs])),
                    metrics={"count_last_hour": len(bs), "time_lost_s": round(lost)}, window="last 60 min"))

        # 4-6. attention-based
        att = h.attention_latest or {}
        win_min = self.cfg.attention.window_min
        for zid, r in att.items():
            st = r.get("status")
            if st == "IGNORED":
                out.append(_cand(
                    "ignored_zone", zid, "medium", f"{zname[zid]} is ignored",
                    f"{zname[zid]}: {r['exposures']} shoppers passed, only {r['stopping_power']:.0%} stopped. "
                    f"Refresh signage / swap product.",
                    "merchandising", "today",
                    f"Lifting stopping power to {self.acfg.promo_stopping_power_prior:.0%} ≈ "
                    f"+{max(0, r['exposures'] * self.acfg.promo_stopping_power_prior - r['stops']):.0f} stops per {win_min:.0f} min",
                    0.4 + min(0.5, r["exposures"] / 200), people=r["exposures"],
                    metrics={k: r[k] for k in ("exposures", "entries", "stops", "stopping_power")},
                    window=f"last {win_min:.0f} min"))
            elif st == "UNVISITED":
                share = r.get("traffic_share") or 0
                out.append(_cand(
                    "unvisited_zone", zid, "low" if ztype[zid] != "promo" else "medium", f"{zname[zid]} is a dead zone",
                    f"{zname[zid]} receives {share:.0%} of median traffic. Add wayfinding or move a traffic-driver product there.",
                    "layout_planning", "this_week",
                    f"Bringing it to median traffic ≈ +{max(0, r.get('median_exposure', 0) - r['exposures']):.0f} exposures per {win_min:.0f} min",
                    0.45 + min(0.45, (r.get("median_exposure") or 0) / 150), people=0,
                    metrics={"exposures": r["exposures"], "median_exposure": r.get("median_exposure"),
                             "traffic_share": share}, window=f"last {win_min:.0f} min"))
        for w in what_if(att, zname, ztype, win_min * 60.0, self.acfg)[:2]:
            out.append(_cand(
                "what_if", w["from_zone"], "medium", f"What-if: move promo to {w['to_name']}",
                f"What-if: moving the promo from {w['from_name']} to {w['to_name']} (stopping power "
                f"{w['to_stopping_power']:.0%}) ≈ +{w['uplift_mid_per_day']:.0f} engaged visits/day "
                f"(range {w['uplift_low_per_day']:.0f}–{w['uplift_high_per_day']:.0f}, estimate).",
                "merchandising", "this_week",
                f"+{w['uplift_low_per_day']:.0f}–{w['uplift_high_per_day']:.0f} engaged visits/day (estimate)",
                0.45, metrics=w, window=f"last {win_min:.0f} min"))

        # 7. occupancy anomaly
        n_s = max(1, a.anomaly_smooth_windows)
        for zid, wnd in h.density.latest.items():
            if ztype[zid] == "checkout":
                continue
            hist = h.baseline.data.get((zid, "occupancy"))
            occ = float(np.mean(list(hist)[-n_s:])) if hist else wnd.occupancy  # 30 s smoothed
            sc = h.baseline.score(zid, "occupancy", occ, exclude_last=n_s + 6)
            if sc and sc["z"] > a.anomaly_z and (sc["ratio"] or 0) >= a.anomaly_min_ratio \
                    and occ >= a.anomaly_min_occupancy:
                out.append(_cand(
                    "occupancy_anomaly", zid, "high", f"Unusual crowd at {zname[zid]}",
                    f"Unusual crowd at {zname[zid]} — {sc['ratio']:.1f}× normal. Send a staff member.",
                    "floor_staff", "now", "Staff presence prevents spill-over congestion and lost sales",
                    min(0.95, 0.5 + sc["z"] / 20), people=round(occ),
                    metrics={"occupancy": round(occ, 1), "normal": round(sc["mean"], 2), "z": round(sc["z"], 1)},
                    ratio=sc["ratio"], window=f"{self.cfg.density.window_s:.0f}s window"))

        # 8. long stationary shopper in service / aisle zone
        best: dict[str, tuple[float, int]] = {}
        for p in h.people():
            if p["stationary_s"] < a.long_stationary_s:
                continue
            for zid in h.zones.zones_at(p["x"], p["y"]):
                if ztype[zid] in ("service", "aisle") and p["stationary_s"] > best.get(zid, (0, 0))[0]:
                    best[zid] = (p["stationary_s"], p["id"])
        for zid, (secs, tid) in best.items():
            verb = "waiting at" if ztype[zid] == "service" else "stationary in"
            out.append(_cand(
                "assist_shopper", zid, "medium" if ztype[zid] == "service" else "low",
                f"Shopper may need help at {zname[zid]}",
                f"Shopper {verb} {zname[zid]} for {fmt_dur(secs)} — assist.",
                "floor_staff", "now", "Assisted shoppers convert more and wait less", 0.7, people=1,
                metrics={"stationary_s": round(secs)}, window="live"))
        return out

    # --------------------------------------------------------------- engine
    def evaluate(self, now: float, force: bool = False) -> list[Insight]:
        if not force and now - self._last_eval < self.acfg.eval_every_s:
            return []
        self._last_eval = now
        changed: list[Insight] = []
        cands = self.candidates(now)
        seen = set()
        for c in sorted(cands, key=lambda c: -SEV_W[c["severity"]] * c["confidence"]):
            key = f"{c['type']}:{c['zone_id']}"
            if key in seen:
                continue
            seen.add(key)
            self.last_true[key] = now
            prio = SEV_W[c["severity"]] * c["confidence"] * (1 + math.log1p(c["people_affected"]))
            if key in self.active:
                ins = self.active[key]
                ins.title, ins.action, ins.expected_impact = c["title"], c["action"], c["expected_impact"]
                ins.severity, ins.confidence, ins.priority = c["severity"], c["confidence"], prio
                ins.people_affected = c["people_affected"]
                snap = ins.evidence.snapshot_path
                ins.evidence = Evidence(**c["evidence"], snapshot_path=snap)
                ins.updated_ts = now
                changed.append(ins)
                continue
            if now < self.cooldown_until.get(key, -1e18):
                continue
            ins = Insight(id=uuid.uuid4().hex[:10], created_ts=now, updated_ts=now, type=c["type"],
                          zone_id=c["zone_id"], zone_name=self.hub.zname.get(c["zone_id"]), severity=c["severity"],
                          title=c["title"], evidence=Evidence(**c["evidence"]), action=c["action"], owner=c["owner"],
                          horizon=c["horizon"], expected_impact=c["expected_impact"], confidence=c["confidence"],
                          priority=prio, people_affected=c["people_affected"])
            if self.snapshot_fn is not None:
                try:
                    ins.evidence.snapshot_path = self.snapshot_fn(ins)
                except Exception:  # snapshots are best-effort evidence
                    pass
            self.active[key] = ins
            self.all[ins.id] = ins
            changed.append(ins)

        grace = self.acfg.resolve_grace_s  # hysteresis: condition must stay clear before resolving
        for key in [k for k in self.active if k not in seen and now - self.last_true.get(k, -1e18) >= grace]:
            ins = self.active.pop(key)
            ins.status, ins.resolved_ts, ins.updated_ts = "resolved", now, now
            self.cooldown_until[key] = now + self.acfg.cooldown_s
            changed.append(ins)

        # cap active list
        if len(self.active) > self.acfg.max_active:
            ranked = sorted(self.active.items(), key=lambda kv: -kv[1].priority)
            for key, ins in ranked[self.acfg.max_active:]:
                ins.status, ins.updated_ts = "suppressed", now
                self.active.pop(key)
                self.cooldown_until[key] = now + self.acfg.cooldown_s
                changed.append(ins)

        if changed:
            self.version += 1
            for ins in changed:
                self.changed_ids[ins.id] = self.version
            if self.on_change:
                self.on_change(changed)
        return changed

    def ack(self, insight_id: str) -> Insight | None:
        ins = self.all.get(insight_id)
        if ins is None:
            return None
        if ins.status == "open":
            ins.status = "acked"
            self.version += 1
            self.changed_ids[ins.id] = self.version
            if self.on_change:
                self.on_change([ins])
        return ins

    def active_sorted(self) -> list[Insight]:
        return sorted(self.active.values(), key=lambda i: -i.priority)

    def history(self) -> list[Insight]:
        return sorted(self.all.values(), key=lambda i: -i.created_ts)

    def delta_since(self, version: int) -> list[dict]:
        return [self.all[i].model_dump() for i, v in self.changed_ids.items() if v > version and i in self.all]
