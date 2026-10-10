"""Analytics accuracy vs simulator ground truth.

For each (scenario, seed) we run the simulator once and feed two analytics hubs in lock-step:
  * ORACLE   — perfect data: true positions, true speeds, true identities (no tracker in the loop)
  * MEASURED — what FlowSight sees in production: σ=0.15 m position noise, random dropouts and ID
               breaks, passed through the real TrackStore (EMA smoothing + ID stitching)
Ground-truth visits / dwell / bottlenecks / checkout times are the oracle's output under the exact
metric definitions, so the comparison isolates measurement robustness. Zone classification is scored
against the *scenario intent* (e.g. ignored_promo => the promo must be IGNORED).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from flowsight.config import ROOT, load_config, load_store

EXPECTED = {  # scenario -> {zone: expected status}; every other merch zone must be neither IGNORED nor UNVISITED
    "ignored_promo": {"z_promo1": "IGNORED"},
    "dead_aisle": {"z_aisle7": "UNVISITED"},
}


def _match_intervals(gt: list, pred: list, key_gt, key_pred, tol: float = 0.0):
    """Greedy one-to-one matching on (key, overlapping interval). Items: (key, t0, t1, payload)."""
    by_key = defaultdict(list)
    for j, p in enumerate(pred):
        by_key[key_pred(p)].append(j)
    used, pairs = set(), []
    for i, g in enumerate(gt):
        best, best_ov = None, 0.0
        for j in by_key.get(key_gt(g), []):
            if j in used:
                continue
            p = pred[j]
            ov = min(g[2], p[2]) - max(g[1], p[1]) + tol
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def run_one(scenario: str, seed: int, duration: float) -> dict:
    from flowsight.analytics.hub import AnalyticsHub
    from flowsight.sim.simulator import Simulator
    from flowsight.tracking.track_store import TrackStore

    cfg, store = load_config(), load_store("demo_store")
    duration = float((store.sim.get("scenarios", {}).get(scenario) or {}).get("duration_s", duration))
    sim = Simulator(store, cfg, scenario=scenario, seed=seed, duration_s=duration)
    meas = AnalyticsHub(cfg, store)
    orac = AnalyticsHub(cfg, store)
    tstore = TrackStore(cfg.tracking)
    votes: Counter = Counter()
    present: dict[int, float] = {}
    t0 = time.time()
    while sim.t < duration:
        ts, obs = sim.step()
        states = tstore.update(ts, obs)
        for oid, *_ in obs:
            sid = tstore.raw_to_id.get(oid)
            if sid is not None:
                votes[(sid, sim.obs_to_true[oid])] += 1
        meas.step(ts, states, tstore.pop_expired())
        clean = [s for s in sim.clean_states() if s.id not in sim.staff_ids]  # ground truth: shoppers only
        now_ids = {s.id for s in clean}
        gone = [SimpleNamespace(id=i, last_ts=present[i], path_len=0.0) for i in list(present) if i not in now_ids]
        for s in clean:
            present[s.id] = ts
        for g in gone:
            present.pop(g.id, None)
        orac.step(ts, clean, gone)
    ts_end = sim.t
    meas.step(ts_end + 0.1, [], tstore.flush_all())
    orac.step(ts_end + 0.1, [], [SimpleNamespace(id=i, last_ts=t, path_len=0.0) for i, t in present.items()])
    meas.finalize()
    orac.finalize()

    # stable track -> true agent (majority vote)
    best: dict[int, tuple[int, int]] = {}
    for (sid, tid), n in votes.items():
        if n > best.get(sid, (0, -1))[0]:
            best[sid] = (n, tid)
    s2t = {sid: tid for sid, (n, tid) in best.items()}

    staff_visits_left = sum(1 for v in meas.visits.visits if s2t.get(v.track_id) in sim.staff_ids)

    # ---- visits & dwell
    gt_v = [(v.track_id, v.zone_id, v.t_enter, v.t_exit, v) for v in orac.visits.visits]
    pr_v = [(s2t.get(v.track_id, -v.track_id), v.zone_id, v.t_enter, v.t_exit, v) for v in meas.visits.visits]
    gt_iv = [((g[0], g[1]), g[2], g[3], g[4]) for g in gt_v]
    pr_iv = [((p[0], p[1]), p[2], p[3], p[4]) for p in pr_v]
    pairs = _match_intervals(gt_iv, pr_iv, lambda g: g[0], lambda p: p[0], tol=0.5)
    tp = len(pairs)
    prec = tp / len(pr_iv) if pr_iv else 1.0
    rec = tp / len(gt_iv) if gt_iv else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dwell_err = [abs(gt_iv[i][3].dwell_s - pr_iv[j][3].dwell_s) for i, j in pairs]
    eng_err = [abs(gt_iv[i][3].engaged_s - pr_iv[j][3].engaged_s) for i, j in pairs]
    nonwalk = [(i, j) for i, j in pairs if meas.ztype[gt_iv[i][3].zone_id] != "walkway"]
    dwell_err_merch = [abs(gt_iv[i][3].dwell_s - pr_iv[j][3].dwell_s) for i, j in nonwalk]

    # ---- bottlenecks
    gt_b = [(b.zone_id, b.t_start, b.t_end, b) for b in orac.density.bottlenecks]
    pr_b = [(b.zone_id, b.t_start, b.t_end, b) for b in meas.density.bottlenecks]
    bpairs = _match_intervals(gt_b, pr_b, lambda g: g[0], lambda p: p[0], tol=10.0)
    b_rec_strict = len(bpairs) / len(gt_b) if gt_b else None
    b_lat = [pr_b[j][3].t_confirmed - gt_b[i][3].t_confirmed for i, j in bpairs]

    def _ov(a, b, tol=10.0):  # same zone and overlapping in time (±tol)
        return a[0] == b[0] and min(a[2], b[2]) - max(a[1], b[1]) + tol > 0

    # any-overlap event matching (standard for event detection: one jam split differently still counts)
    gt_hit = [any(_ov(g, p) for p in pr_b) for g in gt_b]
    pr_hit = [any(_ov(p, g) for g in gt_b) for p in pr_b]
    b_rec = float(np.mean(gt_hit)) if gt_b else None
    b_prec = float(np.mean(pr_hit)) if pr_b else (None if not gt_b else 0.0)
    sus = [h for g, h in zip(gt_b, gt_hit) if g[2] - g[1] >= 60]
    b_rec_sus = float(np.mean(sus)) if sus else None

    # ---- zone classification (whole-run window)
    att = meas.compute_attention(window_s=duration + 1)
    att_or = orac.compute_attention(window_s=duration + 1)
    merch = [z for z in att if att[z]["status"] != "-"]
    exp = EXPECTED.get(scenario, {})
    correct = 0
    detail = {}
    for z in merch:
        st = att[z]["status"]
        want = exp.get(z)
        ok = (st == want) if want else (st not in ("IGNORED", "UNVISITED"))
        correct += ok
        detail[z] = st
    target_hit = all(att[z]["status"] == s for z, s in exp.items()) if exp else None
    oracle_agree = float(np.mean([att[z]["status"] == att_or[z]["status"] for z in merch])) if merch else None

    # ---- checkout queue: predicted W (Little) vs true time-in-checkout of shoppers entering then
    q = meas.queues.get("z_checkout")
    gt_co = [v for v in orac.visits.visits if v.zone_id == "z_checkout"]
    q_err = []
    block = 300.0
    if q is not None:
        for b0 in np.arange(0, duration, block):
            w = [x.predicted_wait_s for x in q.windows
                 if b0 <= x.window_start < b0 + block and x.predicted_wait_s is not None]
            g = [v.dwell_s for v in gt_co if b0 <= v.t_enter < b0 + block]
            if len(w) >= 3 and len(g) >= 3:
                q_err.append(abs(float(np.mean(w)) - float(np.mean(g))))

    return {
        "scenario": scenario, "seed": seed, "duration_s": duration, "wall_s": round(time.time() - t0, 1),
        "agents": sim._next_agent - 1, "id_breaks": sim.n_id_breaks, "stitches": tstore.n_stitches,
        "tracks_created": tstore.n_tracks_created,
        "fragmentation": tstore.n_tracks_created / max(sim._next_agent - 1, 1),
        "gt_visits": len(gt_iv), "pred_visits": len(pr_iv), "visit_precision": prec, "visit_recall": rec,
        "visit_f1": f1, "dwell_mae_s": float(np.mean(dwell_err)) if dwell_err else None,
        "dwell_mae_merch_s": float(np.mean(dwell_err_merch)) if dwell_err_merch else None,
        "engaged_mae_s": float(np.mean(eng_err)) if eng_err else None,
        "gt_bottlenecks": len(gt_b), "pred_bottlenecks": len(pr_b), "bottleneck_recall": b_rec,
        "bottleneck_recall_strict": b_rec_strict, "bottleneck_recall_sustained": b_rec_sus,
        "gt_sustained_bottlenecks": len(sus),
        "bottleneck_precision": b_prec, "bottleneck_latency_s": float(np.mean(b_lat)) if b_lat else None,
        "zone_class_accuracy": correct / len(merch) if merch else None, "zone_target_hit": target_hit,
        "zone_oracle_agreement": oracle_agree, "zone_status": detail,
        "queue_wait_mae_s": float(np.mean(q_err)) if q_err else None, "queue_blocks": len(q_err),
        "gt_checkout_visits": len(gt_co),
        "staff_agents": len(sim.staff_ids), "staff_tracks_flagged": len(meas.staff.staff),
        "staff_visits_purged": meas.staff_visits_purged, "staff_visits_left": staff_visits_left,
    }


def _fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.0%}" if pct else f"{x:.{nd}f}"


def run_eval(scenarios: list[str], seeds: int = 5, duration: float = 1800.0, workers: int = 4,
             out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or (ROOT / "runs" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, k, duration) for s in scenarios for k in range(seeds)]
    rows = []
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_run_star, jobs):
                rows.append(r)
                print(f"  {r['scenario']:14s} seed={r['seed']} visitF1={r['visit_f1']:.3f} "
                      f"dwellMAE={_fmt(r['dwell_mae_s'])}s bnRec={_fmt(r['bottleneck_recall'])} ({r['wall_s']}s)")
    else:
        for j in jobs:
            rows.append(_run_star(j))
    df = pd.DataFrame(rows)
    df.drop(columns=["zone_status"]).to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    write_report(df, out_dir / "report.md", seeds, duration, time.time() - t0)
    return df


def _run_star(a):
    return run_one(*a)


def write_report(df: pd.DataFrame, path: Path, seeds: int, duration: float, wall: float) -> None:
    num = ["visit_precision", "visit_recall", "visit_f1", "dwell_mae_s", "dwell_mae_merch_s", "engaged_mae_s",
           "bottleneck_recall", "bottleneck_recall_strict", "bottleneck_recall_sustained", "gt_sustained_bottlenecks",
           "bottleneck_precision", "bottleneck_latency_s", "zone_class_accuracy",
           "queue_wait_mae_s", "fragmentation", "gt_bottlenecks", "pred_bottlenecks", "agents", "id_breaks", "stitches"]
    g = df.groupby("scenario", sort=False)[num].agg(lambda s: pd.to_numeric(s, errors="coerce").mean())
    lines = [
        "# FlowSight — analytics accuracy vs simulator ground truth", "",
        f"{len(df)} runs = {df['scenario'].nunique()} scenarios × {seeds} seeds × {duration / 60:.0f} sim-minutes "
        f"(10 Hz). Measured pipeline sees σ=0.15 m noise, 2% dropouts and random ID breaks; it goes through the "
        f"real TrackStore (EMA + ID stitching). Ground truth = the same metric definitions on perfect data. "
        f"Wall time {wall / 60:.1f} min.", "",
        "| Scenario | Visit P | Visit R | Visit F1 | Dwell MAE (s) | Engaged MAE (s) | BN recall | BN recall ≥60 s "
        "| BN recall 1:1 | BN precision | BN latency (s) | Zone class. acc | Queue wait MAE (s) | Track frag. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sc, r in g.iterrows():
        lines.append(f"| {sc} | {_fmt(r.visit_precision, 3)} | {_fmt(r.visit_recall, 3)} | {_fmt(r.visit_f1, 3)} | "
                     f"{_fmt(r.dwell_mae_s)} | {_fmt(r.engaged_mae_s)} | {_fmt(r.bottleneck_recall)} | "
                     f"{_fmt(r.bottleneck_recall_sustained)} | {_fmt(r.bottleneck_recall_strict)} | "
                     f"{_fmt(r.bottleneck_precision)} | {_fmt(r.bottleneck_latency_s, 1)} | "
                     f"{_fmt(r.zone_class_accuracy, pct=True)} | {_fmt(r.queue_wait_mae_s, 1)} | "
                     f"{_fmt(r.fragmentation, 2)} |")
    all_mean = df[num].apply(pd.to_numeric, errors="coerce").mean()
    lines.append(f"| **all** | {_fmt(all_mean.visit_precision, 3)} | {_fmt(all_mean.visit_recall, 3)} | "
                 f"**{_fmt(all_mean.visit_f1, 3)}** | **{_fmt(all_mean.dwell_mae_s)}** | {_fmt(all_mean.engaged_mae_s)} | "
                 f"{_fmt(all_mean.bottleneck_recall)} | {_fmt(all_mean.bottleneck_recall_sustained)} | "
                 f"{_fmt(all_mean.bottleneck_recall_strict)} | {_fmt(all_mean.bottleneck_precision)} | "
                 f"{_fmt(all_mean.bottleneck_latency_s, 1)} | {_fmt(all_mean.zone_class_accuracy, pct=True)} | "
                 f"{_fmt(all_mean.queue_wait_mae_s, 1)} | {_fmt(all_mean.fragmentation, 2)} |")

    def tgt(name, val, ok):
        return f"| {name} | {val} | {'✅' if ok else '❌'} |"

    rush = df[df.scenario == "rush_hour"]
    rush_rec = pd.to_numeric(rush.bottleneck_recall, errors="coerce").mean() if len(rush) else float("nan")
    rush_sus = pd.to_numeric(rush.bottleneck_recall_sustained, errors="coerce").mean() if len(rush) else float("nan")
    hits = {sc: df[df.scenario == sc].zone_target_hit.mean() for sc in EXPECTED if (df.scenario == sc).any()}
    lines += ["", "## Targets", "", "| Target | Measured | Pass |", "|---|---|---|",
              tgt("Dwell MAE < 1.5 s", f"{all_mean.dwell_mae_s:.2f} s", all_mean.dwell_mae_s < 1.5),
              tgt("Visit F1 > 0.9", f"{all_mean.visit_f1:.3f}", all_mean.visit_f1 > 0.9),
              tgt("Bottleneck recall ≥ 0.9 on rush_hour (all episodes, any-overlap)", _fmt(rush_rec), bool(rush_rec >= 0.9)),
              tgt("Bottleneck recall ≥ 0.9 on rush_hour (sustained ≥ 60 s)", _fmt(rush_sus), bool(rush_sus >= 0.9))]
    for sc, h in hits.items():
        z, st = next(iter(EXPECTED[sc].items()))
        lines.append(tgt(f"{sc}: {z} classified {st}", f"{h:.0%} of seeds", h >= 0.999))
    st = df[df.get("staff_agents", 0) > 0] if "staff_agents" in df else df.iloc[0:0]
    if len(st):
        lines += ["", "## Staff filtering (with_staff scenario)", "",
                  f"{int(st.staff_agents.iloc[0])} simulated employees restock zones for the whole run "
                  f"({st.duration_s.iloc[0] / 60:.0f} min). Ground truth contains shoppers only.", "",
                  "| Seed | Staff tracks flagged | Staff visits purged | Staff visits left in analytics | Visit F1 |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r.seed} | {r.staff_tracks_flagged} | {r.staff_visits_purged} | {r.staff_visits_left} | "
                  f"{r.visit_f1:.3f} |" for _, r in st.iterrows()]
    lines += ["", "## Per-run zone status (measured)", "",
              "| Scenario | Seed | " + " | ".join(sorted(df.zone_status.iloc[0])) + " |",
              "|---|---|" + "---|" * len(df.zone_status.iloc[0])]
    for _, r in df.iterrows():
        lines.append(f"| {r.scenario} | {r.seed} | " + " | ".join(r.zone_status[z] for z in sorted(r.zone_status)) + " |")
    lines += ["", "## Notes", "",
              "* **Visit matching**: a measured visit matches a ground-truth visit when it is in the same zone, its "
              "track maps (majority vote over observations) to the same true shopper and the intervals overlap.",
              "* **Bottleneck matching**: same zone and overlapping interval (±10 s), *any-overlap* (a ground-truth jam "
              "counts as detected if any measured episode overlaps it, even when the two sides split it differently). "
              "`BN recall 1:1` is the stricter one-to-one matching. `≥60 s` = recall on sustained congestion only. "
              "Latency = measured confirmation time − ground-truth confirmation time (positive = later).",
              "* **Why short episodes are missed**: ground truth flags every ≥20 s window run at LoS D or speed < 50% "
              "free-flow, so 20–40 s aisle crowds sitting exactly on the threshold are coin-flips once 0.15 m noise is "
              "added. All sustained jams (queue spill-back, blocked walkway) are detected with a few seconds' latency.",
              "* **Queue wait**: per 5-minute block, mean Little's-law prediction W = L/λ vs the mean true time "
              "in the checkout zone of shoppers who entered in that block.",
              "* **Track fragmentation** = stable tracks created / true shoppers (1.0 = every ID break was stitched).",
              ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {path}")
