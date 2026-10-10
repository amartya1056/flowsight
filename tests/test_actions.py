"""Phase-4: one test per action rule + engine behaviour (dedup, cooldown, auto-resolve, cap)."""
import numpy as np

from flowsight.actions.briefing import generate_briefing
from flowsight.actions.engine import ActionEngine, fmt_dur
from flowsight.analytics.density import ZoneWindow
from flowsight.analytics.hub import AnalyticsHub

from tests.conftest import make_store, rect, ts_state

DT = 0.1
ZONES = [
    {"id": "walk", "name": "Main Walkway", "type": "walkway", "polygon": rect(0, 0, 2, 2)},
    {"id": "promo", "name": "Promo End-cap", "type": "promo", "polygon": rect(4, 0, 2, 2)},
    {"id": "a2", "name": "Aisle 2", "type": "aisle", "polygon": rect(8, 0, 2, 2)},
    {"id": "a7", "name": "Aisle 7", "type": "aisle", "polygon": rect(12, 0, 2, 2)},
    {"id": "ph", "name": "Pharmacy", "type": "service", "polygon": rect(4, 6, 2, 2)},
    {"id": "co", "name": "Checkout", "type": "checkout", "polygon": rect(12, 6, 3, 3)},
]


def setup(cfg):
    hub = AnalyticsHub(cfg, make_store(ZONES))
    return hub, ActionEngine(cfg, hub)


def types(engine):
    return {(i.type, i.zone_id) for i in engine.active.values()}


def test_checkout_queue_rule(cfg):
    hub, eng = setup(cfg)
    q = hub.queues["co"]
    for m in range(5):
        q.on_entry(m * 60.0)  # λ = 1 / min
    for k in range(3000):
        q.sample(k * DT, 5)  # L = 5 -> W = 300 s > 180 s target
    hub.ts = 299.9
    eng.evaluate(299.9, force=True)
    ins = [i for i in eng.active.values() if i.type == "checkout_queue"]
    assert len(ins) == 1
    assert "Open another checkout lane" in ins[0].action
    assert ins[0].horizon == "now" and ins[0].severity == "critical"


def test_bottleneck_rule(cfg):
    hub, eng = setup(cfg)
    rng = np.random.default_rng(0)
    for k in range(301):
        t = k * DT
        hub.step(t, [ts_state(i, t, 0.2 + 1.6 * rng.random(), 0.2 + 1.6 * rng.random(), 0.1) for i in range(1, 9)])
    eng.evaluate(30.0, force=True)
    ins = [i for i in eng.active.values() if i.type == "bottleneck"]
    assert len(ins) == 1 and ins[0].zone_id == "walk"
    assert ins[0].action.startswith("Clear Main Walkway") and "LoS" in ins[0].action


def test_recurring_bottleneck_rule(cfg):
    hub, eng = setup(cfg)
    t = 0.0
    rng = np.random.default_rng(1)
    for episode in range(3):  # three 30 s jams separated by 60 s of calm
        for k in range(300):
            hub.step(round(t, 3), [ts_state(i + 100 * episode, t, 0.2 + 1.6 * rng.random(), 0.2 + 1.6 * rng.random(), 0.1)
                                   for i in range(1, 9)]); t += DT
        for k in range(600):
            hub.step(round(t, 3), []); t += DT
    assert len(hub.density.bottlenecks) == 3
    eng.evaluate(t, force=True)
    assert ("recurring_bottleneck", "walk") in types(eng)


def _attention(status_by_zone):
    out = {}
    for zid, (status, exp, stops) in status_by_zone.items():
        out[zid] = {"zone_id": zid, "exposures": exp, "entries": stops, "stops": stops, "engaged_s": 10.0 * stops,
                    "stopping_power": stops / exp if exp else None, "status": status, "reason": "",
                    "median_exposure": 200, "traffic_share": exp / 200}
    return out


def test_ignored_unvisited_and_whatif_rules(cfg):
    hub, eng = setup(cfg)
    hub.ts = 900.0
    hub.attention_latest = _attention({"promo": ("IGNORED", 212, 6), "a2": ("HOT", 200, 62),
                                       "a7": ("UNVISITED", 24, 3), "ph": ("NORMAL", 180, 30)})
    eng.evaluate(900.0, force=True)
    t = types(eng)
    assert ("ignored_zone", "promo") in t
    assert ("unvisited_zone", "a7") in t
    assert ("what_if", "promo") in t
    ign = next(i for i in eng.active.values() if i.type == "ignored_zone")
    assert "212 shoppers passed, only 3% stopped" in ign.action
    wi = next(i for i in eng.active.values() if i.type == "what_if")
    assert "estimate" in wi.action and wi.horizon == "this_week"
    assert wi.evidence.metrics["uplift_high_per_day"] >= wi.evidence.metrics["uplift_low_per_day"] > 0


def test_occupancy_anomaly_rule(cfg):
    hub, eng = setup(cfg)
    for k in range(60):
        hub.baseline.add("a2", "occupancy", 1.0 + 0.2 * (k % 3))
    for k in range(12):
        hub.baseline.add("a2", "occupancy", 6.0)
    hub.density.latest["a2"] = ZoneWindow("a2", 0, 5, 6.0, 1.5, "C", 0.2, 1.0, 1.0)
    hub.ts = 400.0
    eng.evaluate(400.0, force=True)
    ins = [i for i in eng.active.values() if i.type == "occupancy_anomaly"]
    assert len(ins) == 1 and "× normal" in ins[0].action


def test_long_stationary_rule(cfg):
    hub, eng = setup(cfg)
    for k in range(1301):  # 130 s standing at the pharmacy counter
        hub.step(k * DT, [ts_state(1, k * DT, 5.0, 7.0, 0.0)])
    eng.evaluate(130.0, force=True)
    ins = [i for i in eng.active.values() if i.type == "assist_shopper"]
    assert len(ins) == 1 and "Pharmacy" in ins[0].action and "2m10s" in ins[0].action


def test_engine_dedup_resolve_and_cooldown(cfg):
    hub, eng = setup(cfg)
    hub.ts = 100.0
    hub.attention_latest = _attention({"promo": ("IGNORED", 212, 6)})
    eng.evaluate(100.0, force=True)
    eng.evaluate(105.0, force=True)
    assert len([i for i in eng.all.values() if i.type == "ignored_zone"]) == 1  # dedup
    hub.attention_latest = _attention({"promo": ("NORMAL", 212, 40)})
    eng.evaluate(110.0, force=True)
    assert not eng.active and next(iter(eng.all.values())).status == "resolved"  # auto-resolve
    hub.attention_latest = _attention({"promo": ("IGNORED", 212, 6)})
    eng.evaluate(120.0, force=True)
    assert not eng.active  # cooldown
    eng.evaluate(110.0 + cfg.actions.cooldown_s + 1, force=True)
    assert len(eng.active) == 1


def test_engine_caps_active(cfg):
    cfg = cfg.with_overrides(actions={"max_active": 2})
    hub, eng = setup(cfg)
    hub.ts = 100.0
    hub.attention_latest = _attention({"promo": ("IGNORED", 212, 6), "a2": ("IGNORED", 150, 2),
                                       "a7": ("UNVISITED", 10, 0), "ph": ("IGNORED", 90, 1)})
    eng.evaluate(100.0, force=True)
    assert len(eng.active) == 2


def test_briefing_template_mode(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    hub, eng = setup(cfg)
    hub.ts = 900.0
    hub.attention_latest = _attention({"promo": ("IGNORED", 212, 6), "a2": ("HOT", 200, 62)})
    eng.evaluate(900.0, force=True)
    b = generate_briefing(cfg.briefing, hub.kpis(), [i.model_dump() for i in eng.active_sorted()],
                          list(hub.zone_kpis().values()))
    assert b["mode"] == "template" and len(b["bullets"]) == 5


def test_briefing_llm_mode_with_fake_client(cfg, monkeypatch):
    import sys
    import types as t

    class _Msgs:
        def create(self, **kw):
            assert "Never invent numbers" in kw["system"]
            blk = t.SimpleNamespace(type="text", text="- a\n- b\n- c\n- d\n- e")
            return t.SimpleNamespace(content=[blk])

    fake = t.SimpleNamespace(Anthropic=lambda api_key: t.SimpleNamespace(messages=_Msgs()))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    hub, eng = setup(cfg)
    b = generate_briefing(cfg.briefing, hub.kpis(), [], list(hub.zone_kpis().values()))
    assert b["mode"] == "llm" and b["bullets"] == ["a", "b", "c", "d", "e"]


def test_fmt_dur():
    assert fmt_dur(250) == "4m10s" and fmt_dur(42) == "42s"
