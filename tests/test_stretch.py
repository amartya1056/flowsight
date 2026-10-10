"""Stretch features: staff filtering, counter-flow, multi-camera fusion/auto-calibration, MOT GT parsing."""
import numpy as np

from flowsight.analytics.hub import AnalyticsHub
from flowsight.config import TrackingCfg
from flowsight.tracking.staff import StaffClassifier
from flowsight.tracking.track_store import TrackState

from tests.conftest import make_store, rect, ts_state

DT = 0.1


def _state(tid, t, x, y, speed, age):
    return TrackState(id=tid, ts=t, x=x, y=y, speed=speed, heading=0.0, vx=speed, vy=0.0, age=age, mature=True)


# ------------------------------------------------------------------ staff
def test_long_presence_staff_is_purged_retroactively(cfg):
    cfg = cfg.with_overrides(tracking={"staff_long_presence_s": 60})
    store = make_store([{"id": "a", "name": "A", "type": "aisle", "polygon": rect(0, 0, 2, 2)},
                        {"id": "b", "name": "B", "type": "aisle", "polygon": rect(5, 0, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    t = 0.0
    # employee (id 1) stands in A for 20 s, walks to B, works there; shopper (id 2) visits A briefly
    for k in range(700):
        t = round(k * DT, 3)
        x1 = 1.0 if t < 20 else (6.0 if t > 25 else 1.0 + (t - 20))
        states = [_state(1, t, x1, 1.0, 0.0, t)]
        if 5 <= t <= 15:
            states.append(_state(2, t, 1.2, 1.2, 0.0, t - 5))
        hub.step(t, states)
    assert hub.staff.is_staff(1) and hub.staff.staff[1] == "long_presence"
    assert not hub.staff.is_staff(2)
    hub.finalize()
    assert all(v.track_id != 1 for v in hub.visits.visits)  # employee's earlier visit to A was purged
    assert any(v.track_id == 2 for v in hub.visits.visits)
    assert hub.staff_visits_purged >= 1
    assert all(p["id"] != 1 for p in hub.people())


def test_uniform_colour_staff():
    cfg = TrackingCfg(staff_uniform_hsv=[[100, 130, 120, 255, 60, 255]], staff_color_min_samples=5)  # blue
    sc = StaffClassifier(cfg)
    frame = np.full((200, 200, 3), 128, np.uint8)
    frame[40:140, 20:60] = (200, 40, 20)      # blue shirt (BGR) on person 1
    frame[40:140, 120:160] = (20, 40, 200)    # red shirt on person 2
    flagged = [sc.observe_color(1, frame, (20, 20, 60, 180)) for _ in range(5)]
    for _ in range(10):
        sc.observe_color(2, frame, (120, 20, 160, 180))
    assert flagged[-1] and sc.is_staff(1) and sc.staff[1] == "uniform_colour"
    assert not sc.is_staff(2)


def test_staff_filter_can_be_disabled(cfg):
    cfg = cfg.with_overrides(tracking={"staff_long_presence_s": 1, "staff_filter": False})
    hub = AnalyticsHub(cfg, make_store([{"id": "a", "name": "A", "type": "aisle", "polygon": rect(0, 0, 2, 2)}]))
    for k in range(50):
        hub.step(k * DT, [ts_state(1, k * DT, 1, 1)])
    assert len(hub.people()) == 1
