"""Phase-3 acceptance tests with hand-built trajectories (floor metres, 10 Hz)."""
import numpy as np

from flowsight.analytics.density import DensityAnalyzer, fruin_los
from flowsight.analytics.hub import AnalyticsHub
from flowsight.analytics.queue import QueueAnalyzer
from flowsight.config import QueueCfg
from flowsight.geometry.zones import ZoneSet

from tests.conftest import make_store, rect, ts_state

DT = 0.1


def run(hub, frames):
    for t, states in frames:
        hub.step(t, states)
    hub.finalize()
    return hub


def test_stand_10s_is_one_visit_and_a_stop(cfg):
    store = make_store([{"id": "z", "name": "Z", "type": "aisle", "polygon": rect(5, 5, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    frames = []
    t = 0.0
    for k in range(30):  # walk in from the left at 1 m/s
        frames.append((round(t, 3), [ts_state(1, t, 2.0 + k * 0.1, 6.0, 1.0)])); t += DT
    for k in range(101):  # stand inside for exactly 10.0 s
        frames.append((round(t, 3), [ts_state(1, t, 6.0, 6.0, 0.0)])); t += DT
    for k in range(40):  # walk away
        frames.append((round(t, 3), [ts_state(1, t, 9.0 + k * 0.1, 6.0, 1.0)])); t += DT
    run(hub, frames)
    vs = [v for v in hub.visits.visits if v.zone_id == "z"]
    assert len(vs) == 1
    v = vs[0]
    # dwell = the 10 s standing plus the boundary-to-centre walking time (≤ 0.1 s quantisation)
    standing = [v for v in vs]
    assert abs(v.engaged_s - 10.0) <= 0.2
    assert v.is_stop


def test_stand_exactly_in_zone_dwell_10s(cfg):
    store = make_store([{"id": "z", "name": "Z", "type": "aisle", "polygon": rect(5, 5, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    frames = [(round(k * DT, 3), [ts_state(1, k * DT, 6.0, 6.0, 0.0)]) for k in range(101)]
    frames += [(round(10.1 + k * DT, 3), [ts_state(1, 10.1 + k * DT, 12.0, 12.0, 1.0)]) for k in range(30)]
    run(hub, frames)
    assert len(hub.visits.visits) == 1
    v = hub.visits.visits[0]
    assert abs(v.dwell_s - 10.0) <= 0.2
    assert v.is_stop


def test_boundary_jitter_is_one_visit(cfg):
    store = make_store([{"id": "z", "name": "Z", "type": "aisle", "polygon": rect(5, 5, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    frames = []
    for k in range(200):  # 20 s flipping in/out of the zone edge every 0.3 s
        t = k * DT
        inside = (int(t / 0.3) % 2) == 0
        frames.append((round(t, 3), [ts_state(1, t, 5.05 if inside else 4.95, 6.0, 0.1)]))
    frames += [(round(20 + k * DT, 3), [ts_state(1, 20 + k * DT, 0.5, 0.5, 1.0)]) for k in range(30)]
    run(hub, frames)
    assert len([v for v in hub.visits.visits if v.zone_id == "z"]) == 1


def test_pass_by_in_buffer_is_exposure_not_entry(cfg):
    store = make_store([{"id": "z", "name": "Z", "type": "promo", "polygon": rect(5, 5, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    # walk along y = 7.8 (0.8 m outside the zone, inside the 1.5 m buffer)
    frames = [(round(k * DT, 3), [ts_state(1, k * DT, 1.0 + k * 0.12, 7.8, 1.2)]) for k in range(100)]
    run(hub, frames)
    att = hub.compute_attention()
    assert att["z"]["exposures"] == 1
    assert att["z"]["entries"] == 0
    assert len(hub.visits.visits) == 0


def test_crowded_walkway_is_bottleneck(cfg):
    store = make_store([{"id": "w", "name": "W", "type": "walkway", "polygon": rect(0, 0, 2, 2)}])
    hub = AnalyticsHub(cfg, store)
    rng = np.random.default_rng(0)
    frames = []
    for k in range(301):  # 8 slow people in 4 m² for 30 s
        t = k * DT
        frames.append((round(t, 3), [ts_state(i, t, 0.2 + 1.6 * rng.random(), 0.2 + 1.6 * rng.random(), 0.1)
                                     for i in range(1, 9)]))
    run(hub, frames)
    bns = hub.density.bottlenecks
    assert len(bns) == 1
    b = bns[0]
    assert b.peak_los in ("D", "E", "F")
    assert len(b.people) == 8
    assert b.time_lost_s > 0


def test_fruin_thresholds(cfg):
    th = cfg.density.fruin_los_m2_per_person
    assert fruin_los(4.0, th) == "A"
    assert fruin_los(1.0, th) == "D"
    assert fruin_los(0.5, th) == "E"
    assert fruin_los(0.3, th) == "F"


def test_little_law_queue_wait():
    q = QueueAnalyzer("c", QueueCfg(window_s=30, rate_horizon_s=300))
    for minute in range(10):  # lambda = 1 / min
        q.on_entry(minute * 60.0)
    for k in range(6000):  # constant queue of 5 for 10 minutes
        q.sample(k * 0.1, 5)
    st = q.status(599.9)
    assert abs(st["queue_len"] - 5) < 1e-6
    assert abs(st["predicted_wait_s"] - 300) / 300 < 0.05  # W = L / λ = 5 min


def test_heatmap_is_time_weighted(cfg):
    store = make_store([{"id": "z", "name": "Z", "type": "aisle", "polygon": rect(0, 0, 1, 1)}])
    hub = AnalyticsHub(cfg, store)
    for k in range(1, 51):
        hub.step(k * DT, [ts_state(1, k * DT, 3.1, 3.1, 0.0)])
    g = hub.heat.grid("occupancy")
    assert abs(g.sum() - 5.0) < 0.15          # 5 s of person-time, independent of frame count
    assert abs(hub.heat.grid("stationary").sum() - 5.0) < 0.15
