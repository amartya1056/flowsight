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


# ------------------------------------------------------------ counter-flow
def _stream(cfg, two_way: bool):
    store = make_store([{"id": "w", "name": "W", "type": "walkway", "polygon": rect(0, 0, 10, 2)}])
    hub = AnalyticsHub(cfg, store)
    for k in range(800):  # 80 s, 12 walkers at 0.8 m/s wrapping around a 10 x 2 m walkway (LoS C)
        t = round(k * DT, 3)
        states = []
        for i in range(12):
            d = -1 if (two_way and i % 2) else 1
            x = (i * 0.83 + d * 0.8 * t) % 10
            states.append(TrackState(id=i + 1, ts=t, x=x, y=0.3 + 0.12 * i, speed=0.8, heading=0.0,
                                     vx=0.8 * d, vy=0.0, age=30, mature=True))
        hub.step(t, states)
    return hub


def test_counterflow_conflict_detected_for_two_way_stream(cfg):
    hub = _stream(cfg, two_way=True)
    w = hub.density.latest["w"]
    assert w.counterflow is not None and w.counterflow > 0.4
    assert w.direction_entropy is not None and w.direction_entropy < 0.5  # two opposite bins
    assert len(hub.density.conflicts) == 1 and hub.density.conflicts[0].peak_los in ("C", "D")
    from flowsight.actions.engine import ActionEngine

    eng = ActionEngine(cfg, hub)
    eng.evaluate(hub.ts, force=True)
    assert any(i.type == "counter_flow" for i in eng.active.values())


def test_one_way_stream_is_not_a_conflict(cfg):
    hub = _stream(cfg, two_way=False)
    assert hub.density.latest["w"].counterflow == 0.0
    assert not hub.density.conflicts


# --------------------------------------------------- multi-camera calibration
def test_autocalib_recovers_offset_and_homography():
    import cv2

    from flowsight.geometry.autocalib import estimate

    rng = np.random.default_rng(3)
    H_true = np.array([[0.012, 0.002, -0.5], [0.0005, 0.03, -1.0], [0.00005, 0.0015, 1.0]])  # px -> m
    Hinv = np.linalg.inv(H_true)
    T, offset = 700, 20
    walkers = [rng.uniform([0.5, 3], [3.5, 9], 2) for _ in range(3)]
    vel = [rng.normal(0, 0.03, 2) for _ in range(3)]
    truth = {}
    for t in range(T):
        pts = []
        for i in range(3):
            walkers[i] = np.clip(walkers[i] + vel[i], [0.2, 2.0], [4.0, 10.0])
            if walkers[i][0] in (0.2, 4.0) or walkers[i][1] in (2.0, 10.0):
                vel[i] = -vel[i]
            pts.append(walkers[i].copy())
        truth[t] = np.array(pts)
    ref, new = {}, {}
    for t in range(T):
        clutter = rng.uniform([0, 0], [4.5, 16], (1, 2))  # someone the new camera cannot see
        ref[t] = np.vstack([truth[t] + rng.normal(0, 0.05, truth[t].shape), clutter])
        if t + offset < T + offset:  # new camera clock runs `offset` frames ahead
            px = cv2.perspectiveTransform(truth[t][None], Hinv)[0] + rng.normal(0, 1.0, (3, 2))
            junk = rng.uniform([0, 0], [380, 280], (1, 2))  # e.g. a person inside the shop
            new[t + offset] = np.vstack([px[:2], junk])     # one walker hidden in the new view
    r = estimate(ref, new, offsets=range(-40, 41, 4), refine=4, iters=1500)
    assert abs(r.offset_frames - offset) <= 2  # 2 frames = 0.08 s at 25 FPS
    test_px = cv2.perspectiveTransform(truth[100][None], Hinv)[0]
    err = np.linalg.norm(cv2.perspectiveTransform(test_px[None], r.H)[0] - truth[100], axis=1)
    assert err.max() < 0.25


# ------------------------------------------------------------- fusion
def _two_cam_walk(rng, people):
    """people: list of (start_x, y, vx). Cam A sees x < 12, cam B sees x > 8 (overlap 8..12), separate ID spaces."""
    from flowsight.tracking.fusion import CrossCameraFuser
    from flowsight.tracking.track_store import TrackStore
    from flowsight.config import TrackingCfg

    fuser = CrossCameraFuser(["A", "B"], gate_m=1.0)
    ts = TrackStore(TrackingCfg(min_track_age_s=0.0))
    ids_per_person = [set() for _ in people]
    for k in range(200):
        t = round(k * 0.1, 3)
        obs = {"A": [], "B": []}
        for p, (x0, y, vx) in enumerate(people):
            x = x0 + vx * t
            if x < 12:
                obs["A"].append((f"a{p}", x + rng.normal(0, .1), y + rng.normal(0, .1)))
            if x > 8:
                obs["B"].append((f"b{p}", x + rng.normal(0, .1), y + rng.normal(0, .1)))
        fused = fuser.fuse(obs)
        states = ts.update(t, [(f[0], f[1], f[2]) for f in fused])
        for s in states:  # attribute global ids to the true person by nearest true position
            d = [abs(s.y - y) for (_, y, _) in people]
            ids_per_person[int(np.argmin(d))].add(s.id)
    return fuser, ts, ids_per_person


def test_fusion_hands_off_one_id_across_cameras():
    fuser, ts, ids = _two_cam_walk(np.random.default_rng(0), [(0.0, 5.0, 1.0)])
    assert ids == [{1}]  # one global identity from camera A through the overlap into camera B
    s = fuser.stats.summary()
    assert s["handoffs"] == 1 and s["cross_camera_keys"] == 1 and s["median_cross_dist_m"] < 0.3


def test_fusion_keeps_people_apart_in_overlap():
    fuser, ts, ids = _two_cam_walk(np.random.default_rng(1), [(0.0, 3.0, 1.0), (20.0, 6.0, -1.0)])
    assert all(len(s) == 1 for s in ids) and ids[0] != ids[1]
    assert fuser.stats.summary()["handoffs"] == 2
