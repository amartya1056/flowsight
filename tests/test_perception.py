from flowsight.config import TrackingCfg
from flowsight.perception.footpoint import FootpointEstimator, foot_point
from flowsight.tracking.track_store import TrackStore


def test_footpoint_bottom_centre():
    x, y, trunc = foot_point((100, 50, 140, 250), frame_h=480)
    assert (x, y, trunc) == (120, 250, False)


def test_footpoint_truncated_uses_median_height():
    est = FootpointEstimator()
    for _ in range(5):
        est(1, (100, 100, 140, 300), 480)  # height 200, feet visible
    x, y, trunc = est(1, (100, 330, 140, 480), 480)  # touches bottom edge
    assert trunc is True
    assert abs(y - (330 + 200)) < 1e-6  # top + median height


def test_footpoint_truncated_without_history_keeps_bottom():
    x, y, trunc = FootpointEstimator()(7, (0, 300, 50, 479), 480)
    assert trunc and y == 479


def test_stitching_merges_fragment():
    ts = TrackStore(TrackingCfg(stitch_max_gap_s=2.0, stitch_max_dist_m=1.2, min_track_age_s=0.0))
    t = 0.0
    for k in range(20):  # raw id 5 walks right at 1 m/s
        t = k * 0.1
        ts.update(t, [(5, k * 0.1, 0.0)])
    first_id = ts.raw_to_id[5]
    # tracker loses the person for 0.8 s, then reports a new raw id roughly where expected
    out = None
    for k in range(28, 35):
        out = ts.update(k * 0.1, [(9, k * 0.1, 0.05)])
    assert ts.raw_to_id[9] == first_id
    assert ts.n_stitches == 1
    assert out[0].id == first_id


def test_no_stitch_when_too_far_or_too_late():
    ts = TrackStore(TrackingCfg(stitch_max_gap_s=2.0, stitch_max_dist_m=1.2))
    for k in range(10):
        ts.update(k * 0.1, [(1, 0.0, 0.0)])
    ts.update(1.5, [(2, 5.0, 5.0)])         # far away -> new track
    assert ts.raw_to_id[2] != ts.raw_to_id[1]
    for k in range(10):
        ts.update(2.0 + k * 0.1, [(3, 10.0, 10.0)])
    ts.update(6.0, [(4, 10.1, 10.0)])       # raw 3 lost 3.1 s ago -> expired, no stitch
    assert ts.n_stitches == 0


def test_stitch_rejects_backwards_heading():
    ts = TrackStore(TrackingCfg(stitch_max_gap_s=2.0, stitch_max_dist_m=1.2, min_track_age_s=0.0))
    for k in range(20):
        ts.update(k * 0.1, [(1, k * 0.15, 0.0)])  # moving +x at 1.5 m/s, ends at x=2.85
    ts.update(2.3, [(2, 2.2, 0.0)])               # appears ~0.4 m *behind* its last position
    assert ts.n_stitches == 0
