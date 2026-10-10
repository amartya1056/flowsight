import numpy as np

from flowsight.config import ZoneCfg
from flowsight.geometry.calibration import AutoScale, Homography
from flowsight.geometry.zones import ZoneSet

from tests.conftest import rect


def test_homography_round_trip():
    rng = np.random.default_rng(0)
    H_true = np.array([[0.02, 0.004, -3.0], [0.001, 0.05, -4.0], [0.0001, 0.002, 1.0]])
    img = rng.uniform(0, 800, (8, 2))
    hom = np.c_[img, np.ones(8)] @ H_true.T
    floor = hom[:, :2] / hom[:, 2:]
    H = Homography(img, floor)
    pts = rng.uniform(0, 800, (100, 2))
    back = H.floor_to_img(H.img_to_floor(pts))
    assert np.abs(back - pts).max() < 1e-6
    assert H.reprojection_error()["floor_m_max"] < 1e-6


def test_autoscale_recovers_scale_within_10pct():
    rng = np.random.default_rng(1)
    a_true, b_true = 0.4, 20.0  # box height = 0.4*y + 20 px
    s = AutoScale(person_height_m=1.7, image_width=1280)
    for _ in range(400):
        y2 = rng.uniform(150, 700)
        h = (a_true * y2 + b_true) * rng.normal(1.0, 0.04)
        s.add_box((600, y2 - h, 640, y2))
    s.fit()
    for y in (200, 400, 650):
        true_mpp = 1.7 / (a_true * y + b_true)
        assert abs(float(s.mpp(y)) - true_mpp) / true_mpp < 0.10


def test_zones_membership_and_exposure():
    zs = ZoneSet([ZoneCfg(id="a", name="A", type="aisle", polygon=rect(0, 0, 2, 2)),
                  ZoneCfg(id="w", name="W", type="walkway", polygon=rect(1, 0, 4, 2))], exposure_buffer_m=1.0)
    assert set(zs.zones_at(1.5, 1.0)) == {"a", "w"}  # overlap allowed
    m = zs.exposure_membership([2.7, 9.0], [1.0, 9.0])
    assert m[0, 0] and not m[1, 0]
    assert abs(zs["a"].area - 4.0) < 1e-9
