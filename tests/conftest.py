import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowsight.config import AppConfig, StoreConfig  # noqa: E402
from flowsight.tracking.track_store import TrackState  # noqa: E402


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def make_store(zones, w=20.0, h=12.0) -> StoreConfig:
    return StoreConfig.model_validate({
        "store_id": "t", "floorplan": {"image": "none.png", "width_m": w, "height_m": h},
        "cameras": [], "zones": zones})


def rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def ts_state(tid, t, x, y, speed=0.0):
    return TrackState(id=tid, ts=t, x=x, y=y, speed=speed, heading=0.0, vx=speed, vy=0.0, age=10.0, mature=True)
