"""Pydantic config models + loaders. Every tunable lives in config/*.yaml — nothing is hard-coded."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("FLOWSIGHT_CONFIG_DIR", ROOT / "config"))
STORES_DIR = CONFIG_DIR / "stores"

ZoneType = Literal["entrance", "aisle", "promo", "walkway", "checkout", "service"]


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


# ---------------------------------------------------------------- app config
class RuntimeCfg(BaseModel):
    target_fps: float = 10
    device: str = "auto"
    model_gpu: str = "yolo11s.pt"
    model_cpu: str = "yolo11n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 960
    tracker: str = "bytetrack.yaml"
    models_dir: str = "models"


class GeometryCfg(BaseModel):
    heatmap_cell_m: float = 0.25
    heatmap_blur_sigma_cells: float = 1.5
    assumed_person_height_m: float = 1.70


class TrackingCfg(BaseModel):
    smoothing: str = "ema"
    ema_alpha: float = 0.4
    speed_window_s: float = 1.0
    position_noise_m: float = 0.15
    min_track_age_s: float = 1.0
    stitch_max_gap_s: float = 2.0
    stitch_max_dist_m: float = 1.2
    stitch_max_dist_px: float = 80
    staff_filter: bool = True
    staff_long_presence_s: float = 1800
    staff_uniform_hsv: list[list[int]] = Field(default_factory=list)
    staff_color_min_frac: float = 0.35
    staff_color_min_samples: int = 10


class VisitsCfg(BaseModel):
    enter_min_s: float = 1.0
    exit_grace_s: float = 1.5
    engaged_speed_mps: float = 0.3
    stop_min_s: float = 3.0
    exposure_buffer_m: float = 1.5


class DensityCfg(BaseModel):
    window_s: float = 5
    bottleneck_min_duration_s: float = 20
    speed_drop_ratio: float = 0.5
    min_occupancy_for_speed_rule: float = 3
    default_free_flow_mps: float = 1.1
    min_baseline_samples: int = 50
    fruin_los_m2_per_person: dict[str, float] = Field(
        default_factory=lambda: {"A": 3.3, "B": 2.3, "C": 1.4, "D": 0.9, "E": 0.5})
    bottleneck_zone_types: list[str] = Field(default_factory=lambda: ["walkway", "entrance", "aisle"])
    aisle_speed_rule_min_los: str = "C"
    bottleneck_end_windows: int = 2


class AttentionCfg(BaseModel):
    window_min: float = 15
    unvisited_traffic_share: float = 0.2
    ignored_stopping_power: float = 0.08
    hot_quantile: float = 0.9
    hot_min_stopping_power: float = 0.15
    hot_min_traffic_share: float = 0.75
    min_exposures: int = 30
    merch_zone_types: list[str] = Field(default_factory=lambda: ["aisle", "promo", "service"])


class FlowCfg(BaseModel):
    exclude_types: list[str] = Field(default_factory=lambda: ["walkway"])
    sankey_max_steps: int = 5


class QueueCfg(BaseModel):
    service_zone_type: str = "checkout"
    max_wait_target_s: float = 180
    window_s: float = 30
    rate_horizon_s: float = 300
    rising_windows: int = 3


class BaselineCfg(BaseModel):
    history_windows: int = 360
    min_samples: int = 24
    min_std: float = 0.5


class ActionsCfg(BaseModel):
    cooldown_s: float = 300
    max_active: int = 12
    eval_every_s: float = 5
    resolve_grace_s: float = 30
    anomaly_smooth_windows: int = 6
    recurring_bottleneck_count: int = 3
    anomaly_z: float = 3.0
    anomaly_min_ratio: float = 2.0
    anomaly_min_occupancy: float = 4
    long_stationary_s: float = 120
    store_open_hours: float = 12
    promo_stopping_power_prior: float = 0.15


class BriefingCfg(BaseModel):
    llm_model: str = "claude-sonnet-5-5"
    enabled_if_key: bool = True
    max_tokens: int = 600


class PrivacyCfg(BaseModel):
    blur_heads: bool = True
    head_fraction: float = 0.22
    store_frames: str = "evidence_only"


class StorageCfg(BaseModel):
    track_point_hz: float = 2
    flush_every_s: float = 1.0


class RenderCfg(BaseModel):
    trail_s: float = 3.0
    floor_px_per_m: float = 50
    jpeg_quality: int = 80


class SimCfg(BaseModel):
    dt: float = 0.1
    duration_s: float = 1800
    noise_sigma_m: float = 0.15
    dropout_prob: float = 0.02
    id_break_rate_per_s: float = 0.01
    id_break_gap_s: list[float] = Field(default_factory=lambda: [0.3, 1.5])


class ApiCfg(BaseModel):
    live_push_ms: int = 500
    cors_origins: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    geometry: GeometryCfg = Field(default_factory=GeometryCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    visits: VisitsCfg = Field(default_factory=VisitsCfg)
    density: DensityCfg = Field(default_factory=DensityCfg)
    attention: AttentionCfg = Field(default_factory=AttentionCfg)
    flow: FlowCfg = Field(default_factory=FlowCfg)
    queue: QueueCfg = Field(default_factory=QueueCfg)
    baseline: BaselineCfg = Field(default_factory=BaselineCfg)
    actions: ActionsCfg = Field(default_factory=ActionsCfg)
    briefing: BriefingCfg = Field(default_factory=BriefingCfg)
    privacy: PrivacyCfg = Field(default_factory=PrivacyCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    render: RenderCfg = Field(default_factory=RenderCfg)
    sim: SimCfg = Field(default_factory=SimCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)

    def with_overrides(self, **sections) -> "AppConfig":
        """Return a copy with nested overrides, e.g. cfg.with_overrides(visits={"enter_min_s": 0.5})."""
        data = copy.deepcopy(self.model_dump())
        for sec, vals in sections.items():
            data.setdefault(sec, {}).update(vals)
        return AppConfig.model_validate(data)


# -------------------------------------------------------------- store config
class Calibration(BaseModel):
    image_points: list[list[float]]
    floor_points: list[list[float]]


class CameraCfg(BaseModel):
    id: str
    name: str = ""
    source: str
    approximate: bool = False
    calibration: Optional[Calibration] = None


class FloorplanCfg(BaseModel):
    image: str
    width_m: float
    height_m: float


class ZoneCfg(BaseModel):
    id: str
    name: str
    type: ZoneType
    polygon: list[list[float]]
    overflow: list[str] = Field(default_factory=list)  # checkout only: zones a long queue spills into


class FixtureCfg(BaseModel):
    name: str
    rect: list[float]  # x, y, w, h in metres


class DoorCfg(BaseModel):
    name: str
    at: list[float]


class StoreConfig(BaseModel):
    store_id: str
    name: str = ""
    floorplan: FloorplanCfg
    cameras: list[CameraCfg] = Field(default_factory=list)
    zones: list[ZoneCfg] = Field(default_factory=list)
    fixtures: list[FixtureCfg] = Field(default_factory=list)
    doors: list[DoorCfg] = Field(default_factory=list)
    notes: str = ""
    sim: dict = Field(default_factory=dict)  # simulator layout (only for simulated stores)

    def camera(self, cam_id: str) -> CameraCfg:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise KeyError(f"camera {cam_id!r} not in store {self.store_id!r}")

    def zone(self, zone_id: str) -> ZoneCfg:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(zone_id)


# ------------------------------------------------------------------- loaders
def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)


def list_stores() -> list[str]:
    return sorted(p.stem for p in STORES_DIR.glob("*.yaml"))


def load_store(store_id: str) -> StoreConfig:
    path = STORES_DIR / f"{store_id}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return StoreConfig.model_validate(yaml.safe_load(f))


def save_store(store: StoreConfig) -> Path:
    path = STORES_DIR / f"{store.store_id}.yaml"
    data = store.model_dump(exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, width=140)
    return path
