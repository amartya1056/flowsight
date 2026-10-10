"""YOLO11 person detection + ByteTrack (Ultralytics built-in), device auto-selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flowsight.config import RuntimeCfg, resolve_path

log = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    conf: float


def select_device(pref: str = "auto") -> tuple[str, bool]:
    """Return (device, half_precision)."""
    import torch

    if pref not in ("auto", ""):
        return pref, pref.startswith("cuda")
    if torch.cuda.is_available():
        return "cuda:0", True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def ensure_weights(name: str, models_dir: str) -> Path:
    """Make sure YOLO weights exist under models_dir (downloads on first use)."""
    d = resolve_path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(path))
    return path


class DetectorTracker:
    def __init__(self, cfg: RuntimeCfg, model_name: str | None = None, imgsz: int | None = None,
                 tracker: str | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device, self.half = select_device(cfg.device)
        if model_name is None:
            model_name = cfg.model_gpu if self.device.startswith(("cuda", "mps")) else cfg.model_cpu
        self.model_name = model_name
        self.imgsz = imgsz or cfg.imgsz
        self.tracker = tracker or cfg.tracker
        weights = ensure_weights(model_name, cfg.models_dir) if model_name.endswith(".pt") else Path(model_name)
        self.model = YOLO(str(weights))
        # Ultralytics >= 8.4 replaces `half=True` with `quantize=16`; support both
        from ultralytics.cfg import DEFAULT_CFG_DICT
        self._prec = ({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}) if self.half else {}
        log.info("detector: %s on %s (half=%s, imgsz=%d, tracker=%s)",
                 model_name, self.device, self.half, self.imgsz, self.tracker)

    def update(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.track(
            frame, persist=True, classes=[0], conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.imgsz,
            tracker=self.tracker, device=self.device, verbose=False, **self._prec,
        )[0]
        out: list[Detection] = []
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for b, i, c in zip(xyxy, ids, confs):
            out.append(Detection(int(i), tuple(float(v) for v in b), float(c)))
        return out

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Detection only (no tracking state) — used for one-off frames such as calibration previews."""
        res = self.model.predict(frame, classes=[0], conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False, **self._prec)[0]
        if res.boxes is None:
            return []
        return [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
