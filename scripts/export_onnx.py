"""Export the person detector to ONNX for CPU / edge inference (ONNX Runtime, OpenVINO, TensorRT can all load it).

    python scripts/export_onnx.py [--models yolo11n.pt,yolo11s.pt] [--imgsz 640]

The exported file is a drop-in: set `runtime.model_cpu: yolo11n.onnx` in config/default.yaml (Ultralytics runs
ONNX models through the same .track() API, so ByteTrack and the rest of the pipeline are unchanged).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "1")  # never pip-install at export time

from flowsight.config import load_config, resolve_path  # noqa: E402
from flowsight.perception.detector_tracker import ensure_weights  # noqa: E402


def export(name: str, imgsz: int, models_dir: str) -> Path:
    from ultralytics import YOLO

    pt = ensure_weights(name, models_dir)
    out = YOLO(str(pt)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False, opset=17, device="cpu")
    dst = resolve_path(models_dir) / f"{Path(name).stem}_{imgsz}.onnx"
    Path(out).replace(dst)
    print(f"exported {name} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yolo11n.pt,yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    cfg = load_config()
    for m in a.models.split(","):
        export(m.strip(), a.imgsz, cfg.runtime.models_dir)


if __name__ == "__main__":
    main()
