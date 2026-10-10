"""Tracking metrics (CLEAR-MOT / IDF1) on CAVIAR ground truth.

CAVIAR's shopping-centre corridor sequences are annotated per frame with person boxes and identities (CVML XML).
We run the production perception stack on every frame and score two hypotheses with `motmetrics`
(IoU matching, threshold 0.5):
  * ByteTrack   — raw tracker IDs from Ultralytics
  * FlowSight   — the same boxes relabelled with FlowSight's stable IDs (floor projection + EMA + ID stitching)
The difference isolates what the ID-stitching layer adds to identity consistency (IDF1, ID switches).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SEQUENCES = {  # sequence -> (video, GT xml)
    "ThreePastShop1cor": ("data/videos/caviar_ThreePastShop1cor.mp4", "data/gt/c3ps1gt.xml"),
    "OneShopOneWait1cor": ("data/videos/caviar_OneShopOneWait1cor.mp4", "data/gt/cosow1gt.xml"),
    "TwoEnterShop1cor": ("data/videos/caviar_TwoEnterShop1cor.mp4", "data/gt/c2es1gt.xml"),
}


def load_gt_boxes(path) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    out = {}
    for fr in ET.parse(path).getroot().iter("frame"):
        n = int(fr.get("number"))
        d = {}
        for o in fr.iter("object"):
            b = o.find("box")
            xc, yc, w, h = (float(b.get(k)) for k in ("xc", "yc", "w", "h"))
            d[int(o.get("id"))] = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        out[n] = d
    return out


def iou_distance(a_xywh: np.ndarray, b_xywh: np.ndarray, max_iou: float = 0.5) -> np.ndarray:
    """motmetrics-style distance: 1 - IoU, NaN where IoU < 1 - max_iou (motmetrics' own helper breaks on NumPy 2)."""
    if len(a_xywh) == 0 or len(b_xywh) == 0:
        return np.full((len(a_xywh), len(b_xywh)), np.nan)
    a, b = a_xywh[:, None, :], b_xywh[None, :, :]
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter
    d = 1.0 - inter / np.maximum(union, 1e-9)
    d[d > max_iou] = np.nan
    return d


def _xywh(boxes):
    a = np.asarray(boxes, float).reshape(-1, 4)
    return np.c_[a[:, :2], a[:, 2:] - a[:, :2]]


def run_sequence(name: str, cfg, store, root: Path) -> dict:
    import cv2
    import motmetrics as mm

    from flowsight.geometry.calibration import build_mapping
    from flowsight.perception.detector_tracker import DetectorTracker
    from flowsight.perception.footpoint import FootpointEstimator
    from flowsight.tracking.track_store import TrackStore

    video, gtp = SEQUENCES[name]
    gt = load_gt_boxes(root / gtp)
    cap = cv2.VideoCapture(str(root / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = DetectorTracker(cfg.runtime)
    cam = store.camera("cam1")
    mapping = build_mapping(cam, cfg.geometry.assumed_person_height_m, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    foot = FootpointEstimator()
    ts_store = TrackStore(cfg.tracking.model_copy(update={"min_track_age_s": 0.0}))
    acc_bt, acc_fs = mm.MOTAccumulator(auto_id=True), mm.MOTAccumulator(auto_id=True)
    n = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n not in gt:
            continue
        dets = det.update(frame)
        H = frame.shape[0]
        obs = []
        for d in dets:
            fx, fy, tr = foot(d.track_id, d.bbox_xyxy, H)
            fl = mapping.img_to_floor([[fx, fy]])[0]
            obs.append((d.track_id, float(fl[0]), float(fl[1]), tr))
        ts_store.update(n / fps, obs)
        g_ids = list(gt[n])
        g_box = _xywh([gt[n][i] for i in g_ids]) if g_ids else np.zeros((0, 4))
        h_box = _xywh([d.bbox_xyxy for d in dets]) if dets else np.zeros((0, 4))
        dist = iou_distance(g_box, h_box, max_iou=0.5)
        acc_bt.update(g_ids, [d.track_id for d in dets], dist)
        acc_fs.update(g_ids, [ts_store.raw_to_id.get(d.track_id, -d.track_id) for d in dets], dist)
    metrics = ["num_frames", "mota", "motp", "idf1", "idp", "idr", "recall", "precision", "num_switches",
               "mostly_tracked", "mostly_lost", "num_unique_objects", "num_false_positives", "num_misses"]
    mh = mm.metrics.create()
    out = {}
    for label, acc in (("ByteTrack", acc_bt), ("FlowSight", acc_fs)):
        s = mh.compute(acc, metrics=metrics, name=label)
        out[label] = {k: float(s[k].iloc[0]) for k in metrics}
    out["stitches"] = ts_store.n_stitches
    return out


def run_all(cfg, store, root: Path, out_md: Path) -> dict:
    res = {name: run_sequence(name, cfg, store, root) for name in SEQUENCES
           if (root / SEQUENCES[name][0]).exists() and (root / SEQUENCES[name][1]).exists()}
    L = ["# Tracking metrics on CAVIAR ground truth", "",
         f"Detector `{cfg.runtime.model_gpu}` @ imgsz {cfg.runtime.imgsz}, conf {cfg.runtime.conf}, Ultralytics ByteTrack; "
         "every frame (25 FPS, 384×288), IoU ≥ 0.5 matching (`motmetrics`). *FlowSight* = the same boxes with FlowSight's "
         "stable IDs (floor projection + EMA + ID stitching).", "",
         "| Sequence | IDs | MOTA | IDF1 | IDP | IDR | ID switches | Recall | Precision | MT / ML | Stitches |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for lab in ("ByteTrack", "FlowSight"):
            m = r[lab]
            L.append(f"| {name} | {lab} | {m['mota']:.3f} | {m['idf1']:.3f} | {m['idp']:.3f} | {m['idr']:.3f} | "
                     f"{int(m['num_switches'])} | {m['recall']:.3f} | {m['precision']:.3f} | "
                     f"{int(m['mostly_tracked'])} / {int(m['mostly_lost'])} of {int(m['num_unique_objects'])} | "
                     f"{r['stitches'] if lab == 'FlowSight' else '–'} |")
    if res:
        def mean(lab, k):
            return float(np.mean([r[lab][k] for r in res.values()]))
        L += ["", "| Mean | MOTA | IDF1 | ID switches (total) |", "|---|---|---|---|"]
        for lab in ("ByteTrack", "FlowSight"):
            L.append(f"| {lab} | {mean(lab, 'mota'):.3f} | {mean(lab, 'idf1'):.3f} | "
                     f"{int(sum(r[lab]['num_switches'] for r in res.values()))} |")
    L += ["", "Notes: CAVIAR annotates people as small as ~10 px wide at the far end of the corridor (and partially "
          "visible people at the frame edge); these dominate the misses. MOTA/precision/recall are identical for both "
          "rows because the boxes are the same — only the identities differ, so IDF1 and ID switches measure the "
          "stitching layer. FlowSight runs analytics at 10 FPS in production; this evaluation uses every frame."]
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    return res
