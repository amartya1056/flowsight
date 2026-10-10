"""Auto-calibrate a camera (homography + clock offset) from shoppers co-observed with an already-calibrated camera.

    python scripts/autocalibrate.py --store caviar_multicam --ref cam1 --cam cam2 [--stride 2] [--write]

Writes docs/multicam_autocalib.md (+ an overlay image) and, with --write, stores the calibration and
`time_offset_frames` for the new camera in config/stores/<store>.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowsight.config import Calibration, load_config, load_store, resolve_path, save_store  # noqa: E402
from flowsight.geometry.autocalib import estimate_best, moving_points  # noqa: E402
from flowsight.eval.multicam_gt import load_cvml, trajectory_errors  # noqa: E402
from flowsight.geometry.calibration import Homography  # noqa: E402
from flowsight.perception.detector_tracker import DetectorTracker  # noqa: E402
from flowsight.privacy.anonymize import blur_heads  # noqa: E402


def detect_feet(det: DetectorTracker, path: Path, stride: int) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Foot points per frame, using the production tracker + foot-point estimator, so boxes cut off by the bottom
    of the frame get their feet from the track's median height (that is often exactly where two views overlap)."""
    from flowsight.perception.footpoint import FootpointEstimator

    cap = cv2.VideoCapture(str(path))
    out, idx, first = {}, -1, None
    fe = FootpointEstimator()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        dets = det.update(frame)
        h = frame.shape[0]
        feet = []
        for d in dets:
            fx, fy, trunc = fe(d.track_id, d.bbox_xyxy, h)
            if not trunc or fy > d.bbox_xyxy[3] + 1:  # truncated boxes only once a median height is known
                feet.append((fx, fy, d.track_id))
        out[idx] = np.array(feet, float).reshape(-1, 3)
        if first is None and len(dets) >= 2:
            first = blur_heads(frame.copy(), [d.bbox_xyxy for d in dets])
    return out, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="caviar_multicam")
    ap.add_argument("--ref", default="cam1")
    ap.add_argument("--cam", default="cam2")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    st = load_store(a.store)
    ref_cam, new_cam = st.camera(a.ref), st.camera(a.cam)
    H_ref = Homography(ref_cam.calibration.image_points, ref_cam.calibration.floor_points)
    cache = ROOT / "runs" / f"autocalib_{a.store}_{a.ref}_{a.cam}_s{a.stride}_v3.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ref_px, new_px, sample = z["ref"].item(), z["new"].item(), z["sample"]
    else:
        ref_px, _ = detect_feet(DetectorTracker(cfg.runtime), resolve_path(ref_cam.source), a.stride)
        new_px, sample = detect_feet(DetectorTracker(cfg.runtime), resolve_path(new_cam.source), a.stride)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, ref=np.array(ref_px, dtype=object), new=np.array(new_px, dtype=object), sample=sample)
    fps = 25.0 / a.stride
    ref_floor = moving_points(ref_px, H_ref.img_to_floor, 0.4, fps=fps, win=max(2, 12 // a.stride))
    new_mv = moving_points(new_px, None, 8.0, fps=fps, win=max(2, 12 // a.stride))
    print(f"walking detections: ref {sum(len(v) for v in ref_floor.values())}, new {sum(len(v) for v in new_mv.values())}")
    offs = range(-120 // a.stride * a.stride, 121, 2 * a.stride)
    r = estimate_best(ref_floor, new_mv, restarts=3, offsets=offs, refine=2 * a.stride, max_people=3)
    print(f"offset {r.offset_frames} frames ({r.offset_frames / 25:.2f} s), inliers {r.inliers}/{r.candidates}, "
          f"frames {r.frames_used}, median residual {r.median_residual_m:.3f} m")
    new_px = {k: v[:, :2] for k, v in new_px.items()}

    # second-best offset for a confidence statement
    ranked = sorted(r.scores.items(), key=lambda kv: -kv[1])
    runner = next((kv for kv in ranked if abs(kv[0] - r.offset_frames) > 4 * a.stride), ranked[1])

    all_px = np.concatenate([v for v in new_px.values() if len(v)])
    proj = cv2.perspectiveTransform(all_px[None], r.H)[0]
    W, Hm = st.floorplan.width_m, st.floorplan.height_m
    on_plan = all_px[(proj[:, 0] > 0) & (proj[:, 0] < W) & (proj[:, 1] > 0) & (proj[:, 1] < Hm)]
    img_pts, floor_pts = r.image_points(8, pts_px=on_plan if len(on_plan) >= 8 else all_px)

    # overlay: project the shared floor zones into the new camera with the recovered homography
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    if sample is not None:
        Hinv = np.linalg.inv(r.H)
        vis = sample.copy()
        for z in st.zones:
            poly = np.array(z.polygon, float)
            dense = np.concatenate([np.linspace(p, q, 12, endpoint=False) for p, q in zip(poly, np.roll(poly, -1, 0))])
            pts = cv2.perspectiveTransform(dense[None], Hinv)[0].astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(docs / "multicam_autocalib_overlay.jpg"), vis)

    gt_lines = []
    for clip, (gr, gn, start) in {"ThreePastShop1": ("c3ps1gt", "f3ps1gt", 0),
                                  "OneShopOneWait1": ("cosow1gt", "fosow1gt", 1650)}.items():
        pr, pn = ROOT / "data" / "gt" / f"{gr}.xml", ROOT / "data" / "gt" / f"{gn}.xml"
        if not (pr.exists() and pn.exists()):
            continue
        for row in trajectory_errors(load_cvml(pr), load_cvml(pn), H_ref.H, r.H, r.offset_frames):
            gt_lines.append(f"| {clip} | {row['new_id']} | {row['ref_id']} | {row['frames']} | {row['mean_dist_m']:.2f} | "
                            f"{row['abs_dx_m']:.2f} | {row['abs_dy_m']:.2f} |")
    lines = ["# Multi-camera auto-calibration (CAVIAR corridor + front view)", "",
             f"Reference camera `{a.ref}` (calibrated) and new camera `{a.cam}` watch the same corridor from opposite sides. "
             f"Every {a.stride}nd frame of both clips was run through the production detector + tracker (truncated boxes get "
             "feet from the track's median height). Only *walking* people are used (≥ 0.4 m/s in the reference camera, "
             "≥ 8 px/s in the new one): standing people fit any time offset and pull RANSAC onto degenerate solutions. "
             "Every walking foot point of the new camera is paired with every walking floor position of the reference "
             "camera at a candidate clock offset; a guarded LO-RANSAC (4 samples from 4 different frames, fold/area "
             "checks, per-detection capped score) finds the homography, and the offset with the most inliers wins "
             "(best of 3 restarts).", "",
             "| Quantity | Value |", "|---|---|",
             f"| Clock offset | **{r.offset_frames} frames = {r.offset_frames / 25:.2f} s** "
             f"(runner-up offset {runner[0]} has {runner[1]} inliers vs {r.inliers}) |",
             f"| Inlier pairs | {r.inliers} of {r.candidates} candidates from {r.frames_used} co-observed frames |",
             f"| Median residual of inliers | {r.median_residual_m:.3f} m |", "",
             "Inliers by offset (frames → inliers):", "",
             "`" + ", ".join(f"{k}:{v}" for k, v in r.scores.items()) + "`", "",
             "![overlay](multicam_autocalib_overlay.jpg)", "",
             "## Validation against CAVIAR ground truth", "",
             "CAVIAR annotates both views independently (no shared IDs). For each front-view GT person we find the "
             "corridor-view GT person whose floor trajectory is closest over their co-visible frames:", "",
             "| Clip | Front GT id | ↔ corridor GT id | co-visible frames | mean distance (m) | across (m) | along (m) |",
             "|---|---|---|---|---|---|---|"] + gt_lines + ["",
             "*The corridor zones (defined once on the shared floor plan) projected into the front camera with the "
             "recovered homography.*"]
    (docs / "multicam_autocalib.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote docs/multicam_autocalib.md")
    if a.write:
        new_cam.calibration = Calibration(image_points=img_pts, floor_points=floor_pts)
        new_cam.approximate = True
        new_cam.time_offset_frames = int(r.offset_frames)
        save_store(st)
        print("calibration written")


if __name__ == "__main__":
    main()
