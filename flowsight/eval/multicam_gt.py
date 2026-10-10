"""Validate a cross-camera calibration against CAVIAR ground truth.

CAVIAR GT does not share IDs across views, so for every front-view GT person we find the corridor-view GT person
whose floor trajectory is closest over their co-visible frames, and report the along/across-corridor error.
Corridor boxes cut by the bottom of the frame get feet from the person's median untruncated box height (the same
repair the production foot-point estimator applies).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np


def load_cvml(path, frame_h: int = 288) -> dict[int, dict[int, tuple[float, float]]]:
    raw = defaultdict(dict)
    heights = defaultdict(list)
    for fr in ET.parse(path).getroot().iter("frame"):
        n = int(fr.get("number"))
        for o in fr.iter("object"):
            b = o.find("box")
            xc, yc, w, h = (float(b.get(k)) for k in ("xc", "yc", "w", "h"))
            raw[n][int(o.get("id"))] = (xc, yc, w, h)
            if yc + h / 2 < frame_h - 3:
                heights[int(o.get("id"))].append(h)
    med = {i: float(np.median(v)) for i, v in heights.items()}
    out = {}
    for n, objs in raw.items():
        out[n] = {}
        for i, (xc, yc, w, h) in objs.items():
            bottom = yc + h / 2
            if bottom >= frame_h - 3 and i in med and med[i] > h:
                bottom = yc - h / 2 + med[i]
            out[n][i] = (xc, bottom)
    return out


def trajectory_errors(gt_ref, gt_new, H_ref_img2floor, H_new_img2floor, offset_frames: int, min_common: int = 20):
    """Per new-camera GT person: best-matching reference person, mean |dX|, |dY| and distance (m)."""
    def floor(g, H, shift):
        tr = defaultdict(dict)
        for n, objs in g.items():
            for i, p in objs.items():
                tr[i][n - shift] = cv2.perspectiveTransform(np.array([[p]], float), H)[0, 0]
        return tr

    ref = floor(gt_ref, H_ref_img2floor, 0)
    new = floor(gt_new, H_new_img2floor, offset_frames)
    rows = []
    for i, tn in new.items():
        best = None
        for j, tr in ref.items():
            common = [n for n in tn if n in tr]
            if len(common) < min_common:
                continue
            d = np.array([tn[n] - tr[n] for n in common])
            dist = float(np.linalg.norm(d, axis=1).mean())
            if best is None or dist < best["mean_dist_m"]:
                best = {"new_id": i, "ref_id": j, "frames": len(common), "mean_dist_m": dist,
                        "abs_dx_m": float(np.abs(d[:, 0]).mean()), "abs_dy_m": float(np.abs(d[:, 1]).mean())}
        if best:
            rows.append(best)
    return rows
