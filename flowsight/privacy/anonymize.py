"""Privacy-by-design: blur the head region of every person box on any frame that leaves the pipeline.

No face detection, no face recognition, no appearance re-identification. The head region is simply the
top `head_fraction` (default 22%) of each person box, irreversibly blurred (pixelate + Gaussian).
"""
from __future__ import annotations

import cv2
import numpy as np


def blur_heads(frame: np.ndarray, boxes, head_fraction: float = 0.22) -> np.ndarray:
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        bh = y2 - y1
        hx1 = max(0, x1 - int(0.05 * (x2 - x1)))
        hx2 = min(w, x2 + int(0.05 * (x2 - x1)))
        hy1 = max(0, y1 - int(0.03 * bh))
        hy2 = min(h, y1 + int(head_fraction * bh))
        if hx2 - hx1 < 2 or hy2 - hy1 < 2:
            continue
        roi = frame[hy1:hy2, hx1:hx2]
        rh, rw = roi.shape[:2]
        # pixelate to ~6 blocks across, then smooth: irreversible at any resolution
        small = cv2.resize(roi, (max(1, rw // max(1, rw // 6)), max(1, rh // max(1, rw // 6))),
                           interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        k = max(3, (min(rw, rh) // 2) | 1)
        frame[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame
