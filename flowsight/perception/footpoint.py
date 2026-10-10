"""Bounding box -> ground-contact point.

The ground point is the bottom-centre of the box. When the box touches the bottom edge of the frame
the feet are not visible (`truncated`); we then estimate the foot point from the box top plus the
median *untruncated* box height observed for that track.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


def foot_point(bbox, frame_h: int, median_h: float | None = None, margin: int = 2):
    """Return (x, y, truncated) for one xyxy box."""
    x1, y1, x2, y2 = map(float, bbox)
    cx = 0.5 * (x1 + x2)
    truncated = y2 >= frame_h - margin
    if truncated and median_h is not None and median_h > (y2 - y1):
        return cx, y1 + median_h, True
    return cx, y2, truncated


class FootpointEstimator:
    """Keeps per-track box-height history so truncated boxes can be completed."""

    def __init__(self, history: int = 30, margin: int = 2):
        self.heights: dict[int, deque] = defaultdict(lambda: deque(maxlen=history))
        self.margin = margin

    def __call__(self, track_id: int, bbox, frame_h: int):
        x1, y1, x2, y2 = map(float, bbox)
        truncated = y2 >= frame_h - self.margin
        if not truncated:
            self.heights[track_id].append(y2 - y1)
        hist = self.heights.get(track_id)
        med = float(np.median(hist)) if hist else None
        return foot_point(bbox, frame_h, med, self.margin)

    def forget(self, track_id: int) -> None:
        self.heights.pop(track_id, None)
