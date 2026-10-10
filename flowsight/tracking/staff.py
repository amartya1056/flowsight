"""Staff filtering: keep employees out of shopper analytics.

Two independent, privacy-preserving cues (no faces, no identity):
  * long presence — a track on the floor longer than `staff_long_presence_s` (shoppers rarely stay that long)
  * uniform colour — optional HSV ranges for the store's uniform; a track is staff when, over at least
    `staff_color_min_samples` observations, the mean fraction of torso pixels inside a uniform range is
    >= `staff_color_min_frac`. Only colour statistics of the torso are used, never appearance features.

Once a track is flagged, the analytics hub purges its visits/exposures retroactively (see AnalyticsHub.mark_staff).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _ColorStat:
    n: int = 0
    frac_sum: float = 0.0


class StaffClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.staff: dict[int, str] = {}  # track id -> reason
        self.color: dict[int, _ColorStat] = {}
        self.ranges = [np.array(r, dtype=np.int32) for r in (cfg.staff_uniform_hsv or [])]

    # ------------------------------------------------------------- cues
    def check_presence(self, track_id: int, age_s: float) -> bool:
        if track_id not in self.staff and age_s >= self.cfg.staff_long_presence_s:
            self.staff[track_id] = "long_presence"
            return True
        return False

    def torso_fraction(self, frame: np.ndarray, bbox) -> float | None:
        """Fraction of torso pixels (25-60% of box height, middle 50% of width) inside any uniform HSV range."""
        if not self.ranges:
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = y2 - y1, x2 - x1
        tx1, tx2 = max(0, x1 + w // 4), min(frame.shape[1], x2 - w // 4)
        ty1, ty2 = max(0, y1 + int(0.25 * h)), min(frame.shape[0], y1 + int(0.60 * h))
        if tx2 - tx1 < 3 or ty2 - ty1 < 3:
            return None
        hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], bool)
        for r in self.ranges:  # [h_lo, h_hi, s_lo, s_hi, v_lo, v_hi]
            lo = np.array([r[0], r[2], r[4]], np.uint8)
            hi = np.array([r[1], r[3], r[5]], np.uint8)
            mask |= cv2.inRange(hsv, lo, hi) > 0
        return float(mask.mean())

    def observe_color(self, track_id: int, frame: np.ndarray, bbox) -> bool:
        """Accumulate uniform-colour evidence; returns True when the track has just been flagged as staff."""
        if track_id in self.staff:
            return False
        f = self.torso_fraction(frame, bbox)
        if f is None:
            return False
        st = self.color.setdefault(track_id, _ColorStat())
        st.n += 1
        st.frac_sum += f
        if st.n >= self.cfg.staff_color_min_samples and st.frac_sum / st.n >= self.cfg.staff_color_min_frac:
            self.staff[track_id] = "uniform_colour"
            return True
        return False

    def is_staff(self, track_id: int) -> bool:
        return track_id in self.staff
