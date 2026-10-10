"""Rolling per-zone, per-metric baselines and anomaly scores ("3.2x normal")."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class RollingBaseline:
    def __init__(self, history: int = 360, min_samples: int = 24, min_std: float = 0.5):
        self.history = history
        self.min_samples = min_samples
        self.min_std = min_std
        self.data: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=self.history))

    def add(self, zone_id: str, metric: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            return
        self.data[(zone_id, metric)].append(float(value))

    def stats(self, zone_id: str, metric: str, exclude_last: int = 0) -> tuple[float, float, int] | None:
        d = self.data.get((zone_id, metric))
        if not d:
            return None
        arr = np.fromiter(d, float)
        if exclude_last:
            arr = arr[:-exclude_last]
        if arr.size < self.min_samples:
            return None
        return float(arr.mean()), float(max(arr.std(), self.min_std)), int(arr.size)

    def score(self, zone_id: str, metric: str, value: float, exclude_last: int = 0) -> dict | None:
        """z-score and ratio of `value` against the zone's own history (optionally excluding the newest
        `exclude_last` samples, so an ongoing anomaly does not inflate its own baseline)."""
        st = self.stats(zone_id, metric, exclude_last)
        if st is None:
            return None
        mean, std, n = st
        return {"z": (value - mean) / std, "ratio": (value / mean) if mean > 1e-9 else None,
                "mean": mean, "std": std, "n": n}
