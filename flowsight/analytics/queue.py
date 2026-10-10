"""Checkout queue analytics via Little's law.

For each checkout zone:
  * L  = people in the zone moving slower than the engaged threshold (queueing or being served),
         averaged over each `window_s` window
  * λ  = confirmed zone entries per second over the trailing `rate_horizon_s`
  * μ  = zone exits per second over the same horizon
  * W  = L / λ  (Little's law: mean time in the system), cross-checked against the observed dwell of
         completed checkout visits
  * trend = consecutive rising windows of L (used for "queue rising 3 windows in a row")
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class QueueWindow:
    zone_id: str
    window_start: float
    window_end: float
    queue_len: float
    arrival_rate_pm: float
    service_rate_pm: float
    predicted_wait_s: float | None
    observed_wait_s: float | None
    rising_streak: int

    def as_dict(self):
        return dict(self.__dict__)


class QueueAnalyzer:
    def __init__(self, zone_id: str, cfg, t_start: float = 0.0, on_window=None):
        self.zone_id = zone_id
        self.cfg = cfg
        self.t_start = t_start
        self.arrivals: deque[float] = deque()
        self.departures: deque[tuple[float, float]] = deque()  # (t_exit, dwell)
        self.window_idx: int | None = None
        self.l_sum = 0.0
        self.l_n = 0
        self.windows: list[QueueWindow] = []
        self.rising = 0
        self.on_window = on_window
        self.current_len = 0

    def on_entry(self, ts: float) -> None:
        self.arrivals.append(ts)

    def on_exit(self, ts: float, dwell_s: float) -> None:
        self.departures.append((ts, dwell_s))

    def sample(self, ts: float, queue_len: int) -> None:
        w = int(math.floor((ts - self.t_start) / self.cfg.window_s + 1e-9))
        if self.window_idx is None:
            self.window_idx = w
        while w > self.window_idx:
            self._close(self.window_idx)
            self.window_idx += 1
        self.l_sum += queue_len
        self.l_n += 1
        self.current_len = queue_len

    def rates(self, now: float) -> tuple[float, float, float]:
        """(lambda/s, mu/s, horizon_s) over the trailing horizon (shorter at the start of a run)."""
        h = self.cfg.rate_horizon_s
        horizon = max(min(h, now - self.t_start), 1e-6)
        t0 = now - horizon
        while self.arrivals and self.arrivals[0] < now - 4 * h:
            self.arrivals.popleft()
        while self.departures and self.departures[0][0] < now - 4 * h:
            self.departures.popleft()
        lam = sum(1 for t in self.arrivals if t >= t0) / horizon
        mu = sum(1 for t, _ in self.departures if t >= t0) / horizon
        return lam, mu, horizon

    def predicted_wait(self, queue_len: float, now: float) -> float | None:
        lam, _, horizon = self.rates(now)
        if lam <= 0:
            return None if queue_len <= 0 else float("inf")
        return queue_len / lam

    def observed_wait(self, now: float) -> float | None:
        t0 = now - self.cfg.rate_horizon_s
        d = [dw for t, dw in self.departures if t >= t0]
        return float(np.mean(d)) if d else None

    def _close(self, w: int) -> None:
        ws = self.t_start + w * self.cfg.window_s
        we = ws + self.cfg.window_s
        L = self.l_sum / self.l_n if self.l_n else 0.0
        lam, mu, _ = self.rates(we)
        W = self.predicted_wait(L, we)
        if self.windows and L > self.windows[-1].queue_len + 0.25:
            self.rising += 1
        elif self.windows and L < self.windows[-1].queue_len - 0.25:
            self.rising = 0
        win = QueueWindow(self.zone_id, ws, we, L, lam * 60, mu * 60,
                          None if W is None or not np.isfinite(W) else W, self.observed_wait(we), self.rising)
        self.windows.append(win)
        self.l_sum, self.l_n = 0.0, 0
        if self.on_window:
            self.on_window(win)

    def finalize(self) -> None:
        if self.window_idx is not None and self.l_n:
            self._close(self.window_idx)

    def status(self, now: float) -> dict:
        lam, mu, _ = self.rates(now)
        last = self.windows[-1] if self.windows else None
        L = last.queue_len if last else float(self.current_len)
        W = self.predicted_wait(L, now)
        trend = [round(w.queue_len, 2) for w in self.windows[-10:]]
        return {"zone_id": self.zone_id, "queue_len": round(L, 2), "queue_len_now": self.current_len,
                "arrival_rate_pm": round(lam * 60, 2), "service_rate_pm": round(mu * 60, 2),
                "predicted_wait_s": None if W is None or not np.isfinite(W) else round(W, 1),
                "observed_wait_s": self.observed_wait(now), "rising_streak": self.rising,
                "utilisation": (lam / mu) if mu > 0 else None, "trend": trend}
