"""Image <-> floor mapping.

* `Homography` — planar homography from >= 4 point pairs (RANSAC when >= 5), with reprojection error.
* `AutoScale`  — fallback when a camera has no calibration: learns metres-per-pixel as a function of
  image row from tracked person box heights (median box height ~= assumed person height). Gives
  approximate metric speeds/positions; runs are flagged "uncalibrated".
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _as_pts(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    return a.reshape(-1, 2)


class Homography:
    calibrated = True

    def __init__(self, image_points, floor_points):
        ip, fp = _as_pts(image_points), _as_pts(floor_points)
        if len(ip) < 4 or len(ip) != len(fp):
            raise ValueError("need >= 4 matching point pairs")
        method = cv2.RANSAC if len(ip) >= 5 else 0
        H, _ = cv2.findHomography(ip, fp, method, 0.05)
        if H is None:
            raise ValueError("degenerate calibration points")
        self.H = H
        self.Hinv = np.linalg.inv(H)
        self.image_points, self.floor_points = ip, fp

    @staticmethod
    def _apply(M: np.ndarray, pts) -> np.ndarray:
        p = _as_pts(pts)
        if len(p) == 0:
            return p
        return cv2.perspectiveTransform(p[None, :, :], M)[0]

    def img_to_floor(self, pts) -> np.ndarray:
        return self._apply(self.H, pts)

    def floor_to_img(self, pts) -> np.ndarray:
        return self._apply(self.Hinv, pts)

    def reprojection_error(self) -> dict:
        """Mean/max error in floor metres and in image pixels over the calibration points."""
        fe = np.linalg.norm(self.img_to_floor(self.image_points) - self.floor_points, axis=1)
        ie = np.linalg.norm(self.floor_to_img(self.floor_points) - self.image_points, axis=1)
        return {"floor_m_mean": float(fe.mean()), "floor_m_max": float(fe.max()),
                "image_px_mean": float(ie.mean()), "image_px_max": float(ie.max())}

    def floor_to_img_matrix(self) -> np.ndarray:
        return self.Hinv


class AutoScale:
    """Approximate metric mapping from person box heights.

    Model: box height h(y) = a*y + b (linear in the image row of the feet). Metres-per-pixel at row y is
    m(y) = H_person / h(y). Floor X = (x - cx) * m(y); floor Y = integral of m over rows (closed form
    H/a * ln(a*y + b)), measured from the top of the image. This is a ground-plane approximation good
    enough for speeds and relative positions, not for a floor plan overlay.
    """

    calibrated = False

    def __init__(self, person_height_m: float = 1.7, image_width: int = 1280, min_samples: int = 30):
        self.Hp = person_height_m
        self.cx = image_width / 2.0
        self.min_samples = min_samples
        self.samples: list[tuple[float, float]] = []
        self.a: float | None = None
        self.b: float | None = None

    @property
    def ready(self) -> bool:
        return self.a is not None

    def add_box(self, bbox, truncated: bool = False) -> None:
        if truncated:
            return
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h > 10:
            self.samples.append((float(y2), float(h)))
            if len(self.samples) >= self.min_samples and (len(self.samples) % 25 == 0 or self.a is None):
                self.fit()

    def fit(self) -> tuple[float, float]:
        s = np.asarray(self.samples[-5000:])
        y, h = s[:, 0], s[:, 1]
        # robust linear fit: fit, drop residual outliers (sitting people, partial boxes), refit
        A = np.vstack([y, np.ones_like(y)]).T
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        res = h - A @ coef
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        keep = np.abs(res) < 3 * 1.4826 * mad
        if keep.sum() >= 4:
            coef, *_ = np.linalg.lstsq(A[keep], h[keep], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if abs(a) < 1e-6:
            a = 1e-6
        self.a, self.b = a, b
        return a, b

    def mpp(self, y) -> np.ndarray:
        h = np.maximum(self.a * np.asarray(y, dtype=np.float64) + self.b, 5.0)
        return self.Hp / h

    def img_to_floor(self, pts) -> np.ndarray:
        p = _as_pts(pts)
        if not self.ready or len(p) == 0:
            return p * 0.01  # crude 1 cm/px guess until fitted
        x, y = p[:, 0], p[:, 1]
        X = (x - self.cx) * self.mpp(y)
        arg = np.maximum(self.a * y + self.b, 5.0)
        arg0 = max(self.b, 5.0)
        Y = self.Hp / self.a * (np.log(arg) - np.log(arg0))
        return np.stack([X, Y], axis=1)

    def floor_to_img(self, pts) -> np.ndarray:
        P = _as_pts(pts)
        if not self.ready or len(P) == 0:
            return P * 100.0
        X, Y = P[:, 0], P[:, 1]
        arg0 = max(self.b, 5.0)
        arg = np.exp(Y * self.a / self.Hp + np.log(arg0))
        y = (arg - self.b) / self.a
        x = X / self.mpp(y) + self.cx
        return np.stack([x, y], axis=1)


def build_mapping(camera_cfg, person_height_m: float, image_width: int):
    """Homography if the camera is calibrated, else AutoScale fallback (logged)."""
    if camera_cfg is not None and camera_cfg.calibration is not None:
        c = camera_cfg.calibration
        return Homography(c.image_points, c.floor_points)
    log.warning("camera %s has no calibration: running UNCALIBRATED (auto-scale from person height)",
                getattr(camera_cfg, "id", "?"))
    return AutoScale(person_height_m, image_width)
