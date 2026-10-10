"""Floor-grid heatmaps in person-seconds per cell.

Accumulation is *time-weighted* (+dt per person per frame, not +1), so the map is independent of the
frame rate and a cell's value is literally "person-seconds spent here". A second grid accumulates only
stationary time (speed < engaged threshold) — where people actually stop, not just walk.
Rolling windows are kept as one grid per minute bucket.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class Heatmap:
    def __init__(self, width_m: float, height_m: float, cell_m: float = 0.25, bucket_s: float = 60.0,
                 engaged_speed: float = 0.3, max_buckets: int = 24 * 60):
        self.w_m, self.h_m, self.cell = width_m, height_m, cell_m
        self.nx = int(math.ceil(width_m / cell_m))
        self.ny = int(math.ceil(height_m / cell_m))
        self.bucket_s = bucket_s
        self.engaged_speed = engaged_speed
        self.max_buckets = max_buckets
        self.occ = np.zeros((self.ny, self.nx), np.float64)
        self.stat = np.zeros((self.ny, self.nx), np.float64)
        self.buckets: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _cells(self, xs, ys):
        ix = np.clip((np.asarray(xs) / self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip((np.asarray(ys) / self.cell).astype(int), 0, self.ny - 1)
        valid = (np.asarray(xs) >= 0) & (np.asarray(xs) <= self.w_m) & (np.asarray(ys) >= 0) & (np.asarray(ys) <= self.h_m)
        return ix[valid], iy[valid], valid

    def update(self, ts: float, dt: float, xs, ys, speeds) -> None:
        if len(xs) == 0 or dt <= 0:
            return
        ix, iy, valid = self._cells(xs, ys)
        stationary = np.asarray(speeds)[valid] < self.engaged_speed
        b = int(ts // self.bucket_s)
        if b not in self.buckets:
            self.buckets[b] = (np.zeros_like(self.occ), np.zeros_like(self.stat))
            while len(self.buckets) > self.max_buckets:
                self.buckets.popitem(last=False)
        bo, bs = self.buckets[b]
        np.add.at(self.occ, (iy, ix), dt)
        np.add.at(bo, (iy, ix), dt)
        if stationary.any():
            np.add.at(self.stat, (iy[stationary], ix[stationary]), dt)
            np.add.at(bs, (iy[stationary], ix[stationary]), dt)

    def grid(self, kind: str = "occupancy", window_s: float | None = None, now: float | None = None) -> np.ndarray:
        idx = 0 if kind == "occupancy" else 1
        if window_s is None or now is None:
            return (self.occ if idx == 0 else self.stat).copy()
        b_now = int(now // self.bucket_s)
        n = max(1, int(math.ceil(window_s / self.bucket_s)))
        g = np.zeros_like(self.occ)
        for b, grids in self.buckets.items():
            if b_now - n < b <= b_now:
                g += grids[idx]
        return g

    def blurred(self, grid: np.ndarray, sigma_cells: float) -> np.ndarray:
        return gaussian_filter(grid, sigma_cells) if sigma_cells > 0 else grid


def colorize(grid: np.ndarray, size_wh: tuple[int, int], vmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR colour image, alpha mask 0..1) at the requested pixel size."""
    vmax = vmax if vmax is not None else (np.percentile(grid[grid > 0], 99) if (grid > 0).any() else 1.0)
    norm = np.clip(grid / max(vmax, 1e-9), 0, 1)
    img = cv2.resize((norm * 255).astype(np.uint8), size_wh, interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    alpha = np.clip(img.astype(np.float32) / 255.0 * 1.6, 0, 0.85)
    alpha[img < 8] = 0
    return color, alpha


def overlay(base: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return (base.astype(np.float32) * (1 - a) + color.astype(np.float32) * a).astype(np.uint8)


def heatmap_on_floorplan(grid: np.ndarray, floorplan_bgr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    g = gaussian_filter(grid, sigma) if sigma > 0 else grid
    h, w = floorplan_bgr.shape[:2]
    color, alpha = colorize(g, (w, h))
    return overlay(floorplan_bgr, color, alpha)


def heatmap_on_camera(grid: np.ndarray, frame_bgr: np.ndarray, floor_to_img: np.ndarray, width_m: float,
                      height_m: float, sigma: float = 1.5) -> np.ndarray:
    """Warp the floor heatmap into the camera view using the floor->image homography."""
    g = gaussian_filter(grid, sigma) if sigma > 0 else grid
    gh, gw = g.shape
    # grid pixel -> floor metres
    S = np.array([[width_m / gw, 0, 0], [0, height_m / gh, 0], [0, 0, 1]], np.float64)
    M = floor_to_img @ S
    norm = np.clip(g / max(np.percentile(g[g > 0], 99) if (g > 0).any() else 1.0, 1e-9), 0, 1)
    h, w = frame_bgr.shape[:2]
    warped = cv2.warpPerspective((norm * 255).astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(warped, cv2.COLORMAP_TURBO)
    alpha = np.clip(warped.astype(np.float32) / 255.0 * 1.6, 0, 0.7)
    alpha[warped < 8] = 0
    return overlay(frame_bgr, color, alpha)
