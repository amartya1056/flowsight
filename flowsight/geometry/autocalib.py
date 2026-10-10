"""Self-calibration of a new camera from people seen by an already-calibrated camera.

Two cameras watching overlapping floor see the same shoppers at the same moments. For every frame we pair every
foot point of the new camera (pixels) with every floor position from the reference camera (metres); wrong pairs
are outliers, and the right ones are all explained by a single homography. Because CCTV clocks are rarely aligned
we also search the time offset: the right offset explains the most pairs.

A plain RANSAC homography is not enough: with many-to-many candidate pairs it happily finds *degenerate* models
that fold the new camera's whole view onto one busy floor spot. This module therefore runs its own RANSAC:
  * minimal samples are drawn from 4 *different* frames (people at 4 different places),
  * models are rejected unless they map the new camera's detection region to a non-folded floor area of
    plausible size (orientation-preserving at every hull vertex, area within [min_area, max_area] m²),
  * the score counts inliers capped at one per frame per new-camera detection, so a single crowded frame
    cannot dominate.
No calibration target, no survey, no manual clicking: a few minutes of normal traffic is enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class AutoCalibResult:
    offset_frames: int
    H: np.ndarray            # new-camera image px -> floor metres
    inliers: int
    candidates: int
    frames_used: int
    median_residual_m: float
    scores: dict[int, int] = field(default_factory=dict)

    def image_points(self, n: int = 8, seed: int = 0, pts_px: np.ndarray | None = None):
        """Representative (image, floor) point pairs to store in the YAML calibration format."""
        rng = np.random.default_rng(seed)
        src = pts_px[rng.choice(len(pts_px), size=min(n, len(pts_px)), replace=False)]
        dst = cv2.perspectiveTransform(src[None].astype(np.float64), self.H)[0]
        return src.round(1).tolist(), dst.round(3).tolist()


def _pairs(ref: dict[int, np.ndarray], new: dict[int, np.ndarray], offset: int, max_people: int):
    src, dst, fid, nid = [], [], [], []
    frames = 0
    for f, px in new.items():
        fl = ref.get(f - offset)
        if fl is None or not len(fl) or not len(px) or len(px) > max_people or len(fl) > max_people:
            continue
        for i, p in enumerate(px):
            for q in fl:
                src.append(p)
                dst.append(q)
                fid.append(frames)
                nid.append(frames * 64 + i)  # unique id of the new-camera detection
        frames += 1
    return (np.asarray(src, np.float64).reshape(-1, 2), np.asarray(dst, np.float64).reshape(-1, 2),
            np.asarray(fid), np.asarray(nid), frames)


def _apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    w = p[:, 2:3]
    return p[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w), w[:, 0]


def _valid(H, hull_px, min_area, max_area) -> bool:
    proj, w = _apply(H, hull_px)
    if np.any(w <= 0) and np.any(w >= 0):  # region straddles the horizon line -> folded
        return False
    area = 0.5 * abs(np.dot(proj[:, 0], np.roll(proj[:, 1], 1)) - np.dot(proj[:, 1], np.roll(proj[:, 0], 1)))
    if not (min_area <= area <= max_area):
        return False
    return cv2.isContourConvex(proj.astype(np.float32).reshape(-1, 1, 2))


def _score(H, src, dst, nid, thresh):
    proj, _ = _apply(H, src)
    res = np.linalg.norm(proj - dst, axis=1)
    ok = res < thresh
    return len(np.unique(nid[ok])), ok, res


def ransac_offset(src, dst, fid, nid, hull_px, thresh, iters, min_area, max_area, rng):
    best = (0, None)
    n_frames = fid.max() + 1 if len(fid) else 0
    if n_frames < 4:
        return best
    by_frame = [np.flatnonzero(fid == f) for f in range(n_frames)]
    for _ in range(iters):
        fr = rng.choice(n_frames, 4, replace=False)
        idx = np.array([rng.choice(by_frame[f]) for f in fr])
        s4, d4 = src[idx].astype(np.float32), dst[idx].astype(np.float32)
        if cv2.contourArea(cv2.convexHull(s4)) < 150 or cv2.contourArea(cv2.convexHull(d4)) < 0.3:
            continue
        H = cv2.getPerspectiveTransform(s4, d4)
        if not _valid(H, hull_px, min_area, max_area):
            continue
        sc, ok, _ = _score(H, src, dst, nid, thresh)
        if sc > best[0]:
            best = (sc, H)
            # LO-RANSAC: refit on the inliers until the score stops improving
            for _ in range(5):
                if ok.sum() < 6:
                    break
                H2, _ = cv2.findHomography(src[ok], dst[ok], 0)
                if H2 is None or not _valid(H2, hull_px, min_area, max_area):
                    break
                sc2, ok2, _ = _score(H2, src, dst, nid, thresh)
                if sc2 <= best[0]:
                    break
                best, ok = (sc2, H2), ok2
    return best


def estimate(ref_floor: dict[int, np.ndarray], new_px: dict[int, np.ndarray], offsets=range(-100, 101, 4),
             thresh_m: float = 0.4, max_people: int = 4, refine: int = 4, iters: int = 3000,
             min_area_m2: float = 2.0, max_area_m2: float = 400.0, seed: int = 0) -> AutoCalibResult:
    """ref_floor: frame -> (N,2) floor metres from the calibrated camera; new_px: frame -> (M,2) foot pixels."""
    rng = np.random.default_rng(seed)
    allpx = np.concatenate([v for v in new_px.values() if len(v)]).astype(np.float32)
    hull = cv2.convexHull(allpx).reshape(-1, 2).astype(np.float64)

    cache = {}

    def run(off):
        src, dst, fid, nid, frames = _pairs(ref_floor, new_px, off, max_people)
        sc, H = ransac_offset(src, dst, fid, nid, hull, thresh_m, iters, min_area_m2, max_area_m2, rng) \
            if len(src) >= 12 else (0, None)
        cache[off] = (sc, H, src, dst, fid, nid, frames)
        return sc

    scores = {off: run(off) for off in offsets}
    best = max(scores, key=scores.get)
    for off in range(best - refine, best + refine + 1):
        if off not in scores:
            scores[off] = run(off)
    best = max(scores, key=scores.get)
    sc, H, src, dst, fid, nid, frames = cache[best]
    if H is None:
        raise RuntimeError("auto-calibration failed: no valid homography (is there overlap between the cameras?)")
    # least-squares refinement on inliers (iterate twice, keep only if still valid)
    for _ in range(2):
        _, ok, _ = _score(H, src, dst, nid, thresh_m)
        if ok.sum() < 8:
            break
        H2, _ = cv2.findHomography(src[ok], dst[ok], 0)
        if H2 is None or not _valid(H2, hull, min_area_m2, max_area_m2):
            break
        H = H2
    n_in, ok, res = _score(H, src, dst, nid, thresh_m)
    return AutoCalibResult(best, H, int(n_in), len(src), frames,
                           float(np.median(res[ok])) if ok.any() else float("nan"), dict(sorted(scores.items())))


def moving_points(dets: dict[int, np.ndarray], to_floor=None, min_speed: float = 0.4, fps: float = 25.0,
                  win: int = 12) -> dict[int, np.ndarray]:
    """Keep only detections of people who are walking.

    dets: frame -> (N,3) [x_px, y_px, track_id]. Speed is measured over +-win frames of the same track, in metres
    when `to_floor` is given (reference camera) else in pixels (new camera). Standing people fit *any* time offset
    and attract degenerate homographies onto one spot; walking people pin down both the offset and the geometry.
    """
    from collections import defaultdict

    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for n, arr in dets.items():
        for x, y, t in np.asarray(arr).reshape(-1, 3):
            tracks[int(t)][n] = np.array([x, y], float)
    out = defaultdict(list)
    for pts in tracks.values():
        frames = sorted(pts)
        P = {n: (to_floor(pts[n][None])[0] if to_floor else pts[n]) for n in frames}
        for i, n in enumerate(frames):
            nb = [m for m in frames[max(0, i - win):i + win + 1] if abs(m - n) <= win]
            if len(nb) < 4 or nb[-1] == nb[0]:
                continue
            if np.linalg.norm(P[nb[-1]] - P[nb[0]]) / ((nb[-1] - nb[0]) / fps) >= min_speed:
                out[n].append(P[n])
    return {n: np.array(v) for n, v in out.items()}


def estimate_best(ref_floor, new_px, restarts: int = 3, **kw) -> AutoCalibResult:
    """Run `estimate` with several RANSAC seeds and keep the solution with the most inliers."""
    best = None
    for seed in range(restarts):
        r = estimate(ref_floor, new_px, seed=seed, **kw)
        if best is None or r.inliers > best.inliers:
            best = r
    return best
