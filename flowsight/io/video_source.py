"""Unified frame source: file, RTSP/HTTP URL or webcam index -> (frame_idx, timestamp_s, frame).

Timestamps use the source's real FPS for files (so analytics are reproducible and independent of
processing speed) and wall-clock for live streams. Frames are decimated to `target_fps`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from flowsight.config import resolve_path


@dataclass
class SourceInfo:
    fps: float
    width: int
    height: int
    n_frames: int
    live: bool


class VideoSource:
    def __init__(self, source: str | int, target_fps: float = 10.0, max_seconds: float | None = None,
                 start_frame: int = 0):
        self.source = source
        self.target_fps = float(target_fps)
        self.max_seconds = max_seconds
        self.start_frame = max(0, int(start_frame))  # skip this many source frames (multi-camera clock alignment)
        self.live = False
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self._open_arg: str | int = int(source)
            self.live = True
        elif isinstance(source, str) and "://" in source:
            self._open_arg = source
            self.live = True
        else:
            path = resolve_path(source)
            if not path.exists():
                raise FileNotFoundError(f"video not found: {path}")
            self._open_arg = str(path)
        self.cap = cv2.VideoCapture(self._open_arg)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not np.isfinite(fps) or fps <= 1 or fps > 240:
            fps = 25.0
        self.info = SourceInfo(
            fps=float(fps),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.live else -1,
            live=self.live,
        )

    def read_first(self) -> np.ndarray:
        ok, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ok:
            raise RuntimeError("cannot read first frame")
        return frame

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        step = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_t = 0.0
        idx = -1
        for _ in range(self.start_frame):
            if not self.cap.grab():
                break
        t0 = time.monotonic()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            ts = (time.monotonic() - t0) if self.live else idx / self.info.fps
            if self.max_seconds is not None and ts > self.max_seconds:
                break
            # keep a frame whenever its timestamp crosses the next sampling instant
            if ts + 1e-9 < next_t:
                continue
            next_t = max(next_t + step, ts) if not self.live else ts + step
            yield idx, ts, frame
        self.release()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
