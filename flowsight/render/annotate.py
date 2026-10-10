"""Overlays: zones coloured by live LoS, tracks with trails, ID + live dwell, heatmaps, pulsing
bottlenecks. Two views: the camera view (zones projected back with floor->image homography) and the
top-down floor-plan view (used by the simulator and the floor-plan page)."""
from __future__ import annotations

import math
from collections import defaultdict, deque

import cv2
import numpy as np

LOS_BGR = {"A": (113, 204, 46), "B": (80, 200, 140), "C": (40, 200, 230), "D": (30, 150, 245),
           "E": (40, 80, 235), "F": (40, 30, 200)}
ZONE_TYPE_BGR = {"entrance": (200, 160, 60), "aisle": (150, 150, 150), "promo": (200, 90, 200),
                 "walkway": (120, 120, 100), "checkout": (60, 170, 240), "service": (200, 200, 80)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _id_color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    c = rng.integers(80, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def _label(img, text, org, scale=0.45, color=(255, 255, 255), bg=(20, 20, 20)):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def make_floorplan_image(store, px_per_m: float = 50) -> np.ndarray:
    """Draw a clean floor plan (background, fixtures, doors, zone labels) as a BGR image."""
    W = int(round(store.floorplan.width_m * px_per_m))
    H = int(round(store.floorplan.height_m * px_per_m))
    img = np.full((H, W, 3), (34, 30, 28), np.uint8)
    # floor grid every metre
    for x in range(0, W, int(px_per_m)):
        cv2.line(img, (x, 0), (x, H), (44, 40, 38), 1)
    for y in range(0, H, int(px_per_m)):
        cv2.line(img, (0, y), (W, y), (44, 40, 38), 1)
    cv2.rectangle(img, (1, 1), (W - 2, H - 2), (120, 115, 110), 2)
    for f in store.fixtures:
        x, y, w, h = [v * px_per_m for v in f.rect]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (78, 72, 68), -1)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (110, 104, 98), 1)
    for d in store.doors:
        x, y = d.at[0] * px_per_m, d.at[1] * px_per_m
        cv2.circle(img, (int(x), int(y)), 9, (90, 200, 255), -1)
        _label(img, d.name, (x + 12, y + 5), 0.45, (90, 200, 255), (34, 30, 28))
    for z in store.zones:
        pts = (np.array(z.polygon) * px_per_m).astype(np.int32)
        cv2.polylines(img, [pts], True, ZONE_TYPE_BGR.get(z.type, (150, 150, 150)), 1, cv2.LINE_AA)
    return img


class Renderer:
    def __init__(self, store, cfg, mapping=None, floorplan_img: np.ndarray | None = None):
        self.store, self.cfg = store, cfg
        self.mapping = mapping
        self.ppm = cfg.render.floor_px_per_m
        self.floorplan = floorplan_img if floorplan_img is not None else make_floorplan_image(store, self.ppm)
        self.fp_scale = self.floorplan.shape[1] / store.floorplan.width_m
        self.pixel_trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=int(cfg.render.trail_s * cfg.runtime.target_fps) + 1))
        self._zone_img_polys: dict[str, np.ndarray] | None = None

    # -------------------------------------------------------------- helpers
    def _zone_polys_img(self) -> dict[str, np.ndarray]:
        if self._zone_img_polys is None:
            out = {}
            for z in self.store.zones:
                # densify edges so projected polygons stay correct under perspective
                poly = np.array(z.polygon, float)
                dense = []
                for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                    for t in np.linspace(0, 1, 12, endpoint=False):
                        dense.append(a + (b - a) * t)
                out[z.id] = self.mapping.floor_to_img(np.array(dense)).astype(np.int32)
            self._zone_img_polys = out
        return self._zone_img_polys

    def invalidate_zones(self):
        self._zone_img_polys = None

    @staticmethod
    def _pulse(t: float) -> float:
        return 0.5 + 0.5 * math.sin(t * 6.0)

    # --------------------------------------------------------- camera view
    def draw_camera(self, frame: np.ndarray, dets_by_track: dict[int, tuple], hub, t: float,
                    heat_grid: np.ndarray | None = None) -> np.ndarray:
        """dets_by_track: stable track id -> (bbox_xyxy, foot_px). Frame should already be privacy-blurred."""
        img = frame
        if heat_grid is not None and self.mapping is not None and getattr(self.mapping, "calibrated", False):
            from flowsight.analytics.heatmap import heatmap_on_camera

            img = heatmap_on_camera(heat_grid, img, self.mapping.floor_to_img_matrix(),
                                    self.store.floorplan.width_m, self.store.floorplan.height_m,
                                    self.cfg.geometry.heatmap_blur_sigma_cells)
        if self.mapping is not None and getattr(self.mapping, "calibrated", False):
            overlay = img.copy()
            polys = self._zone_polys_img()
            live = {z["id"]: z for z in hub.live_zones()} if hub else {}
            for z in self.store.zones:
                lz = live.get(z.id, {})
                col = LOS_BGR.get(lz.get("los", "A"), (150, 150, 150))
                cv2.fillPoly(overlay, [polys[z.id]], col)
            img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
            for z in self.store.zones:
                lz = live.get(z.id, {})
                if lz.get("bottleneck"):
                    p = self._pulse(t)
                    cv2.polylines(img, [polys[z.id]], True, (40, 40, int(150 + 105 * p)), 2 + int(3 * p), cv2.LINE_AA)
                else:
                    cv2.polylines(img, [polys[z.id]], True, ZONE_TYPE_BGR.get(z.type), 1, cv2.LINE_AA)
                c = polys[z.id].mean(axis=0)
                if 0 <= c[0] < img.shape[1] and 0 <= c[1] < img.shape[0]:
                    _label(img, f"{z.name} {lz.get('occupancy', 0)}", (c[0] - 30, c[1]), 0.38)
        dwell = self._live_dwell(hub)
        for tid, (bbox, foot) in dets_by_track.items():
            col = _id_color(tid)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            tr = self.pixel_trails[tid]
            tr.append((int(foot[0]), int(foot[1])))
            if len(tr) > 1:
                cv2.polylines(img, [np.array(tr, np.int32)], False, col, 2, cv2.LINE_AA)
            is_staff = hub is not None and hub.staff.is_staff(tid)
            if is_staff:
                col = (150, 150, 150)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            txt = "STAFF" if is_staff else f"#{tid}" + (f" {dwell[tid]:.0f}s" if tid in dwell else "")
            _label(img, txt, (x1, max(12, y1 - 4)), 0.45, (255, 255, 255), col)
        for tid in [k for k in self.pixel_trails if k not in dets_by_track]:
            del self.pixel_trails[tid]
        return img

    @staticmethod
    def _live_dwell(hub) -> dict[int, float]:
        if hub is None:
            return {}
        out = {}
        for v in hub.visits.open_visits():
            if hub.ztype.get(v.zone_id) == "walkway":
                continue
            out[v.track_id] = max(out.get(v.track_id, 0.0), v.dwell_s)
        return out

    # ----------------------------------------------------------- floor view
    def draw_floor(self, hub, t: float, heat: str | None = None, title: str | None = None) -> np.ndarray:
        img = self.floorplan.copy()
        s = self.fp_scale
        if heat and hub is not None:
            from flowsight.analytics.heatmap import heatmap_on_floorplan

            img = heatmap_on_floorplan(hub.heat.grid(heat), img, self.cfg.geometry.heatmap_blur_sigma_cells)
        live = {z["id"]: z for z in hub.live_zones()} if hub is not None else {}
        overlay = img.copy()
        for z in self.store.zones:
            pts = (np.array(z.polygon) * s).astype(np.int32)
            lz = live.get(z.id, {})
            cv2.fillPoly(overlay, [pts], LOS_BGR.get(lz.get("los", "A")))
        img = cv2.addWeighted(overlay, 0.16, img, 0.84, 0)
        for z in self.store.zones:
            pts = (np.array(z.polygon) * s).astype(np.int32)
            lz = live.get(z.id, {})
            if lz.get("bottleneck"):
                p = self._pulse(t)
                cv2.polylines(img, [pts], True, (40, 40, int(150 + 105 * p)), 2 + int(3 * p), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True, ZONE_TYPE_BGR.get(z.type), 1, cv2.LINE_AA)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            _label(img, z.name, (x + 3, y + 13), 0.36, (230, 230, 230), (40, 36, 34))
        if hub is not None:
            dwell = self._live_dwell(hub)
            for tid, tr in hub.trails.items():
                if len(tr) > 1:
                    cv2.polylines(img, [(np.array(tr) * s).astype(np.int32)], False, _id_color(tid), 1, cv2.LINE_AA)
            for p in hub.people():
                c = (int(p["x"] * s), int(p["y"] * s))
                col = (80, 220, 255) if p["speed"] < self.cfg.visits.engaged_speed_mps else (255, 220, 120)
                cv2.circle(img, c, 6, col, -1, cv2.LINE_AA)
                cv2.circle(img, c, 6, (20, 20, 20), 1, cv2.LINE_AA)
                if p["id"] in dwell and dwell[p["id"]] >= 10:
                    cv2.putText(img, f"{dwell[p['id']]:.0f}s", (c[0] + 7, c[1] - 5), FONT, 0.33, (230, 230, 230), 1, cv2.LINE_AA)
        if title:
            _label(img, title, (8, img.shape[0] - 10), 0.5, (255, 255, 255), (60, 50, 40))
        return img

    def highlight_zone(self, img: np.ndarray, zone_id: str, camera: bool) -> np.ndarray:
        img = img.copy()
        z = self.store.zone(zone_id)
        if camera:
            if self.mapping is None or not getattr(self.mapping, "calibrated", False):
                return img
            pts = self._zone_polys_img()[zone_id]
        else:
            pts = (np.array(z.polygon) * self.fp_scale).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 4, cv2.LINE_AA)
        c = pts.mean(axis=0)
        _label(img, z.name, (c[0] - 30, c[1]), 0.6, (255, 255, 255), (0, 0, 200))
        return img
