"""Zone polygons in floor metres, exposure buffers and fast point-in-zone queries.

Zones may overlap (e.g. a walkway and an aisle mouth); a point can be in several zones at once.
The *exposure area* of a zone is the zone dilated by `exposure_buffer_m`: anyone inside it has had
the chance to see the zone (used for the exposure -> stop funnel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.prepared import prep


@dataclass
class Zone:
    id: str
    name: str
    type: str
    polygon: Polygon
    exposure_area: Polygon  # polygon dilated by the buffer (includes the polygon itself)
    exposure_ring: Polygon  # buffer minus the polygon

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return float(c.x), float(c.y)

    def coords(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in list(self.polygon.exterior.coords)[:-1]]


class ZoneSet:
    def __init__(self, zone_cfgs, exposure_buffer_m: float = 1.5):
        self.zones: list[Zone] = []
        for zc in zone_cfgs:
            poly = Polygon(zc.polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)
            exp = poly.buffer(exposure_buffer_m, join_style=2)
            self.zones.append(Zone(zc.id, zc.name, zc.type, poly, exp, exp.difference(poly)))
        self.by_id = {z.id: z for z in self.zones}
        self.ids = [z.id for z in self.zones]
        for z in self.zones:
            shapely.prepare(z.polygon)
            shapely.prepare(z.exposure_area)
        self._prepared = [prep(z.polygon) for z in self.zones]

    def __iter__(self):
        return iter(self.zones)

    def __len__(self):
        return len(self.zones)

    def __getitem__(self, zid: str) -> Zone:
        return self.by_id[zid]

    def membership(self, xs, ys) -> np.ndarray:
        """Boolean matrix [n_points, n_zones]: point inside zone polygon."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.size == 0:
            return np.zeros((0, len(self.zones)), bool)
        return np.stack([shapely.contains_xy(z.polygon, xs, ys) for z in self.zones], axis=1)

    def exposure_membership(self, xs, ys) -> np.ndarray:
        """Boolean matrix [n_points, n_zones]: point inside zone OR its exposure buffer."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if xs.size == 0:
            return np.zeros((0, len(self.zones)), bool)
        return np.stack([shapely.contains_xy(z.exposure_area, xs, ys) for z in self.zones], axis=1)

    def zones_at(self, x: float, y: float) -> list[str]:
        m = self.membership([x], [y])[0]
        return [self.ids[i] for i in np.flatnonzero(m)]

    def of_type(self, *types: str) -> list[Zone]:
        return [z for z in self.zones if z.type in types]
