"""Multi-camera fusion in shared floor coordinates (no appearance features, privacy-preserving).

Every camera projects its detections onto the same floor plan, so the same shopper seen by two cameras lands on
(roughly) the same floor point. Per frame we:
  1. cluster observations across cameras: camera by camera, Hungarian assignment of its observations to the
     existing clusters on floor distance (gate `gate_m`); observations already sharing a fused key get a bonus,
  2. give each cluster a *fused key*, inherited from any member that already has one (oldest wins), otherwise new,
  3. remember (camera, raw tracker id) -> fused key.
Step 3 is the hand-off: while a shopper is in the overlap, camera B's tracker ID is bound to the key that camera
A's track already carries, so when A loses the shopper, B's observations continue the *same* global track.
Fused positions are the mean of member observations; the result feeds one global TrackStore (smoothing,
de-biased speed, ID stitching across short gaps between non-overlapping views).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FusionStats:
    frames: int = 0
    cross_matches: int = 0          # observation pairs merged across cameras
    keys_created: int = 0
    key_cams: dict = field(default_factory=dict)   # key -> set of cameras that ever saw it
    key_first_cam: dict = field(default_factory=dict)
    key_last_cams: dict = field(default_factory=dict)
    match_dist: list = field(default_factory=list)

    def summary(self) -> dict:
        multi = [k for k, c in self.key_cams.items() if len(c) >= 2]
        handoffs = [k for k in multi if self.key_first_cam[k] not in self.key_last_cams.get(k, set())]
        md = np.array(self.match_dist[-20000:]) if self.match_dist else np.array([])
        per_cam: dict = {}
        for cams in self.key_cams.values():
            for c in cams:
                per_cam[c] = per_cam.get(c, 0) + 1
        return {"fused_keys": self.keys_created, "identities_seen_per_camera": per_cam, "cross_camera_keys": len(multi), "handoffs": len(handoffs),
                "cross_matches": self.cross_matches,
                "median_cross_dist_m": float(np.median(md)) if md.size else None}


class CrossCameraFuser:
    def __init__(self, cam_ids: list[str], gate_m: float = 1.0, key_bonus_m: float = 0.5):
        self.cams = list(cam_ids)
        self.gate = gate_m
        self.bonus = key_bonus_m
        self.key_of: dict[tuple[str, object], int] = {}
        self._next = 1
        self.stats = FusionStats()

    def _new_key(self) -> int:
        k = self._next
        self._next += 1
        self.stats.keys_created += 1
        return k

    def fuse(self, obs_by_cam: dict[str, list[tuple]]) -> list[tuple]:
        """obs_by_cam: cam -> [(raw_id, x, y, ...)]. Returns [(fused_key, x, y, n_cams)]."""
        self.stats.frames += 1
        clusters: list[dict] = []  # {"members": [(cam, raw, x, y)], "x":, "y":}
        for cam in self.cams:
            obs = obs_by_cam.get(cam) or []
            if not obs:
                continue
            if not clusters:
                clusters = [{"members": [(cam, o[0], o[1], o[2])], "x": o[1], "y": o[2]} for o in obs]
                continue
            C = np.full((len(obs), len(clusters)), 1e6)
            for i, o in enumerate(obs):
                k_o = self.key_of.get((cam, o[0]))
                for j, c in enumerate(clusters):
                    if any(m[0] == cam for m in c["members"]):
                        continue  # one observation per camera per cluster
                    d = float(np.hypot(o[1] - c["x"], o[2] - c["y"]))
                    if d > self.gate:
                        continue
                    same_key = k_o is not None and any(self.key_of.get((m[0], m[1])) == k_o for m in c["members"])
                    C[i, j] = d - (self.bonus if same_key else 0.0)
            ri, ci = linear_sum_assignment(C)
            used = set()
            for i, j in zip(ri, ci):
                if C[i, j] >= 1e5:
                    continue
                o, c = obs[i], clusters[j]
                self.stats.match_dist.append(float(np.hypot(o[1] - c["x"], o[2] - c["y"])))
                c["members"].append((cam, o[0], o[1], o[2]))
                n = len(c["members"])
                c["x"] += (o[1] - c["x"]) / n
                c["y"] += (o[2] - c["y"]) / n
                self.stats.cross_matches += 1
                used.add(i)
            for i, o in enumerate(obs):
                if i not in used:
                    clusters.append({"members": [(cam, o[0], o[1], o[2])], "x": o[1], "y": o[2]})

        out, taken = [], set()
        for c in clusters:
            keys = sorted({self.key_of[(m[0], m[1])] for m in c["members"] if (m[0], m[1]) in self.key_of} - taken)
            key = keys[0] if keys else self._new_key()
            taken.add(key)
            cams = set()
            for m in c["members"]:
                self.key_of[(m[0], m[1])] = key
                cams.add(m[0])
            st = self.stats
            st.key_cams.setdefault(key, set()).update(cams)
            st.key_first_cam.setdefault(key, sorted(cams, key=self.cams.index)[0])
            st.key_last_cams[key] = cams
            out.append((key, c["x"], c["y"], len(cams)))
        return out
