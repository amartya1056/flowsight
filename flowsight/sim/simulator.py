"""Agent-based shopper simulator with ground truth.

Shoppers arrive (Poisson, optionally time-varying), draw a shopping list of zones weighted by
attractiveness, walk a waypoint graph (with per-agent lane offsets and Weidmann density slow-down in
walkways), browse inside zones (lognormal dwell, slow wandering with pauses), queue at a single-server
checkout (lognormal service) and leave.

Each step yields:
  * clean ground truth  — `self.clean`: (true_id, x, y, vx, vy) for every agent
  * noisy observations  — what a tracker would report: σ-noise, random dropouts and ID breaks
                          (a new observation ID after a short silent gap), mapping kept in obs_to_true

Scenario knobs (config/stores/<store>.yaml -> sim.scenarios): rush_hour (arrival spike -> queue spills
into the walkway), blocked_aisle (obstacle narrows the walkway), ignored_promo (promo nobody stops at),
dead_aisle (zone off the main path nobody visits).
"""
from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon

WALK, BROWSE, QUEUE, SERVICE, DONE = "WALK", "BROWSE", "QUEUE", "SERVICE", "DONE"


@dataclass
class Agent:
    id: int
    x: float
    y: float
    v0: float
    lane: float
    plan: list
    buyer: bool
    state: str = WALK
    path: list = field(default_factory=list)
    goal: tuple = ("none",)
    node: str | None = None
    zone: str | None = None
    browse_until: float = 0.0
    pause_until: float = 0.0
    sub: tuple | None = None
    vx: float = 0.0
    vy: float = 0.0
    obs_id: int = 0
    silent_until: float = -1.0
    t_spawn: float = 0.0


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Simulator:
    def __init__(self, store, cfg, scenario: str = "baseline", seed: int = 0, overrides: dict | None = None,
                 duration_s: float | None = None):
        if not store.sim:
            raise ValueError(f"store {store.store_id} has no `sim` layout")
        self.store, self.cfg, self.scenario = store, cfg, scenario
        lay = copy.deepcopy(store.sim)
        scen = lay.get("scenarios", {}).get(scenario)
        if scen is None:
            raise ValueError(f"unknown scenario {scenario!r}; have {list(lay.get('scenarios', {}))}")
        attract_over = scen.get("attract", {})
        lay = _merge(lay, {k: v for k, v in scen.items() if k != "attract"})
        lay = _merge(lay, overrides or {})
        for z, a in attract_over.items():
            lay["destinations"][z]["attract"] = a
        self.lay = lay
        self.scfg = cfg.sim
        self.dt = cfg.sim.dt
        self.duration_s = duration_s or cfg.sim.duration_s
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.agents: list[Agent] = []
        self.done_agents = 0
        self._next_agent = 1
        self._next_obs = 1
        self.obs_to_true: dict[int, int] = {}
        self.clean: list[tuple] = []
        self.n_id_breaks = 0

        self.nodes = {k: (float(v["at"][0]), float(v["at"][1])) for k, v in lay["nodes"].items()}
        self.lane_nodes = {k for k, v in lay["nodes"].items() if v.get("lane")}
        self.adj: dict[str, list] = {k: [] for k in self.nodes}
        for a, b in lay["edges"]:
            d = math.dist(self.nodes[a], self.nodes[b])
            self.adj[a].append((b, d))
            self.adj[b].append((a, d))
        self.zpoly = {z.id: Polygon(z.polygon) for z in store.zones}
        self.zinner = {k: (p.buffer(-0.25) if p.buffer(-0.25).area > 0.2 else p) for k, p in self.zpoly.items()}
        self.dest = lay["destinations"]
        self.walkways = [z for z in store.zones if z.type == "walkway"]
        self.walk_poly = {z.id: Polygon(z.polygon) for z in self.walkways}
        co = lay["checkout"]
        self.server = tuple(co["server_at"])
        self.qdir = np.array(co["queue_dir"], float) / np.linalg.norm(co["queue_dir"])
        self.queue: list[Agent] = []
        self.serving: Agent | None = None
        self.serve_until = 0.0
        self.served = 0
        # ground-truth log (for evaluation)
        self.gt_queue: list[dict] = []
        self._q_join: dict[int, float] = {}
        self.route_cache: dict = {}

    # ----------------------------------------------------------- utilities
    def arrival_rate_pm(self, t: float) -> float:
        prof = self.lay.get("arrival_profile")
        if prof:
            r = prof[0][1]
            for t0, rate in prof:
                if t >= t0:
                    r = rate
            return float(r)
        return float(self.lay.get("arrival_rate_pm", 3.0))

    def _route(self, a: str, b: str) -> list[str]:
        key = (a, b)
        if key in self.route_cache:
            return self.route_cache[key]
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > dist.get(u, 1e18):
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        path, u = [b], b
        while u != a:
            u = prev[u]
            path.append(u)
        path.reverse()
        self.route_cache[key] = path
        return path

    def _pt(self, node: str, ag: Agent) -> tuple[float, float]:
        x, y = self.nodes[node]
        if node in self.lane_nodes:
            y += ag.lane
        return x, y

    def _rand_in(self, zid: str) -> tuple[float, float]:
        p = self.zinner[zid]
        minx, miny, maxx, maxy = p.bounds
        for _ in range(50):
            x, y = self.rng.uniform(minx, maxx), self.rng.uniform(miny, maxy)
            if p.contains(Point(x, y)):
                return float(x), float(y)
        c = p.centroid
        return float(c.x), float(c.y)

    def _plan_next(self, ag: Agent, from_node: str) -> None:
        """Set ag.path / ag.goal towards the next item on its list, checkout, or the exit."""
        if ag.plan:
            z = ag.plan.pop(0)
            tgt = self.dest[z]["node"]
            nodes = self._route(from_node, tgt)
            ag.path = [self._pt(n, ag) for n in nodes[1:]] + [self._rand_in(z)]
            ag.goal = ("browse", z, tgt)
        elif ag.buyer:
            acc = self.lay["checkout"]["access_node"]
            nodes = self._route(from_node, acc)
            ag.path = [self._pt(n, ag) for n in nodes[1:]]
            ag.goal = ("queue",)
        else:
            nodes = self._route(from_node, "E")
            ag.path = [self._pt(n, ag) for n in nodes[1:]] + [tuple(self.lay["spawn"])]
            ag.goal = ("exit",)
        ag.state = WALK

    def _spawn(self) -> None:
        lam = self.arrival_rate_pm(self.t) / 60.0 * self.dt
        for _ in range(self.rng.poisson(lam)):
            plan = [z for z, d in self.dest.items() if self.rng.random() < d["attract"]]
            if not plan:
                zs = list(self.dest)
                w = np.array([max(self.dest[z]["attract"], 1e-3) for z in zs])
                plan = [zs[self.rng.choice(len(zs), p=w / w.sum())]]
            plan.sort(key=lambda z: (self.nodes[self.dest[z]["node"]][0], self.nodes[self.dest[z]["node"]][1]))
            lo, hi = self.lay["walk_speed"]
            sx, sy = self.lay["spawn"]
            ag = Agent(id=self._next_agent, x=sx, y=sy + self.rng.uniform(-0.6, 0.6),
                       v0=float(self.rng.uniform(lo, hi)),
                       lane=float(self.rng.uniform(-self.lay["lane_offset_m"], self.lay["lane_offset_m"])),
                       plan=plan, buyer=bool(self.rng.random() < self.lay["p_checkout"]), t_spawn=self.t)
            self._next_agent += 1
            ag.obs_id = self._new_obs(ag)
            ag.path = [self._pt("E", ag)]
            ag.goal = ("start",)
            self.agents.append(ag)

    def _new_obs(self, ag: Agent) -> int:
        oid = self._next_obs
        self._next_obs += 1
        self.obs_to_true[oid] = ag.id
        return oid

    # -------------------------------------------------------------- motion
    def _speed_factor(self, ag: Agent, dens: dict[str, float]) -> float:
        f = 1.0
        p = Point(ag.x, ag.y)
        for zid, poly in self.walk_poly.items():
            if poly.contains(p):
                rho = dens.get(zid, 0.0)
                if rho > 0.2:  # Weidmann fundamental diagram
                    f = min(f, max(0.15, 1 - math.exp(-1.913 * (1 / rho - 1 / 5.4))))
        return f

    def _cap(self, ag: Agent) -> float:
        cap = 1e9
        for ob in self.lay.get("obstacles", []) or []:
            x, y, w, h = ob["rect"]
            if ob.get("from_s", 0) <= self.t <= ob.get("to_s", 1e18) and x <= ag.x <= x + w and y <= ag.y <= y + h:
                cap = min(cap, ob["max_speed"])
        return cap

    def _move_towards(self, ag: Agent, tx: float, ty: float, v: float) -> bool:
        dx, dy = tx - ag.x, ty - ag.y
        d = math.hypot(dx, dy)
        step = v * self.dt
        if d <= step or d < 1e-6:
            ag.x, ag.y = tx, ty
            return True
        ag.x += dx / d * step
        ag.y += dy / d * step
        return False

    def _walk(self, ag: Agent, v: float) -> None:
        while ag.path:
            tx, ty = ag.path[0]
            if self._move_towards(ag, tx, ty, v):
                ag.path.pop(0)
                # only consume one waypoint per step unless we arrived exactly
                if ag.path:
                    return
            else:
                return
        self._arrive(ag)

    def _arrive(self, ag: Agent) -> None:
        g = ag.goal[0]
        if g == "start":
            self._plan_next(ag, "E")
        elif g == "browse":
            _, z, node = ag.goal
            med, sig = self.dest[z]["dwell_median_s"], self.dest[z]["dwell_sigma"]
            ag.state, ag.zone, ag.node = BROWSE, z, node
            ag.browse_until = self.t + float(med * math.exp(self.rng.normal(0, sig)))
            ag.pause_until = self.t + float(self.rng.uniform(2, 6))
            ag.sub = None
        elif g == "queue":
            ag.state = QUEUE
            self.queue.append(ag)
            self._q_join[ag.id] = self.t
        elif g == "exit":
            ag.state = DONE

    def _slot(self, i: int) -> tuple[float, float]:
        sp = self.lay["checkout"]["slot_spacing_m"]
        return self.server[0] + self.qdir[0] * sp * i, self.server[1] + self.qdir[1] * sp * i

    # ---------------------------------------------------------------- step
    def step(self) -> tuple[float, list[tuple]]:
        self._spawn()
        dens = {}
        for zid, poly in self.walk_poly.items():
            n = sum(1 for a in self.agents if poly.contains(Point(a.x, a.y)))
            dens[zid] = n / poly.area
        # checkout server
        if self.serving is not None and self.t >= self.serve_until:
            ag = self.serving
            self.serving = None
            self.served += 1
            self.gt_queue.append({"agent": ag.id, "t_join": self._q_join.pop(ag.id, self.t), "t_done": self.t})
            co = self.lay["checkout"]
            ag.path = [self._pt(co["exit_node"], ag), tuple(self.lay["exit"])]
            ag.goal = ("exit",)
            ag.state = WALK
        if self.serving is None and self.queue:
            head = self.queue[0]
            if math.dist((head.x, head.y), self.server) < 0.3:
                self.queue.pop(0)
                self.serving = head
                head.state = SERVICE
                co = self.lay["checkout"]
                self.serve_until = self.t + float(co["service_median_s"] * math.exp(self.rng.normal(0, co["service_sigma"])))

        for ag in self.agents:
            px, py = ag.x, ag.y
            if ag.state == WALK:
                v = min(ag.v0 * self._speed_factor(ag, dens), self._cap(ag))
                self._walk(ag, v)
            elif ag.state == BROWSE:
                if self.t >= ag.browse_until:
                    self._plan_next(ag, ag.node)
                    ag.path.insert(0, self._pt(ag.node, ag))
                elif self.t >= ag.pause_until:
                    if ag.sub is None:
                        ag.sub = self._rand_in(ag.zone)
                    if self._move_towards(ag, ag.sub[0], ag.sub[1], 0.35):
                        ag.sub = None
                        ag.pause_until = self.t + float(self.rng.uniform(3, 8))
            elif ag.state == QUEUE:
                i = self.queue.index(ag) if ag in self.queue else 0
                tx, ty = self._slot(i)
                v = min(0.9, ag.v0, self._cap(ag))
                self._move_towards(ag, tx, ty, v)
            elif ag.state == SERVICE:
                self._move_towards(ag, self.server[0], self.server[1], 0.5)
            ag.vx, ag.vy = (ag.x - px) / self.dt, (ag.y - py) / self.dt

        before = len(self.agents)
        self.agents = [a for a in self.agents if a.state != DONE]
        self.done_agents += before - len(self.agents)
        self.t = round(self.t + self.dt, 6)
        return self.t, self._observe()

    def _observe(self) -> list[tuple]:
        sig = self.scfg.noise_sigma_m
        self.clean = [(a.id, a.x, a.y, a.vx, a.vy) for a in self.agents]
        obs = []
        lo, hi = self.scfg.id_break_gap_s
        for a in self.agents:
            if self.t < a.silent_until:
                continue
            if self.rng.random() < self.scfg.id_break_rate_per_s * self.dt:
                a.obs_id = self._new_obs(a)
                a.silent_until = self.t + float(self.rng.uniform(lo, hi))
                self.n_id_breaks += 1
                continue
            if self.rng.random() < self.scfg.dropout_prob:
                continue
            obs.append((a.obs_id, a.x + self.rng.normal(0, sig), a.y + self.rng.normal(0, sig)))
        return obs

    def clean_states(self):
        """Ground-truth TrackStates (true ID, true position, true speed) for the oracle analytics."""
        from flowsight.tracking.track_store import TrackState

        out = []
        for a in self.agents:
            sp = math.hypot(a.vx, a.vy)
            out.append(TrackState(id=a.id, ts=self.t, x=a.x, y=a.y, speed=sp, heading=math.atan2(a.vy, a.vx),
                                  vx=a.vx, vy=a.vy, age=self.t - a.t_spawn, mature=True))
        return out
