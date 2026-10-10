"""Customer journeys: zone sequences, transitions, Sankey, top paths, conversion.

Sequences are built from confirmed visits ordered by entry time; consecutive duplicates collapse and
transit zones (walkways by default) are excluded so paths read as destinations.
The Sankey is *step-layered* ("Aisle 3 · 2" = Aisle 3 as the 2nd stop) which keeps it acyclic.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class FlowAnalyzer:
    def __init__(self, zones, cfg):
        self.cfg = cfg
        self.zone_type = {z.id: z.type for z in zones}
        self.zone_name = {z.id: z.name for z in zones}
        self.excluded = set(cfg.exclude_types)
        self.track_first: dict[int, float] = {}
        self.track_last: dict[int, float] = {}
        self.track_len: dict[int, float] = {}

    def observe_track(self, tid: int, ts: float, path_len: float) -> None:
        self.track_first.setdefault(tid, ts)
        self.track_last[tid] = ts
        self.track_len[tid] = path_len

    def sequences(self, visits) -> dict[int, list[str]]:
        by_track: dict[int, list] = defaultdict(list)
        for v in visits:
            if self.zone_type.get(v.zone_id) in self.excluded:
                continue
            by_track[v.track_id].append(v)
        seqs = {}
        for tid, vs in by_track.items():
            vs.sort(key=lambda v: v.t_enter)
            seq: list[str] = []
            for v in vs:
                if not seq or seq[-1] != v.zone_id:
                    seq.append(v.zone_id)
            seqs[tid] = seq
        return seqs

    def summary(self, visits, top_n: int = 10) -> dict:
        seqs = self.sequences(visits)
        trans = Counter()
        for seq in seqs.values():
            for a, b in zip(seq, seq[1:]):
                trans[(a, b)] += 1
        paths = Counter(tuple(s) for s in seqs.values() if len(s) >= 2)

        # step-layered sankey
        nodes, links = {}, Counter()
        max_steps = self.cfg.sankey_max_steps
        for seq in seqs.values():
            s = seq[:max_steps]
            for k, z in enumerate(s):
                nodes[f"{self.zone_name.get(z, z)} · {k + 1}"] = True
            for k, (a, b) in enumerate(zip(s, s[1:])):
                links[(f"{self.zone_name.get(a, a)} · {k + 1}", f"{self.zone_name.get(b, b)} · {k + 2}")] += 1

        entr = {t for t, s in seqs.items() if any(self.zone_type.get(z) == "entrance" for z in s)}
        conv = {t for t in entr if any(self.zone_type.get(z) == "checkout" for z in seqs[t])}
        times = [self.track_last[t] - self.track_first[t] for t in self.track_first if t in self.track_last]
        lens = [v for v in self.track_len.values() if v > 0]
        return {
            "transitions": [{"from": a, "to": b, "count": c} for (a, b), c in trans.most_common()],
            "sankey": {
                "nodes": [{"name": n} for n in nodes],
                "links": [{"source": a, "target": b, "value": c} for (a, b), c in links.items()],
            },
            "top_paths": [{"path": list(p), "names": [self.zone_name.get(z, z) for z in p], "count": c}
                          for p, c in paths.most_common(top_n)],
            "entrance_visitors": len(entr),
            "converted": len(conv),
            "conversion": (len(conv) / len(entr)) if entr and "checkout" in self.zone_type.values() else None,
            "avg_path_length_m": (sum(lens) / len(lens)) if lens else 0.0,
            "avg_time_in_store_s": (sum(times) / len(times)) if times else 0.0,
            "tracks": len(self.track_first),
        }
