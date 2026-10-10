"""DuckDB storage: one file per run (runs/<run_id>/flowsight.duckdb) + Parquet export.

Writes are buffered and flushed in batches (DataFrame inserts) — never row-by-row. A lock serialises
access so the API thread can query while the pipeline thread writes.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = {
    "runs": "run_id VARCHAR, store_id VARCHAR, source VARCHAR, started_at DOUBLE, ended_at DOUBLE, "
            "device VARCHAR, model VARCHAR, calibrated BOOLEAN, meta VARCHAR",
    "frames": "ts DOUBLE, n_people INTEGER",
    "track_points": "track_id INTEGER, ts DOUBLE, x DOUBLE, y DOUBLE, speed DOUBLE, cam VARCHAR",
    "visits": "track_id INTEGER, zone_id VARCHAR, t_enter DOUBLE, t_exit DOUBLE, dwell_s DOUBLE, "
              "engaged_s DOUBLE, is_stop BOOLEAN, mean_speed DOUBLE",
    "zone_windows": "zone_id VARCHAR, window_start DOUBLE, window_end DOUBLE, occupancy DOUBLE, density DOUBLE, "
                    "los VARCHAR, mean_speed DOUBLE, flow_per_min DOUBLE, free_flow_mps DOUBLE",
    "bottlenecks": "zone_id VARCHAR, t_start DOUBLE, t_confirmed DOUBLE, t_end DOUBLE, duration_s DOUBLE, "
                   "peak_density DOUBLE, peak_los VARCHAR, people_affected INTEGER, time_lost_s DOUBLE, reason VARCHAR",
    "attention_windows": "ts DOUBLE, zone_id VARCHAR, exposures INTEGER, entries INTEGER, stops INTEGER, "
                         "engaged_s DOUBLE, stopping_power DOUBLE, status VARCHAR",
    "queue_windows": "zone_id VARCHAR, window_start DOUBLE, window_end DOUBLE, queue_len DOUBLE, "
                     "arrival_rate_pm DOUBLE, service_rate_pm DOUBLE, predicted_wait_s DOUBLE, "
                     "observed_wait_s DOUBLE, rising_streak INTEGER",
    "insights": "id VARCHAR, created_ts DOUBLE, updated_ts DOUBLE, type VARCHAR, zone_id VARCHAR, severity VARCHAR, "
                "title VARCHAR, action VARCHAR, owner VARCHAR, horizon VARCHAR, expected_impact VARCHAR, "
                "confidence DOUBLE, status VARCHAR, priority DOUBLE, evidence VARCHAR",
}
COLUMNS = {t: [c.strip().split(" ")[0] for c in s.split(",")] for t, s in SCHEMA.items()}


class RunDB:
    def __init__(self, path: str | Path, flush_every_s: float = 1.0, read_only: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        self.lock = threading.RLock()
        self.flush_every_s = flush_every_s
        self._buf: dict[str, list] = {t: [] for t in SCHEMA}
        self._last_flush = time.monotonic()
        if not read_only:
            with self.lock:
                for t, cols in SCHEMA.items():
                    self.con.execute(f"CREATE TABLE IF NOT EXISTS {t} ({cols})")

    # ---------------------------------------------------------------- writes
    def add(self, table: str, row: dict) -> None:
        self._buf[table].append(row)

    def add_many(self, table: str, rows: list[dict]) -> None:
        self._buf[table].extend(rows)

    def maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush >= self.flush_every_s:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            for t, rows in self._buf.items():
                if not rows:
                    continue
                df = pd.DataFrame(rows).reindex(columns=COLUMNS[t])
                self.con.register("_buf_df", df)
                self.con.execute(f"INSERT INTO {t} SELECT * FROM _buf_df")
                self.con.unregister("_buf_df")
                self._buf[t] = []
        self._last_flush = time.monotonic()

    def upsert_insights(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.lock:
            ids = [r["id"] for r in rows]
            self.con.execute(f"DELETE FROM insights WHERE id IN ({','.join('?' * len(ids))})", ids)
            df = pd.DataFrame([{**r, "evidence": json.dumps(r.get("evidence", {}), default=str)} for r in rows])
            df = df.reindex(columns=COLUMNS["insights"])
            self.con.register("_ins_df", df)
            self.con.execute("INSERT INTO insights SELECT * FROM _ins_df")
            self.con.unregister("_ins_df")

    def replace_table(self, table: str, rows: list[dict]) -> None:
        with self.lock:
            self.con.execute(f"DELETE FROM {table}")
            if rows:
                df = pd.DataFrame(rows).reindex(columns=COLUMNS[table])
                self.con.register("_rep_df", df)
                self.con.execute(f"INSERT INTO {table} SELECT * FROM _rep_df")
                self.con.unregister("_rep_df")

    # ---------------------------------------------------------------- reads
    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self.lock:
            return self.con.execute(sql, params or []).df()

    def records(self, sql: str, params: list | None = None) -> list[dict]:
        df = self.query(sql, params)
        return json.loads(df.to_json(orient="records"))

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {t: int(self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in SCHEMA}

    def export_parquet(self, out_dir: str | Path) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        with self.lock:
            for t in SCHEMA:
                p = out / f"{t}.parquet"
                self.con.execute(f"COPY {t} TO '{p.as_posix()}' (FORMAT PARQUET)")
                paths.append(p)
        return paths

    def close(self) -> None:
        with self.lock:
            try:
                self.flush()
            finally:
                self.con.close()
