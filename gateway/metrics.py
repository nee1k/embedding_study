"""Per-request metrics.

One SQLite row per request, written on every path — hit and miss alike. The
schema carries everything §6 asks for, plus the field that makes the threshold
sweep free: ``top1_similarity`` is recorded even when the nearest entry lost,
so hit rate at any candidate threshold is re-derivable from a single run.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id          TEXT PRIMARY KEY,
    ts                  REAL NOT NULL,
    run_id              TEXT,
    query_text          TEXT NOT NULL,
    category            TEXT,
    variant_of          TEXT,           -- benchmark Query ID this paraphrases
    variant_kind        TEXT,           -- exact | paraphrase | near_miss | novel
    cache_enabled       INTEGER NOT NULL,
    hit                 INTEGER NOT NULL,
    top1_similarity     REAL,           -- logged even on a miss
    threshold           REAL NOT NULL,
    reused_from_id      TEXT,
    response_text       TEXT,
    latency_total_ms    REAL NOT NULL,
    latency_embed_ms    REAL NOT NULL,
    latency_search_ms   REAL NOT NULL,
    latency_backend_ms  REAL NOT NULL,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    cost_usd            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS idx_requests_cat ON requests(category);

-- Populated later by bench/judge.py; kept separate so re-judging never
-- rewrites the measurement rows.
CREATE TABLE IF NOT EXISTS judgments (
    request_id      TEXT PRIMARY KEY,
    correct         INTEGER,        -- 1 = reuse still answered the new query
    rationale       TEXT,
    judge_model     TEXT
);
"""


@dataclass
class RequestRecord:
    request_id: str
    ts: float
    query_text: str
    cache_enabled: bool
    hit: bool
    threshold: float
    latency_total_ms: float
    latency_embed_ms: float
    latency_search_ms: float
    latency_backend_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    run_id: Optional[str] = None
    category: Optional[str] = None
    variant_of: Optional[str] = None
    variant_kind: Optional[str] = None
    top1_similarity: Optional[float] = None
    reused_from_id: Optional[str] = None
    response_text: Optional[str] = None


class MetricsStore:
    """Thread-safe SQLite writer.

    FastAPI runs sync handlers in a worker thread pool, so writes are guarded by
    a lock and the connection is opened with ``check_same_thread=False``.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def record(self, rec: RequestRecord) -> None:
        row = asdict(rec)
        row["cache_enabled"] = int(row["cache_enabled"])
        row["hit"] = int(row["hit"])
        cols = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self._lock:
            self._conn.execute(f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({placeholders})", row)
            self._conn.commit()

    def summary(self) -> dict:
        """Small live view for ``GET /stats``; full analysis lives in bench/report.py."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(hit), 0),
                       COALESCE(SUM(CASE WHEN hit THEN prompt_tokens + completion_tokens END), 0),
                       COALESCE(SUM(CASE WHEN hit THEN cost_usd END), 0.0)
                FROM requests WHERE cache_enabled = 1
                """
            )
            total, hits, tokens_avoided, cost_avoided = cur.fetchone()
        return {
            "requests": total,
            "hits": hits,
            "hit_rate": (hits / total) if total else 0.0,
            "tokens_avoided": tokens_avoided,
            "cost_usd_avoided": cost_avoided,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def compute_cost(prompt_tokens: int, completion_tokens: int, settings) -> float:
    """Dollars derived from a configurable rate; tokens remain the primary unit."""
    return (
        prompt_tokens / 1000.0 * settings.usd_per_1k_prompt_tokens
        + completion_tokens / 1000.0 * settings.usd_per_1k_completion_tokens
    )
