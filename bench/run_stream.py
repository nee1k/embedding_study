"""Layer-2/3 mixed-stream runner (§5).

Builds a realistic interleaved stream — novel queries, exact repeats, and
paraphrase variants — and plays it through the gateway twice:

* **arm A (cache on)**  — the system under test.
* **arm B (cache off)** — every item forced to the backend via cache bypass,
  giving the ground-truth response for the same query.

Both arms write to the same SQLite file with a shared ``run_id``, so
``bench/judge.py`` can join them on query text and label whether a reused
response actually answered the new query.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Optional

from bench.paraphrase import OUT_PATH as PARAPHRASE_PATH
from bench.paraphrase import load_variants
from bench.queries import load_benchmark


@dataclass(frozen=True)
class StreamItem:
    text: str
    category: str
    variant_of: str
    variant_kind: str      # novel | exact | near_duplicate | paraphrase | loose


def build_stream(
    *,
    paraphrase_path: str = PARAPHRASE_PATH,
    repeat_rate: float = 0.2,
    seed: int = 1337,
) -> list[StreamItem]:
    """Interleave originals, repeats and variants into one reproducible stream.

    Every source query appears as ``novel`` before any of its variants, so a
    variant always has something it *could* legitimately reuse. Beyond that
    ordering constraint the stream is shuffled deterministically.
    """
    queries = load_benchmark()
    by_id = {q.query_id: q for q in queries}
    variants = load_variants(paraphrase_path)

    rng = random.Random(seed)

    # Originals first, in shuffled order.
    originals = [
        StreamItem(q.query, q.category, q.query_id, "novel") for q in queries
    ]
    rng.shuffle(originals)

    followups: list[StreamItem] = []
    for v in variants:
        q = by_id.get(v.query_id)
        if q is None:
            continue
        followups.append(StreamItem(v.text, q.category, q.query_id, v.kind))

    # A slice of exact repeats, which must always hit.
    n_repeats = int(len(queries) * repeat_rate)
    for q in rng.sample(queries, k=min(n_repeats, len(queries))):
        followups.append(StreamItem(q.query, q.category, q.query_id, "exact"))

    rng.shuffle(followups)
    return originals + followups


def run(
    *,
    metrics_db: str,
    run_id: str,
    fake: bool,
    threshold: Optional[float],
    paraphrase_path: str,
    seed: int,
    limit: Optional[int],
) -> dict:
    from gateway.app import Gateway, QueryRequest
    from gateway.config import load_settings

    settings = load_settings()
    settings.metrics_db = metrics_db
    if fake:
        settings.embedder_impl = "bagofwords"
        settings.cache_impl = "memory"
        settings.backend_impl = "echo"
    if threshold is not None:
        settings.reuse_threshold = threshold

    stream = build_stream(paraphrase_path=paraphrase_path, seed=seed)
    if limit:
        stream = stream[:limit]

    gateway = Gateway(settings)
    gateway.cache.reset()

    # --- arm A: cache enabled -------------------------------------------------
    hits = 0
    for item in stream:
        resp = gateway.handle(
            QueryRequest(
                query=item.text,
                category=item.category,
                variant_of=item.variant_of,
                variant_kind=item.variant_kind,
                run_id=run_id,
            )
        )
        hits += int(resp.hit)

    # --- arm B: cache bypassed, same stream, for ground truth -----------------
    for item in stream:
        gateway.handle(
            QueryRequest(
                query=item.text,
                category=item.category,
                variant_of=item.variant_of,
                variant_kind=item.variant_kind,
                run_id=run_id,
            ),
            bypass_cache=True,
        )

    return {
        "run_id": run_id,
        "stream_length": len(stream),
        "hits": hits,
        "hit_rate": hits / len(stream) if stream else 0.0,
        "threshold": settings.reuse_threshold,
        "metrics_db": metrics_db,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Phase 2 mixed query stream.")
    ap.add_argument("--out", default="runs/phase2.sqlite", help="SQLite metrics path.")
    ap.add_argument("--run-id", default=None, help="Run identifier (default: timestamp).")
    ap.add_argument("--fake", action="store_true", help="In-process fakes, no network.")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--paraphrases", default=PARAPHRASE_PATH)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=None, help="Truncate the stream (smoke runs).")
    args = ap.parse_args()

    run_id = args.run_id or __import__("time").strftime("%Y%m%dT%H%M%S")
    summary = run(
        metrics_db=args.out,
        run_id=run_id,
        fake=args.fake,
        threshold=args.threshold,
        paraphrase_path=args.paraphrases,
        seed=args.seed,
        limit=args.limit,
    )
    print(f"run_id={summary['run_id']}  stream={summary['stream_length']}  "
          f"hits={summary['hits']}  hit_rate={summary['hit_rate']:.3f}  "
          f"threshold={summary['threshold']}")
    print(f"wrote {summary['metrics_db']}")
    print("next: python -m bench.judge --db "
          f"{summary['metrics_db']} && python -m bench.report --db {summary['metrics_db']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
