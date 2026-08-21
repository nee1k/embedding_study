"""Phase 2 report (§6).

Emits every metric §6 requires: hit rate, latency percentiles split by path with
the embed leg isolated, tokens/cost avoided, false-reuse rate, the similarity
distributions that the separability study turns on, and all of it sliced by the
seven query categories. Plus the offline threshold sweep.

Threshold sweep caveat: the sweep recomputes ``similarity >= t`` over the
similarities logged during one run, holding cache *contents* fixed at what was
actually stored under that run's threshold. A different threshold would have
changed which queries were stored, so the sweep is a counterfactual estimate,
not a substitute for re-running at the chosen threshold. It is still the cheap
way to narrow the search before spending inference calls.
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Optional, Sequence

from bench.queries import CATEGORIES


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile; avoids a numpy dependency in the report path."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def _fmt(value: Optional[float], width: int = 8, places: int = 1) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.{places}f}"


def _rows(conn, run_id, where="", params=()):
    sql = "SELECT * FROM requests WHERE cache_enabled = 1"
    if run_id:
        sql += " AND run_id = ?"
        params = (run_id,) + tuple(params)
    if where:
        sql += f" AND {where}"
    return conn.execute(sql, params).fetchall()


def section_overall(conn, run_id) -> str:
    rows = _rows(conn, run_id)
    if not rows:
        return "No cache-enabled requests found.\n"

    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]
    tokens_avoided = sum(r["prompt_tokens"] + r["completion_tokens"] for r in hits)
    cost_avoided = sum(r["cost_usd"] for r in hits)
    tokens_spent = sum(r["prompt_tokens"] + r["completion_tokens"] for r in misses)

    lines = ["## Overall", ""]
    lines.append(f"requests            {len(rows)}")
    lines.append(f"hits                {len(hits)}")
    lines.append(f"hit rate            {len(hits) / len(rows):.3f}")
    lines.append(f"threshold           {rows[0]['threshold']}")
    lines.append(f"tokens avoided      {tokens_avoided}")
    lines.append(f"tokens spent        {tokens_spent}")
    if tokens_avoided + tokens_spent:
        lines.append(
            f"token reduction     {tokens_avoided / (tokens_avoided + tokens_spent):.3f}"
        )
    lines.append(f"cost avoided (USD)  {cost_avoided:.4f}")
    return "\n".join(lines) + "\n"


def section_latency(conn, run_id) -> str:
    rows = _rows(conn, run_id)
    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]

    lines = ["## Latency (ms)", ""]
    lines.append(f"{'path':10s} {'n':>5s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
    lines.append("-" * 44)
    for label, subset in (("hit", hits), ("miss", misses), ("all", rows)):
        vals = [r["latency_total_ms"] for r in subset]
        lines.append(
            f"{label:10s} {len(subset):5d} "
            f"{_fmt(_pct(vals, 50))} {_fmt(_pct(vals, 95))} {_fmt(_pct(vals, 99))}"
        )

    lines.append("")
    lines.append("Legs (all requests):")
    lines.append(f"{'leg':10s} {'n':>5s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
    lines.append("-" * 44)
    for label, col in (("embed", "latency_embed_ms"), ("search", "latency_search_ms")):
        vals = [r[col] for r in rows]
        lines.append(
            f"{label:10s} {len(rows):5d} "
            f"{_fmt(_pct(vals, 50))} {_fmt(_pct(vals, 95))} {_fmt(_pct(vals, 99))}"
        )
    backend_vals = [r["latency_backend_ms"] for r in misses]
    lines.append(
        f"{'backend':10s} {len(misses):5d} "
        f"{_fmt(_pct(backend_vals, 50))} {_fmt(_pct(backend_vals, 95))} "
        f"{_fmt(_pct(backend_vals, 99))}"
    )
    lines.append("")
    lines.append("The embed leg is the hit-path floor: no hit can be faster than it.")
    return "\n".join(lines) + "\n"


def section_by_category(conn, run_id) -> str:
    lines = ["## Per category", ""]
    lines.append(
        f"{'category':13s} {'n':>4s} {'hits':>5s} {'rate':>6s} "
        f"{'hit p50':>8s} {'miss p50':>9s} {'tok avoid':>10s}"
    )
    lines.append("-" * 62)
    for cat in CATEGORIES:
        rows = _rows(conn, run_id, "category = ?", (cat,))
        if not rows:
            continue
        hits = [r for r in rows if r["hit"]]
        misses = [r for r in rows if not r["hit"]]
        lines.append(
            f"{cat:13s} {len(rows):4d} {len(hits):5d} "
            f"{len(hits) / len(rows):6.3f} "
            f"{_fmt(_pct([r['latency_total_ms'] for r in hits], 50))} "
            f"{_fmt(_pct([r['latency_total_ms'] for r in misses], 50), 9)} "
            f"{sum(r['prompt_tokens'] + r['completion_tokens'] for r in hits):10d}"
        )
    return "\n".join(lines) + "\n"


def section_separability(conn, run_id) -> str:
    """RQ5: do similarity scores separate correct reuse from incorrect reuse?"""
    sql = """
        SELECT r.top1_similarity AS sim, j.correct AS correct, r.category AS category
        FROM requests r JOIN judgments j ON j.request_id = r.request_id
        WHERE r.cache_enabled = 1 AND r.hit = 1 AND j.correct IS NOT NULL
    """
    params: tuple = ()
    if run_id:
        sql += " AND r.run_id = ?"
        params = (run_id,)
    rows = conn.execute(sql, params).fetchall()

    lines = ["## Separability of reuse similarity (RQ5)", ""]
    if not rows:
        lines.append("No judged hits. Run bench/judge.py first.")
        return "\n".join(lines) + "\n"

    correct = [r["sim"] for r in rows if r["correct"] == 1 and r["sim"] is not None]
    wrong = [r["sim"] for r in rows if r["correct"] == 0 and r["sim"] is not None]

    lines.append(f"{'label':10s} {'n':>5s} {'min':>7s} {'p50':>7s} {'max':>7s}")
    lines.append("-" * 40)
    for label, vals in (("correct", correct), ("incorrect", wrong)):
        lines.append(
            f"{label:10s} {len(vals):5d} "
            f"{_fmt(min(vals) if vals else None, 7, 3)} "
            f"{_fmt(_pct(vals, 50), 7, 3)} "
            f"{_fmt(max(vals) if vals else None, 7, 3)}"
        )

    judged = len(correct) + len(wrong)
    if judged:
        lines.append("")
        lines.append(f"false-reuse rate    {len(wrong) / judged:.3f}  ({len(wrong)}/{judged} judged hits)")

    if correct and wrong:
        overlap_lo, overlap_hi = min(correct), max(wrong)
        lines.append("")
        if overlap_hi <= overlap_lo:
            lines.append(
                f"Distributions are separable: every incorrect reuse scored at or below "
                f"{overlap_hi:.3f} and every correct reuse at or above {overlap_lo:.3f}. "
                f"A threshold in that gap separates them cleanly."
            )
        else:
            n_overlap = sum(1 for s in correct + wrong if overlap_lo <= s <= overlap_hi)
            lines.append(
                f"Distributions OVERLAP on [{overlap_lo:.3f}, {overlap_hi:.3f}] "
                f"({n_overlap} of {judged} judged hits). No threshold separates correct "
                f"from incorrect reuse in this range — the text-only caching literature "
                f"reports the same, and it redirects the work toward scoping signals "
                f"rather than threshold tuning (§8.3)."
            )
    return "\n".join(lines) + "\n"


def section_threshold_sweep(conn, run_id, candidates: Sequence[float]) -> str:
    rows = _rows(conn, run_id, "top1_similarity IS NOT NULL")
    lines = ["## Threshold sweep (counterfactual)", ""]
    if not rows:
        lines.append("No similarities logged.")
        return "\n".join(lines) + "\n"

    sims = [r["top1_similarity"] for r in rows]
    actual = rows[0]["threshold"]
    lines.append(
        f"Recomputed over {len(sims)} logged similarities from the run at "
        f"threshold {actual}. Cache contents are held fixed at what that run "
        f"stored, so these are estimates for narrowing the search, not a "
        f"substitute for re-running at the chosen threshold."
    )
    lines.append("")
    lines.append(f"{'threshold':>10s} {'hits':>6s} {'hit rate':>9s}")
    lines.append("-" * 28)
    for t in candidates:
        n = sum(1 for s in sims if s >= t)
        lines.append(f"{t:10.2f} {n:6d} {n / len(sims):9.3f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Report Phase 2 measurements.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--thresholds",
        default="0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.98,0.99,1.00",
        help="Comma-separated candidate thresholds for the sweep.",
    )
    args = ap.parse_args()

    candidates = [float(t) for t in args.thresholds.split(",") if t.strip()]
    conn = _connect(args.db)

    print(f"# Semantic Result Reuse — Phase 2 report")
    print(f"db={args.db}  run_id={args.run_id or '(all)'}\n")
    for section in (
        section_overall(conn, args.run_id),
        section_latency(conn, args.run_id),
        section_by_category(conn, args.run_id),
        section_separability(conn, args.run_id),
        section_threshold_sweep(conn, args.run_id, candidates),
    ):
        print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
