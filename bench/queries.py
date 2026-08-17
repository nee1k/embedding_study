"""Load the 75-query discovery benchmark.

Vendored from ``Plale-Lab/vector-volume-discovery-eval`` (``Vector Discovery
Benchmark 75.csv``). Only query text, category, source book and ground-truth
page ranges travel with it — no copyrighted page content.

Category labels are normalized on load: the source CSV stores ``"multi-modal "``
with a trailing space, which would otherwise split one category into two.
"""

from __future__ import annotations

import ast
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "benchmark75.csv")

CATEGORIES = (
    "conceptual",
    "multi-modal",
    "multi-page",
    "numerical",
    "tabular",
    "textual",
    "visual",
)


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    category: str
    source_book: Optional[str] = None
    ground_truth_pages: tuple[int, ...] = ()


def _parse_ground_truth(raw: str) -> tuple[int, ...]:
    """Expand the CSV's page-range notation into a flat page tuple.

    The column holds things like ``([89, 91], [97, 99])`` meaning inclusive
    ranges. An empty value means "all pages in range" and yields ().
    """
    text = (raw or "").strip()
    if not text:
        return ()
    pairs = re.findall(r"\[([^\]]*)\]", text)
    pages: list[int] = []
    for pair in pairs:
        try:
            nums = ast.literal_eval(f"[{pair}]")
        except (ValueError, SyntaxError):
            continue
        nums = [int(n) for n in nums if isinstance(n, (int, float))]
        if len(nums) == 2 and nums[0] <= nums[1]:
            pages.extend(range(nums[0], nums[1] + 1))
        else:
            pages.extend(nums)
    return tuple(sorted(set(pages)))


def load_benchmark(path: str = DATA_PATH) -> list[BenchmarkQuery]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out: list[BenchmarkQuery] = []
    for row in rows:
        category = (row.get("query type") or "").strip().lower()
        if category not in CATEGORIES:
            raise ValueError(f"Unrecognized category {category!r} in {path}")
        out.append(
            BenchmarkQuery(
                query_id=(row.get("Query ID") or "").strip(),
                query=(row.get("Query") or "").strip(),
                category=category,
                source_book=(row.get("Primary Expected Source for Answer (Textbook Name)") or "").strip()
                or None,
                ground_truth_pages=_parse_ground_truth(
                    row.get("Ground Truth: [] means all pages in range", "")
                ),
            )
        )
    return out


def category_counts(queries: list[BenchmarkQuery]) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for q in queries:
        counts[q.category] += 1
    return counts


if __name__ == "__main__":
    qs = load_benchmark()
    print(f"{len(qs)} queries")
    for cat, n in sorted(category_counts(qs).items(), key=lambda kv: -kv[1]):
        print(f"  {cat:12s} {n}")
