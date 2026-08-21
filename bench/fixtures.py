"""Layer-1 correctness fixtures (§5).

Hand-written, not generated. Each fixture primes the cache with one query and
then probes with a second, asserting whether reuse should occur:

* exact repeat      -> must hit
* obvious paraphrase-> should hit
* unrelated         -> must miss
* near-miss         -> must miss  (the expensive-error case)

The near-miss pairs deliberately share heavy vocabulary while asking for
something different; a generator does not reliably produce those, which is why
they are written by hand.

``requires`` marks fixtures that only mean something with a real semantic
embedder. Under the bag-of-words fake they are SKIPPED rather than reported as
passing, because the fake scores lexical overlap, not meaning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    name: str
    kind: str            # exact | paraphrase | unrelated | near_miss
    setup: str
    probe: str
    expect_hit: bool
    requires: str = "any"   # "any" | "semantic"
    note: str = ""


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        name="exact-repeat",
        kind="exact",
        setup="What is a deadlock in an operating system?",
        probe="What is a deadlock in an operating system?",
        expect_hit=True,
        note="Identical text must always be reused.",
    ),
    Fixture(
        name="exact-repeat-whitespace",
        kind="exact",
        setup="What is virtual memory?",
        probe="  What is virtual memory?  ",
        expect_hit=True,
        note="Incidental whitespace must not defeat reuse.",
    ),
    Fixture(
        name="paraphrase-lexical",
        kind="paraphrase",
        setup="What is the difference between a process and a thread?",
        probe="What is the difference between a thread and a process?",
        expect_hit=True,
        note="Reordered but same words; works under either embedder.",
    ),
    Fixture(
        name="paraphrase-semantic",
        kind="paraphrase",
        setup="What is a deadlock in an operating system?",
        probe="Explain how an OS can end up in a circular wait it cannot escape.",
        expect_hit=True,
        requires="semantic",
        note="Same intent, almost no shared vocabulary.",
    ),
    Fixture(
        name="unrelated",
        kind="unrelated",
        setup="What is a deadlock in an operating system?",
        probe="How long should I proof sourdough dough before baking?",
        expect_hit=False,
        note="Different domain entirely; reuse here would be a clear error.",
    ),
    Fixture(
        name="near-miss-definition-vs-prevention",
        kind="near_miss",
        setup="What is a deadlock in an operating system?",
        probe="How do you prevent a deadlock in an operating system?",
        expect_hit=False,
        note="Heavy overlap, different intent: definition vs. prevention.",
    ),
    Fixture(
        name="near-miss-time-vs-space",
        kind="near_miss",
        setup="What is the time complexity of quicksort?",
        probe="What is the space complexity of quicksort?",
        expect_hit=False,
        note="One word apart, materially different answer.",
    ),
    Fixture(
        name="near-miss-advantage-vs-disadvantage",
        kind="near_miss",
        setup="What are the advantages of paging in memory management?",
        probe="What are the disadvantages of paging in memory management?",
        expect_hit=False,
        note="Polar opposite intent under near-identical wording.",
    ),
)


@dataclass
class FixtureResult:
    fixture: Fixture
    status: str                 # PASS | FAIL | SKIP
    hit: bool | None = None
    similarity: float | None = None


def run_fixtures(gateway, fixtures=FIXTURES, *, embedder_is_semantic: bool) -> list[FixtureResult]:
    from gateway.app import QueryRequest

    results: list[FixtureResult] = []
    for fx in fixtures:
        if fx.requires == "semantic" and not embedder_is_semantic:
            results.append(FixtureResult(fx, "SKIP"))
            continue

        # Each fixture starts from an empty cache so the probe can only match
        # its own setup query.
        gateway.cache.reset()
        gateway.handle(QueryRequest(query=fx.setup, variant_kind="novel"))
        probe = gateway.handle(QueryRequest(query=fx.probe, variant_kind=fx.kind))

        status = "PASS" if probe.hit == fx.expect_hit else "FAIL"
        results.append(FixtureResult(fx, status, probe.hit, probe.top1_similarity))
    return results


def format_results(results: list[FixtureResult]) -> str:
    lines = [f"{'fixture':42s} {'kind':11s} {'want':5s} {'got':5s} {'sim':>6s}  status"]
    lines.append("-" * 88)
    for r in results:
        sim = "  -   " if r.similarity is None else f"{r.similarity:6.3f}"
        got = "-" if r.hit is None else str(r.hit).lower()
        lines.append(
            f"{r.fixture.name:42s} {r.fixture.kind:11s} "
            f"{str(r.fixture.expect_hit).lower():5s} {got:5s} {sim}  {r.status}"
        )
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    lines.append("-" * 88)
    lines.append(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Layer-1 reuse correctness fixtures.")
    ap.add_argument("--fake", action="store_true",
                    help="Use in-process fakes (no Docker/GPU/network).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the reuse threshold.")
    ap.add_argument("--metrics-db", default=None, help="SQLite path for the run.")
    args = ap.parse_args()

    from gateway.app import Gateway
    from gateway.config import load_settings

    settings = load_settings()
    if args.fake:
        settings.embedder_impl = "bagofwords"
        settings.cache_impl = "memory"
        settings.backend_impl = "echo"
    if args.threshold is not None:
        settings.reuse_threshold = args.threshold
    if args.metrics_db:
        settings.metrics_db = args.metrics_db

    semantic = settings.embedder_impl.lower() in {"minilm", "sentence-transformers", "real"}
    gateway = Gateway(settings)

    results = run_fixtures(gateway, embedder_is_semantic=semantic)
    print(f"embedder={settings.embedder_impl}  cache={settings.cache_impl}  "
          f"backend={settings.backend_impl}  threshold={settings.reuse_threshold}")
    print(format_results(results))
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
