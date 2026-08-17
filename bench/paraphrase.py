"""Layer-2 paraphrase generation (§5).

Produces variants of each benchmark query spanning near-duplicate to loosely
related, written once to ``data/paraphrases.json`` and committed so the query
stream is reproducible across runs.

Two methods:

* ``llm``      — the real one. Generates variants with the backend LM. Run once
                 on the lab machine; the committed JSON is the artifact.
* ``template`` — deterministic string rewrites. Exists so the stream harness can
                 be exercised with no network. Lexical, not semantic:
                 **harness testing only, never a reported number.**

Calibration warning (§5): thresholds calibrated on generated paraphrases have
been shown to fail on real query variation. Any threshold derived from this file
is provisional and must be reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass

from bench.queries import BenchmarkQuery, load_benchmark

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "paraphrases.json")

# Variant kinds, ordered by expected semantic distance from the original.
KIND_NEAR_DUPLICATE = "near_duplicate"
KIND_PARAPHRASE = "paraphrase"
KIND_LOOSE = "loose"

GEN_PROMPT = """You are helping build an evaluation set for a semantic caching study.

Given a source query, write exactly three variants:

1. NEAR_DUPLICATE: same question, trivially reworded (a synonym or reordering).
2. PARAPHRASE: same information need, substantially different wording.
3. LOOSE: same topic, but a genuinely DIFFERENT information need — answering the
   source query should NOT answer this one.

Return only a JSON object with keys "near_duplicate", "paraphrase", "loose".
No commentary.

Source query: {query}"""


@dataclass
class Variant:
    query_id: str
    kind: str
    text: str


def _template_variants(q: BenchmarkQuery) -> list[Variant]:
    """Deterministic rewrites. Lexical only — harness testing, not measurement."""
    text = q.query.strip().rstrip("?").strip()
    lowered = text[0].lower() + text[1:] if text else text

    near = re.sub(r"^How do\b", "In what way do", text, flags=re.I)
    if near == text:
        near = re.sub(r"^What (is|are)\b", r"What exactly \1", text, flags=re.I)
    if near == text:
        near = f"{text}, specifically"

    para = f"Could you explain {lowered}"
    loose = f"What are the practical limitations of {lowered}"

    return [
        Variant(q.query_id, KIND_NEAR_DUPLICATE, f"{near}?"),
        Variant(q.query_id, KIND_PARAPHRASE, f"{para}?"),
        Variant(q.query_id, KIND_LOOSE, f"{loose}?"),
    ]


def _llm_variants(q: BenchmarkQuery, backend) -> list[Variant]:
    raw = backend.complete(GEN_PROMPT.format(query=q.query)).text
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"No JSON object in generation for {q.query_id}: {raw[:200]}")
    data = json.loads(match.group(0))
    return [
        Variant(q.query_id, KIND_NEAR_DUPLICATE, str(data["near_duplicate"]).strip()),
        Variant(q.query_id, KIND_PARAPHRASE, str(data["paraphrase"]).strip()),
        Variant(q.query_id, KIND_LOOSE, str(data["loose"]).strip()),
    ]


def generate(method: str, out_path: str = OUT_PATH) -> dict:
    queries = load_benchmark()
    backend = None
    if method == "llm":
        from gateway.backend import build_backend
        from gateway.config import load_settings

        settings = load_settings()
        settings.backend_impl = "litellm"
        backend = build_backend(settings)

    variants: list[Variant] = []
    for q in queries:
        variants.extend(
            _llm_variants(q, backend) if method == "llm" else _template_variants(q)
        )

    payload = {
        "method": method,
        "source": "bench/data/benchmark75.csv",
        "n_source_queries": len(queries),
        "n_variants": len(variants),
        "warning": (
            "Thresholds calibrated on generated paraphrases are provisional; "
            "they are not validated against real query logs."
            + (
                "  METHOD=template: lexical rewrites for harness testing only, "
                "not for reported numbers."
                if method == "template"
                else ""
            )
        ),
        "variants": [asdict(v) for v in variants],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def load_variants(path: str = OUT_PATH) -> list[Variant]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [Variant(**v) for v in payload["variants"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate paraphrase variants.")
    ap.add_argument("--method", choices=["llm", "template"], default="template")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    payload = generate(args.method, args.out)
    print(f"method={payload['method']}  "
          f"{payload['n_variants']} variants from {payload['n_source_queries']} queries")
    print(f"wrote {args.out}")
    if payload["method"] == "template":
        print("NOTE: template variants are lexical only — harness testing, not measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
