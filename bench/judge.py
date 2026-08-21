"""Layer-3 correctness labeling (§5).

For every cache hit, decide whether the reused response actually answered the
new query. The cache-off arm of the same run supplies the ground-truth response
for comparison.

Two methods:

* ``llm``    — Qwen3-32B, temperature 0.0, ``enable_thinking: False``, one call
               per reuse decision. Deliberately a different model family from
               the Llama 4-17B generator, following the judge protocol already
               established in ``Plale-Lab/patra-metadata-augmentation-study`` to
               limit self-enhancement bias.
* ``design`` — labels from the stream's *design intent* rather than the
               responses: ``exact``/``near_duplicate`` reuse is correct,
               ``loose`` reuse is incorrect, ``paraphrase`` is left unlabeled
               because it is genuinely ambiguous. This exercises the report
               pipeline with no network. It is a statement about the stream, not
               a judgment of the responses — never a reported false-reuse rate.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3

JUDGE_PROMPT = """You are evaluating a semantic cache.

A user asked a NEW QUERY. Instead of running the model, the system reused a
response that had been generated for a DIFFERENT, earlier query.

NEW QUERY:
{query}

REUSED RESPONSE (what the user actually received):
{reused}

REFERENCE RESPONSE (generated fresh for the new query):
{fresh}

Did the reused response adequately answer the new query? Answer strictly as
JSON: {{"correct": true or false, "rationale": "<one sentence>"}}"""

# Reuse on these variant kinds is correct/incorrect by construction.
_DESIGN_LABELS = {"exact": 1, "near_duplicate": 1, "loose": 0}


def _connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _hits_needing_judgment(conn: sqlite3.Connection, run_id: str | None) -> list[sqlite3.Row]:
    """Cache hits joined to the cache-off response for the same query."""
    sql = """
        SELECT h.request_id, h.query_text, h.variant_kind, h.category,
               h.response_text AS reused_text,
               (SELECT b.response_text FROM requests b
                 WHERE b.cache_enabled = 0
                   AND b.query_text = h.query_text
                   AND (b.run_id IS h.run_id)
                 LIMIT 1) AS fresh_text
        FROM requests h
        LEFT JOIN judgments j ON j.request_id = h.request_id
        WHERE h.cache_enabled = 1 AND h.hit = 1 AND j.request_id IS NULL
    """
    params: tuple = ()
    if run_id:
        sql += " AND h.run_id = ?"
        params = (run_id,)
    return conn.execute(sql, params).fetchall()


def _label_design(rows) -> list[tuple]:
    out = []
    for r in rows:
        correct = _DESIGN_LABELS.get(r["variant_kind"])
        if correct is None:
            continue  # paraphrase: genuinely ambiguous, leave unlabeled
        out.append((r["request_id"], correct,
                    f"design intent for variant_kind={r['variant_kind']}", "design"))
    return out


def _label_llm(rows, settings) -> list[tuple]:
    from gateway.backend import LiteLLMBackend

    backend = LiteLLMBackend(
        base_url=settings.tapis_base_url,
        api_key=settings.tapis_api_key,
        model=settings.lm_model,
        temperature=settings.lm_temperature,
        max_tokens=settings.lm_max_tokens,
        timeout_s=settings.lm_timeout_s,
    )

    out = []
    for r in rows:
        prompt = JUDGE_PROMPT.format(
            query=r["query_text"],
            reused=r["reused_text"] or "",
            fresh=r["fresh_text"] or "(no reference response recorded)",
        )
        raw = backend.complete_raw(
            [{"role": "user", "content": prompt}],
            model=settings.judge_model,
            temperature=settings.judge_temperature,
            extra_body={"enable_thinking": False},
        )
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            out.append((r["request_id"], None, f"unparseable judge output: {raw[:160]}",
                        settings.judge_model))
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            out.append((r["request_id"], None, f"invalid JSON: {raw[:160]}", settings.judge_model))
            continue
        out.append((
            r["request_id"],
            1 if bool(data.get("correct")) else 0,
            str(data.get("rationale", ""))[:500],
            settings.judge_model,
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Label cache hits correct/incorrect.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--method", choices=["llm", "design"], default="llm")
    args = ap.parse_args()

    conn = _connect(args.db)
    rows = _hits_needing_judgment(conn, args.run_id)
    if not rows:
        print("no unjudged cache hits found")
        return 0

    if args.method == "design":
        labels = _label_design(rows)
    else:
        from gateway.config import load_settings

        labels = _label_llm(rows, load_settings())

    conn.executemany(
        "INSERT OR REPLACE INTO judgments (request_id, correct, rationale, judge_model) "
        "VALUES (?, ?, ?, ?)",
        labels,
    )
    conn.commit()

    n_correct = sum(1 for _, c, _, _ in labels if c == 1)
    n_incorrect = sum(1 for _, c, _, _ in labels if c == 0)
    print(f"method={args.method}  judged {len(labels)} of {len(rows)} hits "
          f"({n_correct} correct, {n_incorrect} incorrect, "
          f"{len(rows) - len(labels)} left unlabeled)")
    if args.method == "design":
        print("NOTE: design labels describe the stream's construction, not the "
              "responses — not a reportable false-reuse rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
