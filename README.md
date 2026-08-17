# Semantic Result Reuse — Proof of Concept

Request-layer cost and latency reduction for library-scale discovery: if a query is
semantically equivalent to one already answered, skip the pipeline entirely rather than
making it faster.

This is the companion to the Vector Study
([`vector-volume-discovery-eval`](https://github.com/Plale-Lab/vector-volume-discovery-eval)),
which attacked the same cost at the **index layer** (ColPali + Qdrant, HNSW, binary
quantization). The two levers are orthogonal: the index work reduces the cost of *doing*
the work, this reduces *how often the work is done at all*.

## Architecture

```
client ──> Gateway (FastAPI)
             ├─ embed query (MiniLM, 384-d, normalized)     [every request]
             ├─ Valkey KNN top-1, scoped to model+params    [cache-enabled requests]
             ├─ hit  → return the stored response
             ├─ miss → Llama 4-17B (LiteLLM/Tapis) → return → store in background
             └─ log one row to SQLite                       [every request]
```

The gateway is thin by design: no logic beyond the cache decision. Each external —
embedder, cache store, backend LM — sits behind a small protocol with a deterministic
in-process fake, so the whole system is testable with no GPU, no Docker, and no network.

### Two design decisions worth knowing

**Similarity is logged on every request, including misses.** The nearest neighbour's score
is recorded even when it loses to the threshold. Hit rate at any candidate threshold is
therefore re-derivable from a single run (`bench/report.py` sweeps it), so narrowing the
threshold costs no extra inference calls. The sweep holds cache *contents* fixed at what
the run actually stored, so it is a counterfactual estimate — good for narrowing the
search, not a substitute for re-running at the chosen threshold.

**Reuse is scoped, not just thresholded.** An entry is only a candidate if it came from the
same model with the same generation parameters (`model` and `params_hash` TAG pre-filter).
Similarity alone never authorizes reuse across incomparable requests.

## Quickstart — runs anywhere

No GPU, Docker, model weights, or network needed. Everything below uses the fakes.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                  # 34 tests
python -m bench.fixtures --fake            # Layer-1 reuse correctness fixtures

python -m bench.paraphrase --method template
python -m bench.run_stream --fake --out runs/smoke.sqlite --run-id smoke
python -m bench.judge  --db runs/smoke.sqlite --run-id smoke --method design
python -m bench.report --db runs/smoke.sqlite --run-id smoke
```

> The fakes are logic fixtures. `BagOfWordsEmbedder` scores lexical overlap rather than
> meaning, `EchoBackend` returns a deterministic string, and `--method design`/`template`
> describe the stream's construction rather than any model's behaviour. **No number from a
> fake run is reportable.** They exist so the harness can be exercised and regressions
> caught without spending inference calls.

## Real measurement run — lab machine

Requires Docker, the MiniLM weights (downloaded on first use), and Tapis credentials.

```bash
pip install -r requirements.txt -r requirements-lab.txt

docker compose up -d                       # valkey/valkey-bundle (ships valkey-search)
docker compose exec valkey valkey-cli ping

cp .env.example .env                       # set TAPIS_BASE_URL / TAPIS_API_KEY,
                                           # and EMBEDDER_IMPL=minilm CACHE_IMPL=valkey
                                           # BACKEND_IMPL=litellm
uvicorn gateway.app:create_app --factory --port 8080
```

End-to-end check — the same query twice, second should hit with lower latency:

```bash
curl -s localhost:8080/query -H 'content-type: application/json' \
     -d '{"query":"What is a deadlock?"}'
curl -s localhost:8080/query -H 'content-type: application/json' \
     -d '{"query":"What is a deadlock?"}'
curl -s localhost:8080/stats
```

Full Phase 2 measurement:

```bash
python -m bench.paraphrase --method llm         # regenerates bench/data/paraphrases.json
python -m bench.run_stream --out runs/phase2.sqlite --run-id phase2
python -m bench.judge  --db runs/phase2.sqlite --run-id phase2 --method llm
python -m bench.report --db runs/phase2.sqlite --run-id phase2
```

`bench/fixtures.py` run against the real embedder also exercises the `paraphrase-semantic`
fixture, which is skipped under the fake.

## What the report produces

Hit rate; p50/p95/p99 split by hit path vs miss path with the embed leg isolated (it is the
hit-path floor); tokens and dollars avoided; false-reuse rate; the similarity distributions
for correct vs. incorrect reuse; and all of it sliced by the seven query categories — plus
the threshold sweep.

The separability section is the scientifically interesting output. If the correct and
incorrect distributions overlap heavily — as the text-only caching literature reports —
that is a publishable negative result, and it redirects the work toward scoping signals
rather than threshold tuning.

## Layout

| Path | Role |
|---|---|
| `gateway/app.py` | Cache decision + FastAPI transport. `Gateway` is directly instantiable in tests. |
| `gateway/embedder.py` | `Embedder` protocol; MiniLM + bag-of-words fake |
| `gateway/cache.py` | `CacheStore` protocol; Valkey (`FT.CREATE`/`FT.SEARCH`) + in-memory fake |
| `gateway/backend.py` | `LMBackend` protocol; LiteLLM/Tapis + echo fake |
| `gateway/metrics.py` | SQLite schema and writer, one row per request |
| `bench/queries.py` | Loads the 75-query benchmark, normalizes categories |
| `bench/fixtures.py` | Layer-1 hand-written correctness fixtures |
| `bench/paraphrase.py` | Layer-2 variant generation |
| `bench/run_stream.py` | Mixed stream, cache-on and cache-off arms |
| `bench/judge.py` | Layer-3 correctness labeling (Qwen3-32B) |
| `bench/report.py` | Phase 2 tables and threshold sweep |

## Data

`bench/data/benchmark75.csv` is vendored from
[`Plale-Lab/vector-volume-discovery-eval`](https://github.com/Plale-Lab/vector-volume-discovery-eval)
(`Vector Discovery Benchmark 75.csv`): 75 queries across seven categories — multi-page 14,
textual 14, conceptual 13, multi-modal 10, visual 9, numerical 8, tabular 7 — with source
book and ground-truth page ranges. No copyrighted page content travels with it.

Category labels are normalized on load; the source stores `"multi-modal "` with a trailing
space, which would otherwise split one category into two.

## Scope and caveats

Phases 0–2 (baseline → naive cache → measurement). Explicit non-goals: multi-turn handling,
an admission classifier, eviction beyond TTL, and production hardening. Each is a candidate
research extension, not POC scope.

**Thresholds derived here are provisional.** There are no real user query logs, so
calibration rests on generated paraphrases — precisely the setup the literature reports as
failing on real query variation. Any writeup must say so rather than bury it.

The ground-truth page ranges in the benchmark are carried through the loader but unused by
this text-in/text-out POC. They are what would make reuse correctness *objectively*
measurable in the retrieval-modality extension, without an LLM judge.
