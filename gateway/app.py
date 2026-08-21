"""Inference gateway: cache check, forward, store.

Thin by design (§3) — no business logic beyond the cache decision. The
``Gateway`` class holds the logic and is directly instantiable in tests; the
FastAPI app below is only transport.

Token accounting convention: ``prompt_tokens``/``completion_tokens`` are the
tokens *attributable* to a request. On a miss those are the tokens actually
spent; on a hit they are the tokens that would have been spent, estimated from
the reused entry. That makes "cost avoided" a sum over hits.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

# Imported at module scope, not inside create_app: this module uses
# `from __future__ import annotations`, so FastAPI resolves handler annotations
# as strings against module globals. A function-local import would leave
# BackgroundTasks unresolvable and FastAPI would treat it as a query parameter.
from fastapi import BackgroundTasks, FastAPI, Header
from pydantic import BaseModel, Field

from gateway.backend import build_backend
from gateway.cache import CachedResponse, build_cache
from gateway.config import Settings, load_settings
from gateway.embedder import build_embedder
from gateway.metrics import MetricsStore, RequestRecord, compute_cost


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # Optional labels so the bench harness can tag rows without a side channel.
    category: Optional[str] = None
    variant_of: Optional[str] = None
    variant_kind: Optional[str] = None
    run_id: Optional[str] = None


class QueryResponse(BaseModel):
    request_id: str
    response: str
    hit: bool
    top1_similarity: Optional[float]
    threshold: float
    latency_total_ms: float
    latency_embed_ms: float
    latency_search_ms: float
    latency_backend_ms: float


class Gateway:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder=None,
        cache=None,
        backend=None,
        metrics=None,
    ) -> None:
        self.settings = settings or load_settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.cache = cache or build_cache(self.settings)
        self.backend = backend or build_backend(self.settings)
        self.metrics = metrics or MetricsStore(self.settings.metrics_db)

    def handle(
        self,
        req: QueryRequest,
        *,
        bypass_cache: bool = False,
        defer: Optional[Callable[[Callable[[], None]], None]] = None,
    ) -> QueryResponse:
        """Serve one query.

        ``bypass_cache`` drives the cache-disabled arm used for ground-truth
        comparison. ``defer`` schedules the post-response store; when None the
        store runs inline (tests).
        """
        settings = self.settings
        cache_enabled = settings.cache_enabled and not bypass_cache
        request_id = uuid.uuid4().hex
        t_start = time.perf_counter()

        # --- Embed (every request, exactly once) -----------------------------
        t0 = time.perf_counter()
        vec = self.embedder.encode(req.query)
        latency_embed_ms = (time.perf_counter() - t0) * 1000.0

        latency_search_ms = 0.0
        latency_backend_ms = 0.0
        top1_similarity: Optional[float] = None
        reused_from_id: Optional[str] = None
        hit = False
        response_text: str
        prompt_tokens = 0
        completion_tokens = 0

        # --- Search (top-1, scoped to comparable requests) -------------------
        nearest = None
        if cache_enabled:
            t0 = time.perf_counter()
            nearest = self.cache.search(
                vec, model=settings.lm_model, params_hash=settings.params_hash()
            )
            latency_search_ms = (time.perf_counter() - t0) * 1000.0
            if nearest is not None:
                # Recorded even when it loses to the threshold — this is what
                # makes the threshold sweep re-derivable offline.
                top1_similarity = nearest.similarity
                hit = nearest.similarity >= settings.reuse_threshold

        if hit and nearest is not None:
            response_text = nearest.response.response_text
            reused_from_id = nearest.entry_id
            prompt_tokens = nearest.response.prompt_tokens
            completion_tokens = nearest.response.completion_tokens
        else:
            # --- Miss: forward to the backend LM ------------------------------
            t0 = time.perf_counter()
            lm = self.backend.complete(req.query)
            latency_backend_ms = (time.perf_counter() - t0) * 1000.0
            response_text = lm.text
            prompt_tokens = lm.prompt_tokens
            completion_tokens = lm.completion_tokens

            if cache_enabled:
                payload = CachedResponse(
                    response_text=lm.text,
                    prompt_tokens=lm.prompt_tokens,
                    completion_tokens=lm.completion_tokens,
                    query_text=req.query,
                )

                def _store() -> None:
                    self.cache.store(
                        vec,
                        response=payload,
                        model=settings.lm_model,
                        params_hash=settings.params_hash(),
                        ttl_s=settings.cache_ttl_s,
                    )

                # Off the response path when a scheduler is available.
                if defer is not None:
                    defer(_store)
                else:
                    _store()

        latency_total_ms = (time.perf_counter() - t_start) * 1000.0

        self.metrics.record(
            RequestRecord(
                request_id=request_id,
                ts=time.time(),
                run_id=req.run_id,
                query_text=req.query,
                category=req.category,
                variant_of=req.variant_of,
                variant_kind=req.variant_kind,
                cache_enabled=cache_enabled,
                hit=hit,
                top1_similarity=top1_similarity,
                threshold=settings.reuse_threshold,
                reused_from_id=reused_from_id,
                response_text=response_text,
                latency_total_ms=latency_total_ms,
                latency_embed_ms=latency_embed_ms,
                latency_search_ms=latency_search_ms,
                latency_backend_ms=latency_backend_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=compute_cost(prompt_tokens, completion_tokens, settings),
            )
        )

        return QueryResponse(
            request_id=request_id,
            response=response_text,
            hit=hit,
            top1_similarity=top1_similarity,
            threshold=settings.reuse_threshold,
            latency_total_ms=latency_total_ms,
            latency_embed_ms=latency_embed_ms,
            latency_search_ms=latency_search_ms,
            latency_backend_ms=latency_backend_ms,
        )


def create_app(gateway: Optional[Gateway] = None):
    app = FastAPI(title="Semantic Result Reuse Gateway")
    gw = gateway or Gateway()
    app.state.gateway = gw

    @app.post("/query", response_model=QueryResponse)
    def query(
        req: QueryRequest,
        background: BackgroundTasks,
        x_cache_bypass: Optional[str] = Header(default=None),
    ) -> QueryResponse:
        bypass = str(x_cache_bypass or "").strip().lower() in {"1", "true", "yes"}
        return app.state.gateway.handle(
            req, bypass_cache=bypass, defer=background.add_task
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict:
        s = app.state.gateway.settings
        return {
            **app.state.gateway.metrics.summary(),
            "threshold": s.reuse_threshold,
            "embedder": s.embedder_impl,
            "cache": s.cache_impl,
            "backend": s.backend_impl,
            "model": s.lm_model,
        }

    return app
