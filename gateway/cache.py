"""Semantic cache over prior queries.

Two implementations behind one protocol: ``ValkeyCache`` (valkey-search: HNSW,
cosine, TAG pre-filter, native key TTL) and ``InMemoryCache`` (numpy brute
force, identical filter and TTL semantics) for tests.

Design note: ``search`` returns the nearest entry *regardless of threshold*.
The accept/reject decision lives in the gateway so that the similarity of a
rejected near-miss is still logged, which is what makes the threshold sweep
re-derivable from a single run.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class CachedResponse:
    """The payload stored alongside a cache-key vector."""

    response_text: str
    prompt_tokens: int
    completion_tokens: int
    query_text: str


@dataclass
class CacheHit:
    """Nearest neighbour found, before any threshold is applied."""

    entry_id: str
    similarity: float
    response: CachedResponse


# RediSearch/valkey-search treat this punctuation as syntax inside TAG values,
# so it must be backslash-escaped in both the stored value and the query.
_TAG_SPECIAL = set(",.<>{}[]\"'`:;!@#$%^&*()-+=~| /\\?")


def _escape_tag(value: str) -> str:
    return "".join("\\" + ch if ch in _TAG_SPECIAL else ch for ch in value)


@runtime_checkable
class CacheStore(Protocol):
    def search(self, vec: np.ndarray, *, model: str, params_hash: str) -> Optional[CacheHit]:
        """Return the nearest comparable entry, or None if the scope is empty."""
        ...

    def store(
        self,
        vec: np.ndarray,
        *,
        response: CachedResponse,
        model: str,
        params_hash: str,
        ttl_s: int,
    ) -> str:
        """Persist an entry and return its id."""
        ...

    def reset(self) -> None:
        """Drop all entries (test/bench isolation)."""
        ...


class InMemoryCache:
    """Brute-force cosine top-1. Same filter and TTL semantics as ValkeyCache."""

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}

    def _live(self) -> list[tuple[str, dict]]:
        now = time.time()
        expired = [k for k, e in self._entries.items() if e["expires_at"] <= now]
        for k in expired:
            del self._entries[k]
        return list(self._entries.items())

    def search(self, vec: np.ndarray, *, model: str, params_hash: str) -> Optional[CacheHit]:
        candidates = [
            (eid, e)
            for eid, e in self._live()
            if e["model"] == model and e["params_hash"] == params_hash
        ]
        if not candidates:
            return None
        matrix = np.vstack([e["vec"] for _, e in candidates])
        # Vectors are stored L2-normalized, so the dot product is the cosine.
        sims = matrix @ vec.astype(np.float32)
        best = int(np.argmax(sims))
        eid, entry = candidates[best]
        return CacheHit(entry_id=eid, similarity=float(sims[best]), response=entry["response"])

    def store(
        self,
        vec: np.ndarray,
        *,
        response: CachedResponse,
        model: str,
        params_hash: str,
        ttl_s: int,
    ) -> str:
        entry_id = uuid.uuid4().hex
        self._entries[entry_id] = {
            "vec": np.asarray(vec, dtype=np.float32),
            "response": response,
            "model": model,
            "params_hash": params_hash,
            # ttl_s <= 0 means no expiry, matching ValkeyCache (which simply
            # skips the EXPIRE call).
            "expires_at": (time.time() + ttl_s) if ttl_s > 0 else float("inf"),
        }
        return entry_id

    def reset(self) -> None:
        self._entries.clear()


class ValkeyCache:
    """valkey-search backed cache (image: ``valkey/valkey-bundle``).

    Index (created once, idempotently)::

        FT.CREATE <index> ON HASH PREFIX 1 <prefix> SCHEMA
          model TAG params_hash TAG
          embedding VECTOR HNSW 6 TYPE FLOAT32 DIM <dim> DISTANCE_METRIC COSINE
    """

    def __init__(self, host: str, port: int, index: str, prefix: str, dim: int) -> None:
        self.index = index
        self.prefix = prefix
        self.dim = dim
        self._client = self._connect(host, port)
        self.ensure_index()

    @staticmethod
    def _connect(host: str, port: int):
        try:
            import valkey as _driver
        except ImportError:  # valkey-py is a redis-py fork; either driver works
            import redis as _driver
        client = _driver.Redis(host=host, port=port, decode_responses=False)
        client.ping()
        return client

    def ensure_index(self) -> None:
        try:
            self._client.execute_command("FT.INFO", self.index)
            return  # already exists
        except Exception:
            pass
        self._client.execute_command(
            "FT.CREATE", self.index,
            "ON", "HASH",
            "PREFIX", "1", self.prefix,
            "SCHEMA",
            "model", "TAG",
            "params_hash", "TAG",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(self.dim),
            "DISTANCE_METRIC", "COSINE",
        )

    def search(self, vec: np.ndarray, *, model: str, params_hash: str) -> Optional[CacheHit]:
        query = (
            f"(@model:{{{_escape_tag(model)}}} "
            f"@params_hash:{{{_escape_tag(params_hash)}}})"
            f"=>[KNN 1 @embedding $vec AS score]"
        )
        try:
            raw = self._client.execute_command(
                "FT.SEARCH", self.index, query,
                "PARAMS", "2", "vec", np.asarray(vec, dtype=np.float32).tobytes(),
                "RETURN", "5", "score", "response_text", "prompt_tokens",
                "completion_tokens", "query_text",
                "DIALECT", "2",
            )
        except Exception:
            # A search failure must not take down the request path; treat it as
            # a miss so the query still gets a fresh answer.
            return None

        if not raw or int(raw[0]) == 0:
            return None

        entry_id = raw[1].decode() if isinstance(raw[1], bytes) else str(raw[1])
        fields = self._pairs_to_dict(raw[2])
        # COSINE distance in [0, 2]; the gateway thresholds on similarity.
        similarity = 1.0 - float(fields.get("score", 1.0))
        return CacheHit(
            entry_id=entry_id,
            similarity=similarity,
            response=CachedResponse(
                response_text=fields.get("response_text", ""),
                prompt_tokens=int(fields.get("prompt_tokens", 0) or 0),
                completion_tokens=int(fields.get("completion_tokens", 0) or 0),
                query_text=fields.get("query_text", ""),
            ),
        )

    @staticmethod
    def _pairs_to_dict(pairs) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(pairs) - 1, 2):
            key = pairs[i].decode() if isinstance(pairs[i], bytes) else str(pairs[i])
            val = pairs[i + 1]
            out[key] = val.decode() if isinstance(val, bytes) else str(val)
        return out

    def store(
        self,
        vec: np.ndarray,
        *,
        response: CachedResponse,
        model: str,
        params_hash: str,
        ttl_s: int,
    ) -> str:
        entry_id = uuid.uuid4().hex
        key = f"{self.prefix}{entry_id}"
        self._client.hset(
            key,
            mapping={
                "embedding": np.asarray(vec, dtype=np.float32).tobytes(),
                "model": model,
                "params_hash": params_hash,
                "response_text": response.response_text,
                "prompt_tokens": str(response.prompt_tokens),
                "completion_tokens": str(response.completion_tokens),
                "query_text": response.query_text,
            },
        )
        # Native key expiry covers the POC's TTL-only eviction policy.
        if ttl_s > 0:
            self._client.expire(key, ttl_s)
        return entry_id

    def reset(self) -> None:
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{self.prefix}*", count=500)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break


def build_cache(settings) -> CacheStore:
    impl = settings.cache_impl.lower()
    if impl in {"memory", "inmemory", "fake"}:
        return InMemoryCache()
    if impl in {"valkey", "real"}:
        return ValkeyCache(
            host=settings.valkey_host,
            port=settings.valkey_port,
            index=settings.valkey_index,
            prefix=settings.valkey_prefix,
            dim=settings.embedding_dim,
        )
    raise ValueError(f"Unknown CACHE_IMPL: {settings.cache_impl!r}")
