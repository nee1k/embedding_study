"""Environment-driven settings for the semantic reuse gateway.

Mirrors the layout of the Vector Study's ``config.py`` but reads from the
environment rather than hardcoding hosts, so the same image runs against the
in-process fakes (tests/CI) and the real Valkey + Tapis stack (lab machine).
"""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


@dataclass
class Settings:
    """Runtime configuration.

    ``*_impl`` fields select between the real backend and the deterministic
    fake. The fakes exist so gateway logic is testable without Docker, a GPU,
    or network access; they must never produce a reported number.
    """

    # --- Component selection -------------------------------------------------
    embedder_impl: str = field(default_factory=lambda: os.getenv("EMBEDDER_IMPL", "bagofwords"))
    cache_impl: str = field(default_factory=lambda: os.getenv("CACHE_IMPL", "memory"))
    backend_impl: str = field(default_factory=lambda: os.getenv("BACKEND_IMPL", "echo"))

    # --- Cache behaviour -----------------------------------------------------
    # The reuse threshold is compared against cosine *similarity* in [-1, 1].
    reuse_threshold: float = field(default_factory=lambda: _env_float("REUSE_THRESHOLD", 0.95))
    cache_ttl_s: int = field(default_factory=lambda: _env_int("CACHE_TTL_S", 24 * 3600))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))

    # --- Embedding -----------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # --- Valkey --------------------------------------------------------------
    valkey_host: str = field(default_factory=lambda: os.getenv("VALKEY_HOST", "localhost"))
    valkey_port: int = field(default_factory=lambda: _env_int("VALKEY_PORT", 6379))
    valkey_index: str = field(default_factory=lambda: os.getenv("VALKEY_INDEX", "semcache"))
    valkey_prefix: str = field(default_factory=lambda: os.getenv("VALKEY_PREFIX", "semcache:"))

    # --- Backend LM (Llama 4-17B via LiteLLM on TACC Tapis) ------------------
    tapis_base_url: str = field(default_factory=lambda: os.getenv("TAPIS_BASE_URL", ""))
    tapis_api_key: str = field(default_factory=lambda: os.getenv("TAPIS_API_KEY", ""))
    lm_model: str = field(default_factory=lambda: os.getenv("LM_MODEL", "llama4-17b"))
    lm_temperature: float = field(default_factory=lambda: _env_float("LM_TEMPERATURE", 0.2))
    lm_max_tokens: int = field(default_factory=lambda: _env_int("LM_MAX_TOKENS", 512))
    lm_timeout_s: float = field(default_factory=lambda: _env_float("LM_TIMEOUT_S", 120.0))

    # --- Judge (Qwen3-32B, different model family than the generator) --------
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "qwen3-32b"))
    judge_temperature: float = field(default_factory=lambda: _env_float("JUDGE_TEMPERATURE", 0.0))

    # --- Cost ----------------------------------------------------------------
    # Tapis is grant-funded, so dollars may be zero/unpublished. Tokens avoided
    # is the primary unit; dollars are derived from this configurable rate.
    usd_per_1k_prompt_tokens: float = field(
        default_factory=lambda: _env_float("USD_PER_1K_PROMPT_TOKENS", 0.0)
    )
    usd_per_1k_completion_tokens: float = field(
        default_factory=lambda: _env_float("USD_PER_1K_COMPLETION_TOKENS", 0.0)
    )

    # --- Metrics -------------------------------------------------------------
    metrics_db: str = field(default_factory=lambda: os.getenv("METRICS_DB", "runs/metrics.sqlite"))

    def params_hash(self) -> str:
        """Scope key so only comparable requests can be reused for one another.

        Two requests are reusable only if they went to the same model with the
        same generation parameters; this becomes a TAG pre-filter on search.
        """
        import hashlib

        raw = f"{self.lm_model}|{self.lm_temperature}|{self.lm_max_tokens}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_settings() -> Settings:
    return Settings()
