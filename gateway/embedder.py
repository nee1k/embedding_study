"""Query embedding — the cache key.

Two implementations behind one protocol:

* ``MiniLMEmbedder``   — the real one (all-MiniLM-L6-v2, 384-d, CPU).
* ``BagOfWordsEmbedder`` — a deterministic fake for tests. Paraphrases that
  share vocabulary land near each other and unrelated queries do not, which is
  enough to exercise hit/miss logic without model weights or network access.
  It is a logic fixture only: no reported number may come from it.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words carrying no topical signal; dropping them keeps the fake embedder from
# calling every question a paraphrase of every other question.
_STOPWORDS = frozenset(
    """a an the is are was were be been being do does did what which who whom how why when
    where of in on at to for from by with about into over after under between and or as
    if then than that this these those it its can could should would may might will
    explain describe define discuss compare list give tell me you your i""".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    """Maps query text to a unit-norm vector used as the cache key."""

    dim: int

    def encode(self, text: str) -> np.ndarray:
        """Return an L2-normalized float32 vector of length ``dim``."""
        ...


class BagOfWordsEmbedder:
    """Deterministic hashed bag-of-words. Test fake — never a reported number.

    Tokens are hashed into a fixed-width vector with term-frequency weights, so
    the result is stable across processes (unlike Python's salted ``hash``) and
    cosine similarity tracks vocabulary overlap.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            # Keep an all-zero vector rather than inventing signal; cosine
            # against it is 0, so an empty query can never trigger reuse.
            return vec
        import hashlib

        for tok in tokens:
            digest = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via sentence-transformers. Runs on CPU.

    The model is loaded lazily so that importing this module does not require
    the weights to be present.
    """

    def __init__(self, model_name: str, dim: int = 384, device: str = "cpu") -> None:
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dim:
                raise ValueError(
                    f"{self.model_name} produces {actual}-d embeddings but "
                    f"EMBEDDING_DIM is {self.dim}; the Valkey index would be "
                    f"built with the wrong DIM."
                )
        return self._model

    def encode(self, text: str) -> np.ndarray:
        model = self._load()
        # normalize_embeddings makes cosine similarity a plain dot product and
        # matches how vectors are stored in the cache.
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)


def build_embedder(settings) -> Embedder:
    impl = settings.embedder_impl.lower()
    if impl in {"bagofwords", "bow", "fake"}:
        return BagOfWordsEmbedder(dim=settings.embedding_dim)
    if impl in {"minilm", "sentence-transformers", "real"}:
        return MiniLMEmbedder(settings.embedding_model, dim=settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDER_IMPL: {settings.embedder_impl!r}")
