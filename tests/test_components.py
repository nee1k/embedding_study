"""Embedder, cache store, and benchmark loader behaviour."""

import numpy as np
import pytest

from gateway.cache import CachedResponse, InMemoryCache, _escape_tag
from gateway.embedder import BagOfWordsEmbedder


def _payload(text="answer"):
    return CachedResponse(response_text=text, prompt_tokens=3, completion_tokens=7, query_text="q")


# --- embedder -------------------------------------------------------------


def test_embeddings_are_unit_norm():
    emb = BagOfWordsEmbedder(dim=64)
    vec = emb.encode("What is a deadlock in an operating system?")
    assert vec.shape == (64,)
    assert vec.dtype == np.float32
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-6)


def test_embedding_is_deterministic_across_instances():
    a = BagOfWordsEmbedder(dim=64).encode("what is a mutex")
    b = BagOfWordsEmbedder(dim=64).encode("what is a mutex")
    assert np.allclose(a, b)


def test_stopwords_and_case_are_ignored():
    emb = BagOfWordsEmbedder(dim=128)
    assert np.allclose(emb.encode("What is a Deadlock?"), emb.encode("deadlock"))


def test_empty_query_yields_zero_vector_and_never_matches():
    emb = BagOfWordsEmbedder(dim=32)
    vec = emb.encode("   ???   ")
    assert float(np.linalg.norm(vec)) == 0.0

    cache = InMemoryCache()
    cache.store(emb.encode("what is a deadlock"), response=_payload(),
                model="m", params_hash="p", ttl_s=60)
    hit = cache.search(vec, model="m", params_hash="p")
    assert hit is not None and hit.similarity == pytest.approx(0.0)


# --- cache store ----------------------------------------------------------


def test_search_on_empty_scope_returns_none():
    assert InMemoryCache().search(np.ones(4, dtype=np.float32) / 2, model="m", params_hash="p") is None


def test_search_returns_nearest_regardless_of_threshold():
    """The store never applies a threshold; that decision belongs to the gateway."""
    cache = InMemoryCache()
    near = np.array([1.0, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0], dtype=np.float32)
    cache.store(near, response=_payload("near"), model="m", params_hash="p", ttl_s=60)
    cache.store(far, response=_payload("far"), model="m", params_hash="p", ttl_s=60)

    hit = cache.search(np.array([0.0, 1.0], dtype=np.float32), model="m", params_hash="p")
    assert hit.response.response_text == "far"
    assert hit.similarity == pytest.approx(1.0)


def test_scope_filter_isolates_models_and_params():
    cache = InMemoryCache()
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.store(vec, response=_payload(), model="m1", params_hash="p1", ttl_s=60)

    assert cache.search(vec, model="m2", params_hash="p1") is None
    assert cache.search(vec, model="m1", params_hash="p2") is None
    assert cache.search(vec, model="m1", params_hash="p1") is not None


def test_zero_ttl_means_no_expiry():
    cache = InMemoryCache()
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.store(vec, response=_payload(), model="m", params_hash="p", ttl_s=0)
    assert cache.search(vec, model="m", params_hash="p") is not None


def test_reset_clears_entries():
    cache = InMemoryCache()
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.store(vec, response=_payload(), model="m", params_hash="p", ttl_s=60)
    cache.reset()
    assert cache.search(vec, model="m", params_hash="p") is None


def test_tag_escaping_covers_model_name_punctuation():
    """Unescaped '-' or '.' in a TAG value would break the Valkey scope filter."""
    escaped = _escape_tag("meta-llama/Llama-4-17b.v1")
    for ch in "-/.":
        assert f"\\{ch}" in escaped
    assert "\\-" in escaped


# --- benchmark loader -----------------------------------------------------


def test_benchmark_loads_75_queries_across_7_categories():
    from bench.queries import CATEGORIES, category_counts, load_benchmark

    queries = load_benchmark()
    assert len(queries) == 75

    counts = category_counts(queries)
    assert sum(counts.values()) == 75
    assert set(counts) == set(CATEGORIES)
    # Distribution recorded in the plan; guards against a silent data swap.
    assert counts == {
        "multi-page": 14, "textual": 14, "conceptual": 13,
        "multi-modal": 10, "visual": 9, "numerical": 8, "tabular": 7,
    }


def test_category_labels_are_normalized():
    """The source CSV stores 'multi-modal ' with a trailing space."""
    from bench.queries import load_benchmark

    for q in load_benchmark():
        assert q.category == q.category.strip().lower()
        assert q.query, f"{q.query_id} has empty query text"


def test_ground_truth_page_ranges_expand():
    from bench.queries import _parse_ground_truth

    assert _parse_ground_truth("([89, 91], [97, 99])") == (89, 90, 91, 97, 98, 99)
    assert _parse_ground_truth("") == ()
    assert _parse_ground_truth("garbage") == ()


def test_every_benchmark_query_has_ground_truth_pages():
    from bench.queries import load_benchmark

    assert all(q.ground_truth_pages for q in load_benchmark())


# --- stream construction --------------------------------------------------


def test_stream_places_every_original_before_its_variants():
    from bench.run_stream import build_stream

    stream = build_stream(seed=7)
    first_seen: dict[str, int] = {}
    for idx, item in enumerate(stream):
        if item.variant_kind == "novel":
            first_seen.setdefault(item.variant_of, idx)

    for idx, item in enumerate(stream):
        if item.variant_kind != "novel":
            assert first_seen[item.variant_of] < idx, (
                f"{item.variant_kind} for {item.variant_of} appeared before its original"
            )


def test_stream_is_reproducible_for_a_seed():
    from bench.run_stream import build_stream

    assert [i.text for i in build_stream(seed=7)] == [i.text for i in build_stream(seed=7)]
    assert [i.text for i in build_stream(seed=7)] != [i.text for i in build_stream(seed=8)]


def test_fixtures_pass_on_fakes():
    from bench.fixtures import run_fixtures
    from gateway.app import Gateway
    from gateway.config import Settings
    import tempfile, os

    s = Settings()
    s.embedder_impl, s.cache_impl, s.backend_impl = "bagofwords", "memory", "echo"
    s.metrics_db = os.path.join(tempfile.mkdtemp(), "m.sqlite")
    s.reuse_threshold = 0.95

    results = run_fixtures(Gateway(s), embedder_is_semantic=False)
    failures = [r.fixture.name for r in results if r.status == "FAIL"]
    assert not failures, f"fixtures failed: {failures}"
    assert any(r.status == "SKIP" for r in results), "semantic fixture should skip on the fake"
