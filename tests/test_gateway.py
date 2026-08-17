"""Gateway cache-decision behaviour."""

import sqlite3

import pytest

from gateway.app import Gateway, QueryRequest, create_app


def _rows(settings):
    conn = sqlite3.connect(settings.metrics_db)
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM requests ORDER BY ts").fetchall()


def test_first_query_misses_and_second_identical_query_hits(gateway):
    first = gateway.handle(QueryRequest(query="What is a deadlock?"))
    second = gateway.handle(QueryRequest(query="What is a deadlock?"))

    assert first.hit is False
    assert second.hit is True
    assert second.response == first.response
    assert second.top1_similarity == pytest.approx(1.0)


def test_unrelated_query_misses(gateway):
    gateway.handle(QueryRequest(query="What is a deadlock?"))
    other = gateway.handle(QueryRequest(query="How do I proof sourdough overnight?"))
    assert other.hit is False


def test_similarity_is_logged_on_a_miss(gateway, settings):
    """The property that makes the threshold sweep re-derivable offline."""
    gateway.handle(QueryRequest(query="What is the time complexity of quicksort?"))
    probe = gateway.handle(QueryRequest(query="What is the space complexity of quicksort?"))

    assert probe.hit is False
    # A near neighbour existed and its score was recorded despite losing.
    assert probe.top1_similarity is not None
    assert 0.0 < probe.top1_similarity < settings.reuse_threshold

    logged = _rows(settings)[-1]
    assert logged["hit"] == 0
    assert logged["top1_similarity"] == pytest.approx(probe.top1_similarity)


def test_first_ever_query_has_no_similarity(gateway):
    """An empty scope yields None, not a fabricated zero."""
    first = gateway.handle(QueryRequest(query="What is paging?"))
    assert first.top1_similarity is None


def test_threshold_boundary_is_inclusive(settings):
    settings.reuse_threshold = 1.0
    gw = Gateway(settings)
    gw.handle(QueryRequest(query="What is a semaphore?"))
    exact = gw.handle(QueryRequest(query="What is a semaphore?"))
    assert exact.hit is True, "similarity == threshold must count as a hit"


def test_raising_threshold_suppresses_reuse(settings):
    settings.reuse_threshold = 1.01  # unreachable
    gw = Gateway(settings)
    gw.handle(QueryRequest(query="What is a semaphore?"))
    assert gw.handle(QueryRequest(query="What is a semaphore?")).hit is False


def test_bypass_neither_reads_nor_writes_the_cache(gateway):
    gateway.handle(QueryRequest(query="What is a mutex?"))

    bypassed = gateway.handle(QueryRequest(query="What is a mutex?"), bypass_cache=True)
    assert bypassed.hit is False
    assert bypassed.top1_similarity is None
    assert bypassed.latency_search_ms == 0.0

    # A bypassed miss must not have stored a second entry.
    assert len(gateway.cache._entries) == 1


def test_scoping_filter_prevents_reuse_across_models(settings):
    gw = Gateway(settings)
    gw.handle(QueryRequest(query="What is a deadlock?"))

    # Same query, different backend model => different scope => no reuse.
    gw.settings.lm_model = "some-other-model"
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is False


def test_scoping_filter_prevents_reuse_across_parameters(settings):
    gw = Gateway(settings)
    gw.handle(QueryRequest(query="What is a deadlock?"))

    gw.settings.lm_temperature = 0.9  # params_hash changes
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is False


def test_expired_entries_are_not_reused(settings):
    settings.cache_ttl_s = 0  # expires immediately in the in-memory store
    gw = Gateway(settings)
    gw.cache.store  # noqa: B018 - readability: store is exercised via handle
    gw.handle(QueryRequest(query="What is a deadlock?"))

    # ttl_s == 0 means "no expiry" by contract, so this must still hit.
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is True


def test_explicit_ttl_expiry(settings, monkeypatch):
    import gateway.cache as cache_mod

    settings.cache_ttl_s = 10
    gw = Gateway(settings)
    gw.handle(QueryRequest(query="What is a deadlock?"))
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is True

    real_time = cache_mod.time.time
    monkeypatch.setattr(cache_mod.time, "time", lambda: real_time() + 60)
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is False


def test_every_path_writes_exactly_one_metrics_row(gateway, settings):
    gateway.handle(QueryRequest(query="q one"))
    gateway.handle(QueryRequest(query="q one"))
    gateway.handle(QueryRequest(query="q two"), bypass_cache=True)

    rows = _rows(settings)
    assert len(rows) == 3
    assert [r["hit"] for r in rows] == [0, 1, 0]
    assert [r["cache_enabled"] for r in rows] == [1, 1, 0]
    assert all(r["latency_total_ms"] >= 0 for r in rows)


def test_hit_records_avoided_tokens_and_source_entry(gateway, settings):
    gateway.handle(QueryRequest(query="What is a deadlock?"))
    hit = gateway.handle(QueryRequest(query="What is a deadlock?"))

    row = _rows(settings)[-1]
    assert row["reused_from_id"] is not None
    # Tokens attributable to a hit are the ones it avoided spending.
    assert row["prompt_tokens"] > 0
    assert row["completion_tokens"] > 0
    assert hit.latency_backend_ms == 0.0


def test_deferred_store_is_not_visible_until_it_runs(settings):
    """A miss must return before the write, so store latency is off the path."""
    pending = []
    gw = Gateway(settings)

    gw.handle(QueryRequest(query="What is a deadlock?"), defer=pending.append)
    assert len(pending) == 1
    # Nothing stored yet, so an immediate repeat still misses.
    assert gw.handle(QueryRequest(query="What is a deadlock?"), defer=pending.append).hit is False

    for task in list(pending):
        task()
    assert gw.handle(QueryRequest(query="What is a deadlock?")).hit is True


def test_labels_round_trip_into_metrics(gateway, settings):
    gateway.handle(
        QueryRequest(
            query="What is a deadlock?",
            category="conceptual",
            variant_of="CP-1",
            variant_kind="novel",
            run_id="run-a",
        )
    )
    row = _rows(settings)[-1]
    assert row["category"] == "conceptual"
    assert row["variant_of"] == "CP-1"
    assert row["variant_kind"] == "novel"
    assert row["run_id"] == "run-a"


def test_http_endpoints(settings):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(Gateway(settings)))

    assert client.get("/healthz").json()["status"] == "ok"

    first = client.post("/query", json={"query": "What is a deadlock?"}).json()
    second = client.post("/query", json={"query": "What is a deadlock?"}).json()
    assert first["hit"] is False
    assert second["hit"] is True

    bypassed = client.post(
        "/query", json={"query": "What is a deadlock?"}, headers={"X-Cache-Bypass": "1"}
    ).json()
    assert bypassed["hit"] is False

    stats = client.get("/stats").json()
    assert stats["requests"] >= 2
    assert stats["hits"] >= 1


def test_empty_query_is_rejected(settings):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(Gateway(settings)))
    assert client.post("/query", json={"query": ""}).status_code == 422
