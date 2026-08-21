import pytest

from gateway.app import Gateway
from gateway.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fully-faked settings: no Docker, GPU, or network."""
    s = Settings()
    s.embedder_impl = "bagofwords"
    s.cache_impl = "memory"
    s.backend_impl = "echo"
    s.metrics_db = str(tmp_path / "metrics.sqlite")
    s.reuse_threshold = 0.95
    s.cache_ttl_s = 3600
    return s


@pytest.fixture
def gateway(settings) -> Gateway:
    return Gateway(settings)
