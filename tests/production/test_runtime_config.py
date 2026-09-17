"""Tests for the unchanged default runtime network binding."""

from src.production.runtime import RuntimeConfig


def test_runtime_defaults_bind_public_http_port() -> None:
    config = RuntimeConfig()

    assert config.host == "0.0.0.0"
    assert config.port == 8080
