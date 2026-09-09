"""Tests for market data collector foundation."""

import pytest

from src.data.collector.base import MarketDataCollector
from src.data.collector.config import CollectorConfig
from src.data.collector.validation import validate_config


def test_collector_interface_is_abstract():
    with pytest.raises(TypeError):
        MarketDataCollector()


def test_valid_config():
    config = CollectorConfig(
        symbols=("BTCUSDT",),
        timeframes=("5m",),
    )

    validate_config(config)


def test_empty_symbols_rejected():
    config = CollectorConfig(
        symbols=(),
        timeframes=("5m",),
    )

    with pytest.raises(ValueError):
        validate_config(config)


def test_empty_timeframes_rejected():
    config = CollectorConfig(
        symbols=("BTCUSDT",),
        timeframes=(),
    )

    with pytest.raises(ValueError):
        validate_config(config)
