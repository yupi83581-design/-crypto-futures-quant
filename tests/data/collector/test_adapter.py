"""Tests for exchange adapter interface."""

import pytest

from src.data.collector.adapter import ExchangeAdapter


def test_exchange_adapter_is_abstract():
    with pytest.raises(TypeError):
        ExchangeAdapter()
