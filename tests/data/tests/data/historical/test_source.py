"""Tests for the historical data source interface."""

import pytest

from src.data.historical.source import HistoricalDataSource


def test_historical_data_source_is_abstract():
    with pytest.raises(TypeError):
        HistoricalDataSource()
