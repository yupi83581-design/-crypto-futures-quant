"""Tests for the historical data loader interface."""

import pytest

from src.data.historical.loader import HistoricalDataLoader


def test_historical_data_loader_is_abstract():
    with pytest.raises(TypeError):
        HistoricalDataLoader()
