"""Tests for the JSONL historical data source."""

import json

import pytest

from src.data.historical.jsonl_source import JsonlHistoricalDataSource


def test_jsonl_source_filters_by_symbol_timeframe_and_time(tmp_path):
    path = tmp_path / "market.jsonl"

    records = [
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "event_time": "2026-01-01T00:05:00+00:00",
            "close": 100000,
        },
        {
            "symbol": "ETHUSDT",
            "timeframe": "5m",
            "event_time": "2026-01-01T00:10:00+00:00",
            "close": 3000,
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "event_time": "2026-01-01T00:15:00+00:00",
            "close": 101000,
        },
        {
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "event_time": "2026-01-01T00:20:00+00:00",
            "close": 100500,
        },
    ]

    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    source = JsonlHistoricalDataSource(path)

    result = source.fetch(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-01-01T00:10:00+00:00",
    )

    assert len(result) == 1
    assert result[0]["symbol"] == "BTCUSDT"
    assert result[0]["timeframe"] == "5m"
    assert result[0]["close"] == 100000


def test_jsonl_source_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"symbol": "BTCUSDT"}\n{invalid}\n', encoding="utf-8")

    source = JsonlHistoricalDataSource(path)

    with pytest.raises(ValueError, match="Invalid JSON on line 2"):
        source.fetch(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T01:00:00+00:00",
        )
def test_jsonl_source_raises_when_file_is_missing(tmp_path):
    path = tmp_path / "missing.jsonl"

    source = JsonlHistoricalDataSource(path)

    with pytest.raises(FileNotFoundError):
        source.fetch(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T01:00:00+00:00",
        )
