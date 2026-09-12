"""
Proves the real data flow required to close the Phase 1 historical-data
blocker: JSONL fixture -> JSONLHistoricalSource -> DefaultHistoricalDataLoader
-> records.
"""
from pathlib import Path

from src.data.historical.loader import DefaultHistoricalDataLoader
from src.data.historical.source import JSONLHistoricalSource

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_historical.jsonl"


def _make_loader():
    return DefaultHistoricalDataLoader(JSONLHistoricalSource(FIXTURE_PATH))


def test_single_symbol_single_timeframe_reads_fixture():
    records = _make_loader().load(("BTCUSDT",), ("5m",), "2026-01-01T00:00:00Z", "2026-01-01T23:59:59Z")
    assert len(records) == 3
    assert all(r["symbol"] == "BTCUSDT" and r["timeframe"] == "5m" for r in records)


def test_multiple_symbols_and_timeframes_merge_deterministically():
    records = _make_loader().load(("BTCUSDT", "ETHUSDT"), ("5m", "1h"), "2026-01-01T00:00:00Z", "2026-01-01T23:59:59Z")
    assert len(records) == 5
    event_times = [r["event_time"] for r in records]
    assert event_times == sorted(event_times)


def test_time_range_filters_out_of_range_record():
    records = _make_loader().load(("BTCUSDT",), ("5m",), "2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z")
    assert len(records) == 2


def test_empty_result_when_nothing_matches():
    records = _make_loader().load(("SOLUSDT",), ("5m",), "2026-01-01T00:00:00Z", "2026-01-01T23:59:59Z")
    assert records == []
