"""Tests for RSI(14) - Phase 2 Feature Engine."""
import pytest

from src.features.rsi import compute_rsi


def _make_records(closes, start_minute=0):
    records = []
    for i, close in enumerate(closes):
        minute = start_minute + i * 5
        hour, minute = divmod(minute, 60)
        records.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "event_time": f"2026-01-01T{hour:02d}:{minute:02d}:00Z",
                "close": close,
            }
        )
    return records


def test_insufficient_data_returns_all_none():
    records = _make_records([100, 101, 102])
    result = compute_rsi(records, period=14)
    assert len(result) == 3
    assert result == [None, None, None]


def test_exactly_enough_data_produces_one_value_at_last_index():
    records = _make_records([100 + i for i in range(15)])
    result = compute_rsi(records, period=14)
    assert result[:14] == [None] * 14
    assert result[14] is not None
    assert 0.0 <= result[14] <= 100.0


def test_monotonically_increasing_prices_approach_100():
    records = _make_records([100 + i for i in range(30)])
    result = compute_rsi(records, period=14)
    assert result[-1] == 100.0


def test_monotonically_decreasing_prices_approach_0():
    records = _make_records([100 - i for i in range(30)])
    result = compute_rsi(records, period=14)
    assert result[-1] == 0.0


def test_flat_prices_give_neutral_50():
    records = _make_records([100.0] * 20)
    result = compute_rsi(records, period=14)
    assert result[14] == 50.0
    assert result[-1] == 50.0


def test_missing_close_field_raises_value_error():
    records = _make_records([100, 101, 102])
    del records[1]["close"]
    with pytest.raises(ValueError, match="close"):
        compute_rsi(records, period=14)


def test_non_numeric_close_raises_value_error():
    records = _make_records([100, 101, 102])
    records[1]["close"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        compute_rsi(records, period=14)


def test_out_of_order_event_time_raises_value_error():
    records = _make_records([100, 101, 102, 103])
    records[2]["event_time"], records[1]["event_time"] = (
        records[1]["event_time"],
        records[2]["event_time"],
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        compute_rsi(records, period=14)


def test_invalid_period_raises_value_error():
    records = _make_records([100, 101, 102])
    for bad_period in (0, -1, 1.5, "14"):
        with pytest.raises(ValueError):
            compute_rsi(records, period=bad_period)


def test_empty_input_returns_empty_list():
    assert compute_rsi([], period=14) == []


def test_no_look_ahead_truncated_series_matches_full_series():
    """RSI at index i must not depend on any record after index i."""
    records = _make_records([100 + (i % 7) - (i % 3) for i in range(40)])
    full = compute_rsi(records, period=14)
    for cutoff in (14, 20, 30, 39):
        truncated = compute_rsi(records[: cutoff + 1], period=14)
        assert truncated[cutoff] == full[cutoff]
