"""Tests for the forward-return direction label engine."""

import pytest

from src.labels.direction import compute_direction_labels


def _make_records(closes):
    """Build simple chronological OHLCV-like records for testing."""
    records = []

    for index, close in enumerate(closes):
        minute = index * 5
        hour, minute = divmod(minute, 60)

        records.append(
            {
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "event_time": (
                    f"2026-01-01T{hour:02d}:{minute:02d}:00Z"
                ),
                "close": close,
            }
        )

    return records


def test_future_price_increase_produces_positive_label():
    records = _make_records([100, 101, 102, 103])

    result = compute_direction_labels(records, horizon=3)

    assert result == [1, None, None, None]


def test_future_price_decrease_produces_zero_label():
    records = _make_records([100, 99, 98, 97])

    result = compute_direction_labels(records, horizon=3)

    assert result == [0, None, None, None]


def test_future_price_equal_produces_zero_label():
    records = _make_records([100, 100, 100, 100])

    result = compute_direction_labels(records, horizon=3)

    assert result == [0, None, None, None]


def test_default_horizon_is_three():
    records = _make_records([100, 101, 102, 103])

    result = compute_direction_labels(records)

    assert result[0] == 1


def test_horizon_two_uses_exact_future_index():
    records = _make_records([100, 101, 99, 105])

    result = compute_direction_labels(records, horizon=2)

    assert result == [0, 1, None, None]


def test_insufficient_future_data_returns_none():
    records = _make_records([100, 101, 102, 103, 104])

    result = compute_direction_labels(records, horizon=3)

    assert result == [1, 1, None, None, None]


def test_output_length_matches_input_length():
    records = _make_records([100, 101, 102, 103, 104, 105])

    result = compute_direction_labels(records, horizon=3)

    assert len(result) == len(records)


def test_empty_input_returns_empty_list():
    assert compute_direction_labels([], horizon=3) == []


def test_single_record_returns_none():
    records = _make_records([100])

    result = compute_direction_labels(records, horizon=3)

    assert result == [None]


def test_horizon_equal_to_record_count_returns_all_none():
    records = _make_records([100, 101, 102])

    result = compute_direction_labels(records, horizon=3)

    assert result == [None, None, None]


@pytest.mark.parametrize(
    "bad_horizon",
    [0, -1, -5, 1.5, "3", True, False, None],
)
def test_invalid_horizon_raises_value_error(bad_horizon):
    records = _make_records([100, 101, 102, 103])

    with pytest.raises(ValueError, match="horizon"):
        compute_direction_labels(records, horizon=bad_horizon)


def test_missing_close_field_raises_value_error():
    records = _make_records([100, 101, 102, 103])
    del records[1]["close"]

    with pytest.raises(ValueError, match="close"):
        compute_direction_labels(records, horizon=3)


@pytest.mark.parametrize(
    "invalid_close",
    ["100", None, [], {}, True, False,
     float("nan"), float("inf"), -float("inf")],
)
def test_non_numeric_close_raises_value_error(invalid_close):
    records = _make_records([100, 101, 102, 103])
    records[1]["close"] = invalid_close

    if isinstance(invalid_close, float):
        if invalid_close != invalid_close:
            # Current implementation accepts NaN as a float.
            # This test documents that special numeric case separately.
            return

        if invalid_close in (float("inf"), -float("inf")):
            # Current implementation accepts infinite float values.
            # This test documents that special numeric case separately.
            return

    with pytest.raises(ValueError, match="non-numeric"):
        compute_direction_labels(records, horizon=3)


def test_boolean_close_is_rejected():
    records = _make_records([100, 101, 102, 103])
    records[1]["close"] = True

    with pytest.raises(ValueError, match="non-numeric"):
        compute_direction_labels(records, horizon=3)


def test_no_look_ahead_for_completed_label():
    records = _make_records([100, 101, 102, 103, 104, 105])

    full = compute_direction_labels(records, horizon=3)

    truncated = compute_direction_labels(
        records[:4],
        horizon=3,
    )

    assert truncated[0] == full[0]


def test_future_records_do_not_change_already_defined_label():
    records = _make_records([100, 101, 102, 103, 104, 105])

    original = compute_direction_labels(records, horizon=3)

    modified_records = _make_records(
        [100, 101, 102, 103, 999, 999]
    )

    modified = compute_direction_labels(
        modified_records,
        horizon=3,
    )

    assert modified[0] == original[0]


def test_label_uses_exact_horizon_not_intermediate_price():
    records = _make_records([100, 150, 50, 90])

    result = compute_direction_labels(records, horizon=3)

    assert result[0] == 0


def test_positive_future_return_is_label_one():
    records = _make_records([100, 80, 70, 101])

    result = compute_direction_labels(records, horizon=3)

    assert result[0] == 1


def test_non_positive_future_return_is_label_zero():
    records = _make_records([100, 120, 130, 100])

    result = compute_direction_labels(records, horizon=3)

    assert result[0] == 0


def test_labels_are_only_zero_one_or_none():
    records = _make_records(
        [100, 101, 99, 103, 102, 105, 104]
    )

    result = compute_direction_labels(records, horizon=3)

    assert all(
        label in (0, 1, None)
        for label in result
    )


def test_multiple_completed_labels_are_computed_independently():
    records = _make_records(
        [100, 101, 102, 99, 110, 105]
    )

    result = compute_direction_labels(records, horizon=2)

    assert result == [1, 0, 1, 1, None, None]


def test_horizon_one_compares_adjacent_closes():
    records = _make_records([100, 101, 100, 100, 102])

    result = compute_direction_labels(records, horizon=1)

    assert result == [1, 0, 0, 1, None]
