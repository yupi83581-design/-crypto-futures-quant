from datetime import datetime, timedelta

import pytest

from src.data.integrity.gap_detector import detect_gaps


def test_complete_series_has_no_gaps():
    times = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 5),
        datetime(2026, 1, 1, 0, 10),
    ]

    gaps = detect_gaps(times, timedelta(minutes=5))

    assert gaps == []


def test_detects_one_missing_interval():
    times = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 5),
        datetime(2026, 1, 1, 0, 15),
    ]

    gaps = detect_gaps(times, timedelta(minutes=5))

    assert len(gaps) == 1
    assert gaps[0].previous_event == datetime(2026, 1, 1, 0, 5)
    assert gaps[0].next_event == datetime(2026, 1, 1, 0, 15)
    assert gaps[0].missing_duration == timedelta(minutes=5)


def test_detects_multiple_missing_intervals():
    times = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 15),
        datetime(2026, 1, 1, 0, 30),
    ]

    gaps = detect_gaps(times, timedelta(minutes=5))

    assert len(gaps) == 2
    assert all(
        gap.missing_duration == timedelta(minutes=10)
        for gap in gaps
    )


def test_unsorted_events_are_handled():
    times = [
        datetime(2026, 1, 1, 0, 10),
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 5),
    ]

    gaps = detect_gaps(times, timedelta(minutes=5))

    assert gaps == []


def test_empty_series_has_no_gaps():
    gaps = detect_gaps([], timedelta(minutes=5))

    assert gaps == []


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError, match="expected_interval must be positive"):
        detect_gaps(
            [datetime(2026, 1, 1, 0, 0)],
            timedelta(0),
        )
