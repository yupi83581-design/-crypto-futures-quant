from datetime import datetime, timezone
import math

import pytest

from src.monitoring.journal import (
    Journal,
    MonitoringSnapshot,
)


TIMESTAMP = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def test_empty_journal_has_zero_snapshot():
    journal = Journal()

    snapshot = journal.snapshot()

    assert snapshot == MonitoringSnapshot(
        total_entries=0,
        approved_entries=0,
        completed_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        total_pnl=0.0,
        average_pnl=0.0,
        latest_timestamp=None,
    )


def test_record_creates_entry():
    journal = Journal()

    entry = journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.75,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
        timestamp=TIMESTAMP,
    )

    assert entry.symbol == "BTCUSDT"
    assert entry.signal == 1
    assert entry.probability == pytest.approx(0.75)
    assert entry.expected_value == pytest.approx(0.02)
    assert entry.risk_approved is True
    assert entry.approved is True
    assert entry.entry_price is None
    assert entry.exit_price is None
    assert entry.pnl is None
    assert entry.timestamp == TIMESTAMP


def test_entries_are_returned_as_immutable_tuple():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.01,
        risk_approved=True,
        approved=True,
        timestamp=TIMESTAMP,
    )

    entries = journal.entries

    assert isinstance(entries, tuple)
    assert len(entries) == 1


def test_approved_entry_requires_risk_approval():
    journal = Journal()

    with pytest.raises(
        ValueError,
        match="risk approval",
    ):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.01,
            risk_approved=False,
            approved=True,
            timestamp=TIMESTAMP,
        )


def test_completed_trade_records_prices_and_pnl():
    journal = Journal()

    entry = journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.80,
        expected_value=0.03,
        risk_approved=True,
        approved=True,
        entry_price=100_000.0,
        exit_price=101_000.0,
        pnl=1_000.0,
        timestamp=TIMESTAMP,
    )

    assert entry.entry_price == pytest.approx(100_000.0)
    assert entry.exit_price == pytest.approx(101_000.0)
    assert entry.pnl == pytest.approx(1_000.0)


def test_snapshot_counts_approved_entries():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.80,
        expected_value=0.03,
        risk_approved=True,
        approved=True,
        timestamp=TIMESTAMP,
    )

    journal.record(
        symbol="ETHUSDT",
        signal=0,
        probability=0.40,
        expected_value=-0.01,
        risk_approved=True,
        approved=False,
        timestamp=TIMESTAMP,
    )

    snapshot = journal.snapshot()

    assert snapshot.total_entries == 2
    assert snapshot.approved_entries == 1


def test_snapshot_counts_wins_and_losses():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.80,
        expected_value=0.03,
        risk_approved=True,
        approved=True,
        entry_price=100.0,
        exit_price=110.0,
        pnl=10.0,
        timestamp=TIMESTAMP,
    )

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
        entry_price=100.0,
        exit_price=95.0,
        pnl=-5.0,
        timestamp=TIMESTAMP,
    )

    snapshot = journal.snapshot()

    assert snapshot.completed_trades == 2
    assert snapshot.winning_trades == 1
    assert snapshot.losing_trades == 1


def test_zero_pnl_is_counted_as_losing_trade():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.50,
        expected_value=0.01,
        risk_approved=True,
        approved=True,
        entry_price=100.0,
        exit_price=100.0,
        pnl=0.0,
        timestamp=TIMESTAMP,
    )

    snapshot = journal.snapshot()

    assert snapshot.completed_trades == 1
    assert snapshot.winning_trades == 0
    assert snapshot.losing_trades == 1
    assert snapshot.win_rate == pytest.approx(0.0)


def test_snapshot_calculates_win_rate():
    journal = Journal()

    for pnl in (10.0, 20.0, -5.0, -2.0):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            exit_price=101.0,
            pnl=pnl,
            timestamp=TIMESTAMP,
        )

    snapshot = journal.snapshot()

    assert snapshot.win_rate == pytest.approx(0.5)


def test_snapshot_calculates_total_pnl():
    journal = Journal()

    for pnl in (100.0, -40.0, 25.0):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            exit_price=101.0,
            pnl=pnl,
            timestamp=TIMESTAMP,
        )

    snapshot = journal.snapshot()

    assert snapshot.total_pnl == pytest.approx(85.0)


def test_snapshot_calculates_average_pnl():
    journal = Journal()

    for pnl in (100.0, -40.0, 20.0):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            exit_price=101.0,
            pnl=pnl,
            timestamp=TIMESTAMP,
        )

    snapshot = journal.snapshot()

    assert snapshot.average_pnl == pytest.approx(80.0 / 3.0)


def test_latest_timestamp_is_tracked():
    journal = Journal()

    later = datetime(
        2026,
        1,
        2,
        12,
        0,
        tzinfo=timezone.utc,
    )

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
        timestamp=TIMESTAMP,
    )

    journal.record(
        symbol="ETHUSDT",
        signal=1,
        probability=0.75,
        expected_value=0.03,
        risk_approved=True,
        approved=True,
        timestamp=later,
    )

    assert journal.snapshot().latest_timestamp == later


def test_clear_removes_all_entries():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
        timestamp=TIMESTAMP,
    )

    journal.clear()

    assert journal.entries == ()
    assert journal.snapshot().total_entries == 0


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_symbol_is_rejected(symbol):
    journal = Journal()

    with pytest.raises(ValueError, match="symbol"):
        journal.record(
            symbol=symbol,
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "signal",
    [
        -1,
        2,
        1.0,
        True,
        False,
    ],
)
def test_invalid_signal_is_rejected(signal):
    journal = Journal()

    with pytest.raises(ValueError, match="signal"):
        journal.record(
            symbol="BTCUSDT",
            signal=signal,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_probability_is_rejected(probability):
    journal = Journal()

    with pytest.raises(ValueError):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=probability,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "expected_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "invalid",
    ],
)
def test_invalid_expected_value_is_rejected(expected_value):
    journal = Journal()

    with pytest.raises(ValueError):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=expected_value,
            risk_approved=True,
            approved=True,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "risk_approved",
    [
        1,
        0,
        None,
        "true",
    ],
)
def test_invalid_risk_approval_is_rejected(risk_approved):
    journal = Journal()

    with pytest.raises(ValueError, match="risk_approved"):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=risk_approved,
            approved=False,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "approved",
    [
        1,
        0,
        None,
        "true",
    ],
)
def test_invalid_approved_value_is_rejected(approved):
    journal = Journal()

    with pytest.raises(ValueError, match="approved"):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=approved,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "entry_price",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        "invalid",
    ],
)
def test_invalid_entry_price_is_rejected(entry_price):
    journal = Journal()

    with pytest.raises(ValueError):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=entry_price,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "exit_price",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        "invalid",
    ],
)
def test_invalid_exit_price_is_rejected(exit_price):
    journal = Journal()

    with pytest.raises(ValueError):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            exit_price=exit_price,
            timestamp=TIMESTAMP,
        )


def test_exit_price_requires_entry_price():
    journal = Journal()

    with pytest.raises(
        ValueError,
        match="entry_price",
    ):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            exit_price=110.0,
            timestamp=TIMESTAMP,
        )


def test_pnl_requires_exit_price():
    journal = Journal()

    with pytest.raises(
        ValueError,
        match="exit_price",
    ):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            pnl=10.0,
            timestamp=TIMESTAMP,
        )


@pytest.mark.parametrize(
    "pnl",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "invalid",
    ],
)
def test_invalid_pnl_is_rejected(pnl):
    journal = Journal()

    with pytest.raises(ValueError):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            entry_price=100.0,
            exit_price=110.0,
            pnl=pnl,
            timestamp=TIMESTAMP,
        )


def test_timestamp_defaults_to_timezone_aware_utc():
    journal = Journal()

    entry = journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
    )

    assert entry.timestamp.tzinfo is not None
    assert entry.timestamp.utcoffset() is not None


def test_naive_timestamp_is_rejected():
    journal = Journal()

    naive_timestamp = datetime(
        2026,
        1,
        1,
        12,
        0,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            timestamp=naive_timestamp,
        )


def test_non_datetime_timestamp_is_rejected():
    journal = Journal()

    with pytest.raises(
        ValueError,
        match="datetime",
    ):
        journal.record(
            symbol="BTCUSDT",
            signal=1,
            probability=0.70,
            expected_value=0.02,
            risk_approved=True,
            approved=True,
            timestamp="2026-01-01",
        )


def test_snapshot_results_are_finite():
    journal = Journal()

    journal.record(
        symbol="BTCUSDT",
        signal=1,
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        approved=True,
        entry_price=100.0,
        exit_price=110.0,
        pnl=10.0,
        timestamp=TIMESTAMP,
    )

    snapshot = journal.snapshot()

    assert math.isfinite(snapshot.win_rate)
    assert math.isfinite(snapshot.total_pnl)
    assert math.isfinite(snapshot.average_pnl)
