"""Research journal and monitoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


@dataclass(frozen=True)
class JournalEntry:
    """Immutable record of one paper-trading decision/outcome."""

    timestamp: datetime
    symbol: str
    signal: int
    probability: float
    expected_value: float
    risk_approved: bool
    approved: bool
    entry_price: float | None
    exit_price: float | None
    pnl: float | None


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Current monitoring statistics."""

    total_entries: int
    approved_entries: int
    completed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    latest_timestamp: datetime | None


class Journal:
    """In-memory journal for paper-trading research.

    The journal records decisions and completed paper-trade outcomes.
    It does not place orders or connect to an exchange.
    """

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        """Return all journal entries."""
        return tuple(self._entries)

    def record(
        self,
        *,
        symbol: str,
        signal: int,
        probability: float,
        expected_value: float,
        risk_approved: bool,
        approved: bool,
        entry_price: float | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
        timestamp: datetime | None = None,
    ) -> JournalEntry:
        """Record one validated research observation."""
        _validate_symbol(symbol)
        _validate_signal(signal)
        _validate_probability(probability)
        _validate_finite(
            expected_value,
            "expected_value",
        )

        if not isinstance(risk_approved, bool):
            raise ValueError(
                "risk_approved must be boolean"
            )

        if not isinstance(approved, bool):
            raise ValueError(
                "approved must be boolean"
            )

        if approved and not risk_approved:
            raise ValueError(
                "approved entry requires risk approval"
            )

        _validate_optional_positive(
            entry_price,
            "entry_price",
        )
        _validate_optional_positive(
            exit_price,
            "exit_price",
        )
        _validate_optional_finite(
            pnl,
            "pnl",
        )

        if (
            exit_price is not None
            and entry_price is None
        ):
            raise ValueError(
                "exit_price requires entry_price"
            )

        if pnl is not None and exit_price is None:
            raise ValueError(
                "pnl requires exit_price"
            )

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        else:
            _validate_timestamp(timestamp)

        entry = JournalEntry(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal,
            probability=float(probability),
            expected_value=float(expected_value),
            risk_approved=risk_approved,
            approved=approved,
            entry_price=(
                float(entry_price)
                if entry_price is not None
                else None
            ),
            exit_price=(
                float(exit_price)
                if exit_price is not None
                else None
            ),
            pnl=(
                float(pnl)
                if pnl is not None
                else None
            ),
        )

        self._entries.append(entry)
        return entry

    def snapshot(self) -> MonitoringSnapshot:
        """Calculate monitoring statistics."""
        completed = [
            entry
            for entry in self._entries
            if entry.pnl is not None
        ]

        winning = [
            entry
            for entry in completed
            if entry.pnl > 0.0
        ]

        losing = [
            entry
            for entry in completed
            if entry.pnl <= 0.0
        ]

        total_pnl = sum(
            entry.pnl
            for entry in completed
            if entry.pnl is not None
        )

        average_pnl = (
            total_pnl / len(completed)
            if completed
            else 0.0
        )

        return MonitoringSnapshot(
            total_entries=len(self._entries),
            approved_entries=sum(
                1
                for entry in self._entries
                if entry.approved
            ),
            completed_trades=len(completed),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=(
                len(winning) / len(completed)
                if completed
                else 0.0
            ),
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            latest_timestamp=(
                self._entries[-1].timestamp
                if self._entries
                else None
            ),
        )

    def clear(self) -> None:
        """Clear all journal entries."""
        self._entries.clear()


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str):
        raise ValueError("symbol must be a string")

    if not symbol.strip():
        raise ValueError("symbol must not be empty")


def _validate_signal(signal: int) -> None:
    if isinstance(signal, bool) or not isinstance(
        signal,
        int,
    ):
        raise ValueError(
            "signal must be an integer"
        )

    if signal not in (0, 1):
        raise ValueError(
            "signal must be 0 or 1"
        )


def _validate_probability(value: float) -> None:
    _validate_finite(value, "probability")

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            "probability must be between 0 and 1"
        )


def _validate_finite(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{name} must be finite"
        )


def _validate_optional_finite(
    value: float | None,
    name: str,
) -> None:
    if value is not None:
        _validate_finite(value, name)


def _validate_optional_positive(
    value: float | None,
    name: str,
) -> None:
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    numeric = float(value)

    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive"
        )


def _validate_timestamp(timestamp: datetime) -> None:
    if not isinstance(timestamp, datetime):
        raise ValueError(
            "timestamp must be a datetime"
        )

    if timestamp.tzinfo is None:
        raise ValueError(
            "timestamp must be timezone-aware"
        )
