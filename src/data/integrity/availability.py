"""Availability rules for historical market-data types."""

from dataclasses import dataclass
from enum import Enum


class AvailabilityMode(str, Enum):
    """How a market-data type is expected to appear over time."""

    INTERVAL = "interval"
    EVENT = "event"


@dataclass(frozen=True)
class AvailabilityRule:
    """Defines the expected availability semantics for a data type."""

    data_type: str
    mode: AvailabilityMode
    expected_interval_seconds: int | None = None

    def __post_init__(self) -> None:
        if not self.data_type:
            raise ValueError("data_type must not be empty")

        if (
            self.mode is AvailabilityMode.INTERVAL
            and self.expected_interval_seconds is None
        ):
            raise ValueError(
                "interval data requires expected_interval_seconds"
            )

        if (
            self.expected_interval_seconds is not None
            and self.expected_interval_seconds <= 0
        ):
            raise ValueError(
                "expected_interval_seconds must be positive"
            )


AVAILABILITY_RULES: dict[str, AvailabilityRule] = {
    "ohlcv": AvailabilityRule(
        data_type="ohlcv",
        mode=AvailabilityMode.INTERVAL,
        expected_interval_seconds=300,
    ),
    "trades": AvailabilityRule(
        data_type="trades",
        mode=AvailabilityMode.EVENT,
    ),
    "mark_price": AvailabilityRule(
        data_type="mark_price",
        mode=AvailabilityMode.INTERVAL,
        expected_interval_seconds=300,
    ),
    "index_price": AvailabilityRule(
        data_type="index_price",
        mode=AvailabilityMode.INTERVAL,
        expected_interval_seconds=300,
    ),
    "order_book_l2": AvailabilityRule(
        data_type="order_book_l2",
        mode=AvailabilityMode.EVENT,
    ),
    "funding": AvailabilityRule(
        data_type="funding",
        mode=AvailabilityMode.EVENT,
    ),
    "open_interest": AvailabilityRule(
        data_type="open_interest",
        mode=AvailabilityMode.EVENT,
    ),
    "liquidations": AvailabilityRule(
        data_type="liquidations",
        mode=AvailabilityMode.EVENT,
    ),
}
