"""Order-flow research features."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OrderFlowResult:
    """Deterministic order-flow statistics."""

    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance: float
    buy_pressure: float
    sell_pressure: float


def calculate_order_flow(
    *,
    buy_volume: float,
    sell_volume: float,
) -> OrderFlowResult:
    """Calculate normalized buy/sell pressure and volume imbalance.

    Imbalance is defined as:

        (buy_volume - sell_volume) / total_volume

    Buy and sell pressure are normalized shares of total volume.

    This is a descriptive research feature. It does not place orders,
    forecast prices, or make trading decisions.
    """
    _validate_volume(buy_volume, "buy_volume")
    _validate_volume(sell_volume, "sell_volume")

    total_volume = float(buy_volume) + float(sell_volume)

    if total_volume <= 0.0:
        raise ValueError(
            "total volume must be greater than zero"
        )

    imbalance = (
        float(buy_volume) - float(sell_volume)
    ) / total_volume

    buy_pressure = float(buy_volume) / total_volume
    sell_pressure = float(sell_volume) / total_volume

    return OrderFlowResult(
        buy_volume=float(buy_volume),
        sell_volume=float(sell_volume),
        total_volume=total_volume,
        imbalance=imbalance,
        buy_pressure=buy_pressure,
        sell_pressure=sell_pressure,
    )


def _validate_volume(
    value: float,
    name: str,
) -> None:
    """Validate one non-negative finite volume value."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be finite"
        )

    if numeric < 0.0:
        raise ValueError(
            f"{name} must be non-negative"
        )
