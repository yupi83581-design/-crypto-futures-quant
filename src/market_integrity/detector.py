"""Market-integrity analytics for defensive quantitative trading.

This module detects suspicious market microstructure patterns from observed
trades/order-book snapshots. It never creates, cancels, or manipulates orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass(frozen=True)
class MarketIntegrityResult:
    status: str
    spoofing_risk: float
    liquidity_withdrawal_risk: float
    volume_anomaly_risk: float
    reasons: tuple[str, ...]


def assess_market_integrity(
    *,
    bid_depth: Sequence[OrderBookLevel],
    ask_depth: Sequence[OrderBookLevel],
    prior_bid_depth: Sequence[OrderBookLevel] | None = None,
    prior_ask_depth: Sequence[OrderBookLevel] | None = None,
    recent_volume: float | None = None,
    baseline_volume: float | None = None,
    withdrawal_threshold: float = 0.70,
    anomaly_multiplier: float = 3.0,
) -> MarketIntegrityResult:
    """Return a defensive integrity assessment from public market data.

    High-risk patterns are treated as a reason to abstain from trading.
    This is a screening heuristic, not proof that manipulation occurred.
    """
    _validate_levels(bid_depth)
    _validate_levels(ask_depth)
    _validate_optional_levels(prior_bid_depth)
    _validate_optional_levels(prior_ask_depth)
    _validate_fraction(withdrawal_threshold, "withdrawal_threshold")
    if anomaly_multiplier <= 1.0:
        raise ValueError("anomaly_multiplier must exceed 1")

    spoofing = _layered_depth_risk(bid_depth) + _layered_depth_risk(ask_depth)
    spoofing = min(1.0, spoofing / 2.0)

    withdrawal = 0.0
    if prior_bid_depth is not None and prior_ask_depth is not None:
        prior_total = _depth_total(prior_bid_depth) + _depth_total(prior_ask_depth)
        current_total = _depth_total(bid_depth) + _depth_total(ask_depth)
        if prior_total > 0:
            drop = max(0.0, 1.0 - current_total / prior_total)
            withdrawal = min(1.0, drop / withdrawal_threshold)

    volume_anomaly = 0.0
    if recent_volume is not None or baseline_volume is not None:
        if recent_volume is None or baseline_volume is None:
            raise ValueError("recent_volume and baseline_volume must be supplied together")
        if recent_volume < 0 or baseline_volume <= 0:
            raise ValueError("volume values must be non-negative with positive baseline")
        volume_anomaly = min(
            1.0,
            max(0.0, recent_volume / baseline_volume - 1.0)
            / (anomaly_multiplier - 1.0),
        )

    reasons: list[str] = []
    if spoofing >= 0.75:
        reasons.append("layered near-touch liquidity is unusually concentrated")
    if withdrawal >= 0.75:
        reasons.append("observed displayed liquidity withdrew unusually quickly")
    if volume_anomaly >= 0.75:
        reasons.append("recent volume is unusually high versus baseline")

    status = "SUSPICIOUS" if reasons else "NORMAL"
    return MarketIntegrityResult(
        status=status,
        spoofing_risk=spoofing,
        liquidity_withdrawal_risk=withdrawal,
        volume_anomaly_risk=volume_anomaly,
        reasons=tuple(reasons),
    )


def _layered_depth_risk(levels: Sequence[OrderBookLevel]) -> float:
    if len(levels) < 3:
        return 0.0
    quantities = sorted((float(level.quantity) for level in levels), reverse=True)
    top = quantities[0]
    remainder = sum(quantities[1:])
    if remainder <= 0:
        return 1.0
    return min(1.0, top / remainder)


def _depth_total(levels: Sequence[OrderBookLevel]) -> float:
    return sum(level.quantity for level in levels)


def _validate_levels(levels: Sequence[OrderBookLevel]) -> None:
    if isinstance(levels, (str, bytes)) or not levels:
        raise ValueError("order-book levels must be non-empty")
    for level in levels:
        if not isinstance(level, OrderBookLevel):
            raise TypeError("levels must contain OrderBookLevel values")
        if level.price <= 0 or level.quantity <= 0:
            raise ValueError("price and quantity must be positive")


def _validate_optional_levels(levels: Sequence[OrderBookLevel] | None) -> None:
    if levels is not None:
        _validate_levels(levels)


def _validate_fraction(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
