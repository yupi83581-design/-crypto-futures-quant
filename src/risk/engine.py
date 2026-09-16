"""Deterministic risk controls for quantitative research decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for deterministic position-risk controls."""

    max_risk_fraction: float = 0.01
    max_position_fraction: float = 1.0
    max_drawdown_fraction: float = 0.20


@dataclass(frozen=True)
class RiskResult:
    """Risk assessment for one research decision."""

    approved: bool
    reason: str
    risk_fraction: float
    position_fraction: float
    risk_amount: float
    position_notional: float


def assess_risk(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    net_expected_value: float,
    current_drawdown_fraction: float = 0.0,
    config: RiskConfig | None = None,
) -> RiskResult:
    """Assess deterministic risk controls, including drawdown enforcement.

    ``current_drawdown_fraction`` is measured from the relevant equity peak
    and must be supplied by the caller from the current portfolio/equity
    state. A decision is rejected once it reaches the configured drawdown
    limit. This function is research/paper-only and never places orders.
    """
    if config is None:
        config = RiskConfig()

    _validate_config(config)
    _validate_positive(equity, "equity")
    _validate_positive(entry_price, "entry_price")
    _validate_positive(stop_price, "stop_price")
    _validate_finite(net_expected_value, "net_expected_value")
    _validate_fraction(current_drawdown_fraction, "current_drawdown_fraction")

    if current_drawdown_fraction >= config.max_drawdown_fraction:
        return RiskResult(False, "max drawdown limit reached", 0.0, 0.0, 0.0, 0.0)

    if stop_price >= entry_price:
        return RiskResult(False, "stop_price must be below entry_price", 0.0, 0.0, 0.0, 0.0)

    if net_expected_value <= 0.0:
        return RiskResult(False, "net_expected_value must be positive", 0.0, 0.0, 0.0, 0.0)

    stop_distance_fraction = (entry_price - stop_price) / entry_price
    if stop_distance_fraction <= 0.0:
        return RiskResult(False, "stop distance must be positive", 0.0, 0.0, 0.0, 0.0)

    risk_fraction = config.max_risk_fraction
    risk_amount = equity * risk_fraction
    position_notional = min(
        risk_amount / stop_distance_fraction,
        equity * config.max_position_fraction,
    )
    return RiskResult(
        True,
        "risk limits passed",
        risk_fraction,
        position_notional / equity,
        risk_amount,
        position_notional,
    )


def _validate_config(config: RiskConfig) -> None:
    _validate_fraction(config.max_risk_fraction, "max_risk_fraction")
    _validate_fraction(config.max_position_fraction, "max_position_fraction")
    _validate_fraction(config.max_drawdown_fraction, "max_drawdown_fraction")
    if config.max_risk_fraction <= 0.0:
        raise ValueError("max_risk_fraction must be positive")
    if config.max_position_fraction <= 0.0:
        raise ValueError("max_position_fraction must be positive")
    if config.max_drawdown_fraction <= 0.0:
        raise ValueError("max_drawdown_fraction must be positive")


def _validate_positive(value: float, name: str) -> None:
    _validate_finite(value, name)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_fraction(value: float, name: str) -> None:
    _validate_finite(value, name)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
