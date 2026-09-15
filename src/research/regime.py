"""Deterministic market-regime detection for quantitative research.

The regime detector is intentionally:
- deterministic;
- interpretable;
- independent of exchange execution;
- usable with real market-derived price/return data;
- suitable as a conditioning feature for later model research.

A regime classification is NOT treated as a proven trading edge.
Its usefulness must be validated out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds used to classify market conditions."""

    trend_threshold: float = 0.01
    volatility_threshold: float = 0.02
    minimum_observations: int = 20


@dataclass(frozen=True)
class RegimeResult:
    """Deterministic classification of the observed market regime."""

    regime: str
    trend_strength: float
    volatility: float
    return_mean: float
    observations: int


def detect_regime(
    prices: Sequence[float],
    config: RegimeConfig | None = None,
) -> RegimeResult:
    """Classify a price sequence into a simple market regime.

    Classification:

    - TREND_UP:
        return_mean > 0 and trend_strength >= threshold,
        with volatility below the high-volatility threshold.

    - TREND_DOWN:
        return_mean < 0 and trend_strength >= threshold,
        with volatility below the high-volatility threshold.

    - HIGH_VOLATILITY:
        volatility >= volatility threshold.

    - RANGE:
        all remaining valid observations.

    The detector uses percentage returns rather than raw prices so that
    instruments with different absolute price levels remain comparable.

    This function does not forecast prices and does not produce trade
    instructions.
    """
    if config is None:
        config = RegimeConfig()

    _validate_config(config)

    normalized_prices = _validate_prices(
        prices,
        minimum_observations=config.minimum_observations,
    )

    returns = _percentage_returns(normalized_prices)

    return_mean = mean(returns)
    volatility = pstdev(returns)

    trend_strength = abs(
        normalized_prices[-1] / normalized_prices[0] - 1.0
    )

    if volatility >= config.volatility_threshold:
        regime = "HIGH_VOLATILITY"
    elif (
        trend_strength >= config.trend_threshold
        and return_mean > 0.0
    ):
        regime = "TREND_UP"
    elif (
        trend_strength >= config.trend_threshold
        and return_mean < 0.0
    ):
        regime = "TREND_DOWN"
    else:
        regime = "RANGE"

    return RegimeResult(
        regime=regime,
        trend_strength=float(trend_strength),
        volatility=float(volatility),
        return_mean=float(return_mean),
        observations=len(normalized_prices),
    )


def _percentage_returns(
    prices: Sequence[float],
) -> list[float]:
    returns: list[float] = []

    for previous, current in zip(
        prices,
        prices[1:],
    ):
        returns.append(
            current / previous - 1.0
        )

    return returns


def _validate_config(config: RegimeConfig) -> None:
    _validate_non_negative_finite(
        config.trend_threshold,
        "trend_threshold",
    )
    _validate_non_negative_finite(
        config.volatility_threshold,
        "volatility_threshold",
    )

    if (
        isinstance(config.minimum_observations, bool)
        or not isinstance(config.minimum_observations, int)
    ):
        raise ValueError(
            "minimum_observations must be an integer"
        )

    if config.minimum_observations < 2:
        raise ValueError(
            "minimum_observations must be at least 2"
        )


def _validate_prices(
    prices: Sequence[float],
    *,
    minimum_observations: int,
) -> list[float]:
    if isinstance(prices, (str, bytes)):
        raise ValueError(
            "prices must be a numeric sequence"
        )

    try:
        values = list(prices)
    except TypeError as exc:
        raise ValueError(
            "prices must be a numeric sequence"
        ) from exc

    if len(values) < minimum_observations:
        raise ValueError(
            "insufficient price observations"
        )

    normalized: list[float] = []

    for value in values:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                "prices must contain only numeric values"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                "prices must contain only finite values"
            )

        if numeric <= 0.0:
            raise ValueError(
                "prices must be positive"
            )

        normalized.append(numeric)

    return normalized


def _validate_non_negative_finite(
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

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be finite"
        )

    if numeric < 0.0:
        raise ValueError(
            f"{name} must be non-negative"
        )
