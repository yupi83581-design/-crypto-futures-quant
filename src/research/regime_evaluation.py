"""Out-of-sample-style evaluation of market regimes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from src.research.regime import (
    RegimeConfig,
    detect_regime,
)


@dataclass(frozen=True)
class RegimePerformance:
    """Forward-return statistics for one detected regime."""

    observations: int
    mean_forward_return: float
    positive_rate: float


@dataclass(frozen=True)
class RegimeEvaluationResult:
    """Summary of forward returns conditioned on detected regimes."""

    total_observations: int
    overall_mean_forward_return: float
    overall_positive_rate: float
    trend_up: RegimePerformance
    trend_down: RegimePerformance
    high_volatility: RegimePerformance
    range: RegimePerformance


def evaluate_regime_impact(
    prices: list[float],
    *,
    config: RegimeConfig | None = None,
) -> RegimeEvaluationResult:
    """Evaluate whether regimes contain information about next-bar returns.

    For every evaluation point:

    1. Only historical prices available up to that point are used.
    2. The historical window is classified into a regime.
    3. The immediately following return is recorded as the outcome.

    Therefore the forward return is never used to determine the regime.

    This function is descriptive research, not a trading strategy.
    A positive conditional return does not by itself establish a
    statistically significant or economically exploitable edge.
    """
    if config is None:
        config = RegimeConfig()

    _validate_prices(prices)

    minimum = config.minimum_observations

    if len(prices) <= minimum:
        raise ValueError(
            "insufficient prices for forward-return evaluation"
        )

    observations: dict[str, list[float]] = {
        "TREND_UP": [],
        "TREND_DOWN": [],
        "HIGH_VOLATILITY": [],
        "RANGE": [],
    }

    forward_returns: list[float] = []

    for index in range(
        minimum - 1,
        len(prices) - 1,
    ):
        historical_prices = prices[
            index - minimum + 1 : index + 1
        ]

        regime_result = detect_regime(
            historical_prices,
            config=config,
        )

        forward_return = (
            prices[index + 1] / prices[index]
        ) - 1.0

        observations[regime_result.regime].append(
            forward_return
        )
        forward_returns.append(forward_return)

    return RegimeEvaluationResult(
        total_observations=len(forward_returns),
        overall_mean_forward_return=float(
            mean(forward_returns)
        ),
        overall_positive_rate=_positive_rate(
            forward_returns
        ),
        trend_up=_performance(
            observations["TREND_UP"]
        ),
        trend_down=_performance(
            observations["TREND_DOWN"]
        ),
        high_volatility=_performance(
            observations["HIGH_VOLATILITY"]
        ),
        range=_performance(
            observations["RANGE"]
        ),
    )


def _performance(
    returns: list[float],
) -> RegimePerformance:
    if not returns:
        return RegimePerformance(
            observations=0,
            mean_forward_return=0.0,
            positive_rate=0.0,
        )

    return RegimePerformance(
        observations=len(returns),
        mean_forward_return=float(mean(returns)),
        positive_rate=_positive_rate(returns),
    )


def _positive_rate(
    returns: list[float],
) -> float:
    if not returns:
        return 0.0

    positive = sum(
        1
        for value in returns
        if value > 0.0
    )

    return positive / len(returns)


def _validate_prices(
    prices: list[float],
) -> None:
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

    if len(values) < 2:
        raise ValueError(
            "prices must contain at least 2 observations"
        )

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
