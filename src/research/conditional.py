"""Conditional research combining market regime and order flow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from src.research.order_flow import calculate_order_flow
from src.research.regime import RegimeConfig, detect_regime


@dataclass(frozen=True)
class ConditionalPerformance:
    """Forward-return statistics for one regime/order-flow bucket."""

    observations: int
    mean_forward_return: float
    positive_rate: float


@dataclass(frozen=True)
class ConditionalEvaluationResult:
    """Forward returns conditioned on regime and order-flow direction."""

    total_observations: int
    overall_mean_forward_return: float
    overall_positive_rate: float
    trend_up_buy_dominant: ConditionalPerformance
    trend_up_balanced: ConditionalPerformance
    trend_up_sell_dominant: ConditionalPerformance
    trend_down_buy_dominant: ConditionalPerformance
    trend_down_balanced: ConditionalPerformance
    trend_down_sell_dominant: ConditionalPerformance
    high_volatility_buy_dominant: ConditionalPerformance
    high_volatility_balanced: ConditionalPerformance
    high_volatility_sell_dominant: ConditionalPerformance
    range_buy_dominant: ConditionalPerformance
    range_balanced: ConditionalPerformance
    range_sell_dominant: ConditionalPerformance


def evaluate_conditional_impact(
    *,
    buy_volumes: list[float],
    sell_volumes: list[float],
    prices: list[float],
    regime_config: RegimeConfig | None = None,
    imbalance_threshold: float = 0.20,
) -> ConditionalEvaluationResult:
    """Evaluate next-bar returns conditioned on regime and order flow.

    At each time t:

    1. Only prices available through t are used to determine the regime.
    2. Order-flow data at t determines the order-flow bucket.
    3. The return from price[t] to price[t + 1] is recorded as the outcome.

    Therefore the future price is never used to construct either input
    feature.

    This function is descriptive research only. It does not forecast,
    place orders, or establish a statistically significant trading edge.
    """
    if regime_config is None:
        regime_config = RegimeConfig()

    buy_values, sell_values, price_values = _validate_inputs(
        buy_volumes=buy_volumes,
        sell_volumes=sell_volumes,
        prices=prices,
        imbalance_threshold=imbalance_threshold,
    )

    _validate_regime_config(regime_config)

    minimum = regime_config.minimum_observations

    if len(price_values) <= minimum:
        raise ValueError(
            "insufficient prices for conditional evaluation"
        )

    observations: dict[tuple[str, str], list[float]] = {}

    regimes = (
        "TREND_UP",
        "TREND_DOWN",
        "HIGH_VOLATILITY",
        "RANGE",
    )

    flow_buckets = (
        "BUY_DOMINANT",
        "BALANCED",
        "SELL_DOMINANT",
    )

    for regime in regimes:
        for flow_bucket in flow_buckets:
            observations[(regime, flow_bucket)] = []

    forward_returns: list[float] = []

    for index in range(
        minimum - 1,
        len(price_values) - 1,
    ):
        historical_prices = price_values[
            index - minimum + 1 : index + 1
        ]

        regime_result = detect_regime(
            historical_prices,
            config=regime_config,
        )

        flow_result = calculate_order_flow(
            buy_volume=buy_values[index],
            sell_volume=sell_values[index],
        )

        if flow_result.imbalance >= imbalance_threshold:
            flow_bucket = "BUY_DOMINANT"
        elif flow_result.imbalance <= -imbalance_threshold:
            flow_bucket = "SELL_DOMINANT"
        else:
            flow_bucket = "BALANCED"

        forward_return = (
            price_values[index + 1] / price_values[index]
        ) - 1.0

        observations[
            (regime_result.regime, flow_bucket)
        ].append(forward_return)

        forward_returns.append(forward_return)

    return ConditionalEvaluationResult(
        total_observations=len(forward_returns),
        overall_mean_forward_return=float(
            mean(forward_returns)
        ),
        overall_positive_rate=_positive_rate(
            forward_returns
        ),
        trend_up_buy_dominant=_performance(
            observations[("TREND_UP", "BUY_DOMINANT")]
        ),
        trend_up_balanced=_performance(
            observations[("TREND_UP", "BALANCED")]
        ),
        trend_up_sell_dominant=_performance(
            observations[("TREND_UP", "SELL_DOMINANT")]
        ),
        trend_down_buy_dominant=_performance(
            observations[("TREND_DOWN", "BUY_DOMINANT")]
        ),
        trend_down_balanced=_performance(
            observations[("TREND_DOWN", "BALANCED")]
        ),
        trend_down_sell_dominant=_performance(
            observations[("TREND_DOWN", "SELL_DOMINANT")]
        ),
        high_volatility_buy_dominant=_performance(
            observations[("HIGH_VOLATILITY", "BUY_DOMINANT")]
        ),
        high_volatility_balanced=_performance(
            observations[("HIGH_VOLATILITY", "BALANCED")]
        ),
        high_volatility_sell_dominant=_performance(
            observations[("HIGH_VOLATILITY", "SELL_DOMINANT")]
        ),
        range_buy_dominant=_performance(
            observations[("RANGE", "BUY_DOMINANT")]
        ),
        range_balanced=_performance(
            observations[("RANGE", "BALANCED")]
        ),
        range_sell_dominant=_performance(
            observations[("RANGE", "SELL_DOMINANT")]
        ),
    )


def _performance(
    returns: list[float],
) -> ConditionalPerformance:
    if not returns:
        return ConditionalPerformance(
            observations=0,
            mean_forward_return=0.0,
            positive_rate=0.0,
        )

    return ConditionalPerformance(
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


def _validate_inputs(
    *,
    buy_volumes: list[float],
    sell_volumes: list[float],
    prices: list[float],
    imbalance_threshold: float,
) -> tuple[list[float], list[float], list[float]]:
    buy_values = _to_numeric_sequence(
        buy_volumes,
        "buy_volumes",
    )

    sell_values = _to_numeric_sequence(
        sell_volumes,
        "sell_volumes",
    )

    price_values = _to_numeric_sequence(
        prices,
        "prices",
    )

    if len(price_values) < 2:
        raise ValueError(
            "prices must contain at least 2 observations"
        )

    if len(buy_values) != len(sell_values):
        raise ValueError(
            "buy_volumes and sell_volumes must have equal length"
        )

    if len(buy_values) != len(price_values):
        raise ValueError(
            "volume and price sequences must have equal length"
        )

    if (
        isinstance(imbalance_threshold, bool)
        or not isinstance(
            imbalance_threshold,
            (int, float),
        )
    ):
        raise ValueError(
            "imbalance_threshold must be numeric"
        )

    threshold = float(imbalance_threshold)

    if not math.isfinite(threshold):
        raise ValueError(
            "imbalance_threshold must be finite"
        )

    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "imbalance_threshold must be greater than 0 and at most 1"
        )

    for value in buy_values:
        if value < 0.0:
            raise ValueError(
                "buy_volumes must contain only non-negative values"
            )

    for value in sell_values:
        if value < 0.0:
            raise ValueError(
                "sell_volumes must contain only non-negative values"
            )

    for value in price_values:
        if value <= 0.0:
            raise ValueError(
                "prices must be positive"
            )

    for buy, sell in zip(
        buy_values,
        sell_values,
    ):
        if buy + sell <= 0.0:
            raise ValueError(
                "each order-flow observation must have positive total volume"
            )

    return buy_values, sell_values, price_values


def _validate_regime_config(
    config: RegimeConfig,
) -> None:
    if not isinstance(config, RegimeConfig):
        raise ValueError(
            "regime_config must be a RegimeConfig"
        )


def _to_numeric_sequence(
    values: list[float],
    name: str,
) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{name} must be a numeric sequence"
        )

    try:
        sequence = list(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a numeric sequence"
        ) from exc

    result: list[float] = []

    for value in sequence:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise ValueError(
                f"{name} must contain only numeric values"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                f"{name} must contain only finite values"
            )

        result.append(numeric)

    return result
