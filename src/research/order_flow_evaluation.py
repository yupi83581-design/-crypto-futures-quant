"""Evaluation of order-flow information against forward returns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from src.research.order_flow import calculate_order_flow


@dataclass(frozen=True)
class OrderFlowPerformance:
    """Forward-return statistics for one order-flow bucket."""

    observations: int
    mean_forward_return: float
    positive_rate: float


@dataclass(frozen=True)
class OrderFlowEvaluationResult:
    """Forward returns conditioned on order-flow imbalance."""

    total_observations: int
    overall_mean_forward_return: float
    overall_positive_rate: float
    buy_dominant: OrderFlowPerformance
    balanced: OrderFlowPerformance
    sell_dominant: OrderFlowPerformance


def evaluate_order_flow_impact(
    *,
    buy_volumes: list[float],
    sell_volumes: list[float],
    prices: list[float],
    imbalance_threshold: float = 0.20,
) -> OrderFlowEvaluationResult:
    """Evaluate next-bar returns conditioned on order-flow imbalance.

    Order-flow data at time t determines the bucket.
    The return from price[t] to price[t + 1] is the outcome.

    The future price is therefore never used to construct the
    order-flow feature.

    This is research evaluation only. It does not execute trades
    or establish a statistically significant trading edge.
    """
    buy_values = _validate_inputs(
        buy_volumes=buy_volumes,
        sell_volumes=sell_volumes,
        prices=prices,
        imbalance_threshold=imbalance_threshold,
    )

    sell_values = _to_numeric_sequence(
        sell_volumes,
        "sell_volumes",
    )
    price_values = _to_numeric_sequence(
        prices,
        "prices",
    )

    observations: dict[str, list[float]] = {
        "BUY_DOMINANT": [],
        "BALANCED": [],
        "SELL_DOMINANT": [],
    }

    forward_returns: list[float] = []

    for index in range(len(price_values) - 1):
        flow = calculate_order_flow(
            buy_volume=buy_values[index],
            sell_volume=sell_values[index],
        )

        forward_return = (
            price_values[index + 1] / price_values[index]
        ) - 1.0

        if flow.imbalance >= imbalance_threshold:
            bucket = "BUY_DOMINANT"
        elif flow.imbalance <= -imbalance_threshold:
            bucket = "SELL_DOMINANT"
        else:
            bucket = "BALANCED"

        observations[bucket].append(forward_return)
        forward_returns.append(forward_return)

    return OrderFlowEvaluationResult(
        total_observations=len(forward_returns),
        overall_mean_forward_return=float(
            mean(forward_returns)
        ),
        overall_positive_rate=_positive_rate(
            forward_returns
        ),
        buy_dominant=_performance(
            observations["BUY_DOMINANT"]
        ),
        balanced=_performance(
            observations["BALANCED"]
        ),
        sell_dominant=_performance(
            observations["SELL_DOMINANT"]
        ),
    )


def _performance(
    returns: list[float],
) -> OrderFlowPerformance:
    if not returns:
        return OrderFlowPerformance(
            observations=0,
            mean_forward_return=0.0,
            positive_rate=0.0,
        )

    return OrderFlowPerformance(
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
) -> list[float]:
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

    return buy_values


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
