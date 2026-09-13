"""Expected-value and trading-cost calculations for quantitative research."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExpectedValueResult:
    """Expected-value calculation for one binary trade outcome."""

    probability: float
    reward: float
    loss: float
    fee: float
    slippage: float
    gross_expected_value: float
    total_cost: float
    net_expected_value: float


def calculate_expected_value(
    probability: float,
    reward: float,
    loss: float,
    fee: float = 0.0,
    slippage: float = 0.0,
) -> ExpectedValueResult:
    """Calculate gross and net expected value.

    Parameters
    ----------
    probability:
        Probability of the positive/reward outcome.
        Must be between 0 and 1.

    reward:
        Positive payoff when the favorable outcome occurs.

    loss:
        Positive magnitude of the loss when the unfavorable
        outcome occurs.

    fee:
        Non-negative trading cost expressed in the same payoff
        units as reward and loss.

    slippage:
        Non-negative execution-cost estimate expressed in the
        same payoff units as reward and loss.

    Returns
    -------
    ExpectedValueResult
        Gross expected value before costs and net expected value
        after fee and slippage.

    Formula
    -------
    Gross EV = P(reward) * reward - P(loss) * loss

    Net EV = Gross EV - fee - slippage

    Notes
    -----
    This is a mathematical research calculation. A positive
    expected value does not guarantee that any individual trade
    will be profitable or that future realized performance will
    match the estimate.
    """
    _validate_probability(probability)
    _validate_positive_value(reward, "reward")
    _validate_positive_value(loss, "loss")
    _validate_non_negative_value(fee, "fee")
    _validate_non_negative_value(slippage, "slippage")

    probability = float(probability)
    reward = float(reward)
    loss = float(loss)
    fee = float(fee)
    slippage = float(slippage)

    gross_expected_value = (
        probability * reward
        - (1.0 - probability) * loss
    )

    total_cost = fee + slippage
    net_expected_value = gross_expected_value - total_cost

    return ExpectedValueResult(
        probability=probability,
        reward=reward,
        loss=loss,
        fee=fee,
        slippage=slippage,
        gross_expected_value=gross_expected_value,
        total_cost=total_cost,
        net_expected_value=net_expected_value,
    )


def break_even_probability(
    reward: float,
    loss: float,
    fee: float = 0.0,
    slippage: float = 0.0,
) -> float:
    """Calculate the probability required for zero net expected value.

    The break-even probability satisfies:

        p * reward - (1 - p) * loss - fee - slippage = 0

    Therefore:

        p = (loss + fee + slippage) / (reward + loss)

    The returned value may exceed 1.0 when costs are economically
    impossible to overcome with the supplied reward/loss structure.

    Raises
    ------
    ValueError
        If reward/loss/cost inputs are invalid.
    """
    _validate_positive_value(reward, "reward")
    _validate_positive_value(loss, "loss")
    _validate_non_negative_value(fee, "fee")
    _validate_non_negative_value(slippage, "slippage")

    reward = float(reward)
    loss = float(loss)
    fee = float(fee)
    slippage = float(slippage)

    denominator = reward + loss

    if denominator <= 0.0:
        raise ValueError(
            "reward + loss must be positive"
        )

    return (loss + fee + slippage) / denominator


def _validate_probability(probability: float) -> None:
    if isinstance(probability, bool) or not isinstance(
        probability,
        (int, float),
    ):
        raise ValueError(
            f"probability must be numeric, got {probability!r}"
        )

    probability = float(probability)

    if not math.isfinite(probability):
        raise ValueError(
            f"probability must be finite, got {probability!r}"
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"probability must be between 0 and 1, got {probability!r}"
        )


def _validate_positive_value(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric, got {value!r}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {value!r}"
        )

    if value <= 0.0:
        raise ValueError(
            f"{name} must be positive, got {value!r}"
        )


def _validate_non_negative_value(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric, got {value!r}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {value!r}"
        )

    if value < 0.0:
        raise ValueError(
            f"{name} must be non-negative, got {value!r}"
        )
