"""Deterministic validation of conditional research edge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean


@dataclass(frozen=True)
class EdgeValidationResult:
    """Summary of whether a candidate edge improves on baseline."""

    observations: int
    candidate_mean_return: float
    baseline_mean_return: float
    mean_return_uplift: float
    candidate_positive_rate: float
    baseline_positive_rate: float
    positive_rate_uplift: float
    minimum_observations: int
    minimum_mean_uplift: float
    minimum_positive_rate_uplift: float
    passed: bool


def validate_edge(
    *,
    candidate_returns: list[float],
    baseline_returns: list[float],
    minimum_observations: int = 30,
    minimum_mean_uplift: float = 0.0,
    minimum_positive_rate_uplift: float = 0.0,
) -> EdgeValidationResult:
    """Evaluate whether candidate returns improve on a baseline.

    The candidate and baseline are treated as already-defined research
    samples. This function performs descriptive validation only.

    It does not forecast prices, execute trades, or claim statistical
    significance. Statistical significance and out-of-sample robustness
    require a separate research protocol.
    """
    candidate = _validate_returns(
        candidate_returns,
        "candidate_returns",
    )
    baseline = _validate_returns(
        baseline_returns,
        "baseline_returns",
    )

    if isinstance(minimum_observations, bool) or not isinstance(
        minimum_observations,
        int,
    ):
        raise ValueError(
            "minimum_observations must be an integer"
        )

    if minimum_observations <= 0:
        raise ValueError(
            "minimum_observations must be greater than zero"
        )

    _validate_non_negative_finite(
        minimum_mean_uplift,
        "minimum_mean_uplift",
    )
    _validate_non_negative_finite(
        minimum_positive_rate_uplift,
        "minimum_positive_rate_uplift",
    )

    candidate_mean = float(mean(candidate))
    baseline_mean = float(mean(baseline))

    candidate_positive_rate = _positive_rate(candidate)
    baseline_positive_rate = _positive_rate(baseline)

    mean_uplift = candidate_mean - baseline_mean
    positive_rate_uplift = (
        candidate_positive_rate - baseline_positive_rate
    )

    passed = (
        len(candidate) >= minimum_observations
        and mean_uplift >= float(minimum_mean_uplift)
        and positive_rate_uplift
        >= float(minimum_positive_rate_uplift)
    )

    return EdgeValidationResult(
        observations=len(candidate),
        candidate_mean_return=candidate_mean,
        baseline_mean_return=baseline_mean,
        mean_return_uplift=mean_uplift,
        candidate_positive_rate=candidate_positive_rate,
        baseline_positive_rate=baseline_positive_rate,
        positive_rate_uplift=positive_rate_uplift,
        minimum_observations=minimum_observations,
        minimum_mean_uplift=float(minimum_mean_uplift),
        minimum_positive_rate_uplift=float(
            minimum_positive_rate_uplift
        ),
        passed=passed,
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


def _validate_returns(
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

    if not sequence:
        raise ValueError(
            f"{name} must not be empty"
        )

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
