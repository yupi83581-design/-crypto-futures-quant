"""Advanced model research utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean


@dataclass(frozen=True)
class AdvancedModelResult:
    """Summary of a deterministic candidate model evaluation."""

    observations: int
    candidate_mean_probability: float
    baseline_mean_probability: float
    mean_probability_uplift: float
    candidate_brier_score: float
    baseline_brier_score: float
    brier_score_improvement: float
    candidate_accuracy: float
    baseline_accuracy: float
    accuracy_uplift: float
    passed: bool


def evaluate_advanced_model(
    *,
    candidate_probabilities: list[float],
    baseline_probabilities: list[float],
    outcomes: list[int],
    minimum_observations: int = 30,
    minimum_probability_uplift: float = 0.0,
    minimum_brier_improvement: float = 0.0,
    minimum_accuracy_uplift: float = 0.0,
) -> AdvancedModelResult:
    """Compare candidate and baseline probability predictions.

    Outcomes must be binary:
        1 = positive outcome
        0 = non-positive outcome

    Brier score is:

        mean((probability - outcome) ** 2)

    Lower Brier score is better.

    This is a research comparison only. It does not execute trades
    and does not establish statistical significance or profitability.
    """
    candidate = _validate_probabilities(
        candidate_probabilities,
        "candidate_probabilities",
    )
    baseline = _validate_probabilities(
        baseline_probabilities,
        "baseline_probabilities",
    )
    outcome_values = _validate_outcomes(outcomes)

    if len(candidate) != len(baseline):
        raise ValueError(
            "candidate_probabilities and baseline_probabilities "
            "must have equal length"
        )

    if len(candidate) != len(outcome_values):
        raise ValueError(
            "probability and outcome sequences must have equal length"
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
        minimum_probability_uplift,
        "minimum_probability_uplift",
    )
    _validate_non_negative_finite(
        minimum_brier_improvement,
        "minimum_brier_improvement",
    )
    _validate_non_negative_finite(
        minimum_accuracy_uplift,
        "minimum_accuracy_uplift",
    )

    candidate_mean = float(mean(candidate))
    baseline_mean = float(mean(baseline))

    mean_probability_uplift = (
        candidate_mean - baseline_mean
    )

    candidate_brier = _brier_score(
        candidate,
        outcome_values,
    )
    baseline_brier = _brier_score(
        baseline,
        outcome_values,
    )

    brier_improvement = (
        baseline_brier - candidate_brier
    )

    candidate_accuracy = _accuracy(
        candidate,
        outcome_values,
    )
    baseline_accuracy = _accuracy(
        baseline,
        outcome_values,
    )

    accuracy_uplift = (
        candidate_accuracy - baseline_accuracy
    )

    passed = (
        len(candidate) >= minimum_observations
        and mean_probability_uplift
        >= float(minimum_probability_uplift)
        and brier_improvement
        >= float(minimum_brier_improvement)
        and accuracy_uplift
        >= float(minimum_accuracy_uplift)
    )

    return AdvancedModelResult(
        observations=len(candidate),
        candidate_mean_probability=candidate_mean,
        baseline_mean_probability=baseline_mean,
        mean_probability_uplift=mean_probability_uplift,
        candidate_brier_score=candidate_brier,
        baseline_brier_score=baseline_brier,
        brier_score_improvement=brier_improvement,
        candidate_accuracy=candidate_accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_uplift=accuracy_uplift,
        passed=passed,
    )


def _brier_score(
    probabilities: list[float],
    outcomes: list[int],
) -> float:
    errors = [
        (probability - outcome) ** 2
        for probability, outcome in zip(
            probabilities,
            outcomes,
        )
    ]

    return float(mean(errors))


def _accuracy(
    probabilities: list[float],
    outcomes: list[int],
) -> float:
    correct = sum(
        1
        for probability, outcome in zip(
            probabilities,
            outcomes,
        )
        if (
            probability >= 0.50
            and outcome == 1
        )
        or (
            probability < 0.50
            and outcome == 0
        )
    )

    return correct / len(outcomes)


def _validate_probabilities(
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

        if not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"{name} must contain probabilities between 0 and 1"
            )

        result.append(numeric)

    return result


def _validate_outcomes(
    values: list[int],
) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            "outcomes must be a numeric sequence"
        )

    try:
        sequence = list(values)
    except TypeError as exc:
        raise ValueError(
            "outcomes must be a numeric sequence"
        ) from exc

    if not sequence:
        raise ValueError(
            "outcomes must not be empty"
        )

    result: list[int] = []

    for value in sequence:
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise ValueError(
                "outcomes must contain only integer values"
            )

        if value not in (0, 1):
            raise ValueError(
                "outcomes must contain only 0 or 1"
            )

        result.append(value)

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
