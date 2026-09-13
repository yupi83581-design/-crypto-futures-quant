"""Probability calibration metrics for quantitative research models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBucket:
    """Calibration statistics for one probability interval."""

    lower_bound: float
    upper_bound: float
    predicted_probability: float
    observed_frequency: float
    sample_count: int


@dataclass(frozen=True)
class CalibrationResult:
    """Complete probability-calibration evaluation result."""

    brier_score: float
    buckets: tuple[CalibrationBucket, ...]


def evaluate_calibration(
    actual: Sequence[int],
    probabilities: Sequence[float],
    bucket_count: int = 10,
) -> CalibrationResult:
    """Evaluate binary probability predictions.

    Parameters
    ----------
    actual:
        Binary observed outcomes: 0 or 1.
    probabilities:
        Predicted probability of class 1 for each observation.
    bucket_count:
        Number of equally sized probability buckets.

    Returns
    -------
    CalibrationResult
        Contains the Brier score and non-empty calibration buckets.

    Notes
    -----
    This function evaluates calibration only. It does not claim that
    calibrated probabilities imply profitable trading performance.
    """
    _validate_inputs(actual, probabilities, bucket_count)

    if not actual:
        return CalibrationResult(
            brier_score=0.0,
            buckets=(),
        )

    brier_score = sum(
        (probability - outcome) ** 2
        for outcome, probability in zip(actual, probabilities)
    ) / len(actual)

    buckets: list[CalibrationBucket] = []

    for bucket_index in range(bucket_count):
        lower_bound = bucket_index / bucket_count

        if bucket_index == bucket_count - 1:
            upper_bound = 1.0
        else:
            upper_bound = (bucket_index + 1) / bucket_count

        bucket_items = [
            (outcome, probability)
            for outcome, probability in zip(actual, probabilities)
            if (
                probability >= lower_bound
                and (
                    probability <= upper_bound
                    if bucket_index == bucket_count - 1
                    else probability < upper_bound
                )
            )
        ]

        if not bucket_items:
            continue

        predicted_probability = sum(
            probability for _, probability in bucket_items
        ) / len(bucket_items)

        observed_frequency = sum(
            outcome for outcome, _ in bucket_items
        ) / len(bucket_items)

        buckets.append(
            CalibrationBucket(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                predicted_probability=predicted_probability,
                observed_frequency=observed_frequency,
                sample_count=len(bucket_items),
            )
        )

    return CalibrationResult(
        brier_score=brier_score,
        buckets=tuple(buckets),
    )


def _validate_inputs(
    actual: Sequence[int],
    probabilities: Sequence[float],
    bucket_count: int,
) -> None:
    """Validate calibration inputs before computation."""
    if len(actual) != len(probabilities):
        raise ValueError(
            "actual and probabilities must have the same length"
        )

    if (
        not isinstance(bucket_count, int)
        or isinstance(bucket_count, bool)
        or bucket_count < 1
    ):
        raise ValueError(
            f"bucket_count must be a positive integer, got {bucket_count!r}"
        )

    for index, outcome in enumerate(actual):
        if isinstance(outcome, bool) or outcome not in (0, 1):
            raise ValueError(
                f"actual at index {index} must be binary 0 or 1, "
                f"got {outcome!r}"
            )

    for index, probability in enumerate(probabilities):
        if isinstance(probability, bool) or not isinstance(
            probability,
            (int, float),
        ):
            raise ValueError(
                f"probability at index {index} must be numeric, "
                f"got {probability!r}"
            )

        if not math.isfinite(float(probability)):
            raise ValueError(
                f"probability at index {index} must be finite, "
                f"got {probability!r}"
            )

        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"probability at index {index} must be between 0 and 1, "
                f"got {probability!r}"
            )
