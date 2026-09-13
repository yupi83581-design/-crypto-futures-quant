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

    Probability buckets use left-closed, right-open intervals:

        [0.0, 0.1)
        [0.1, 0.2)
        ...
        [0.9, 1.0]

    Therefore a probability exactly equal to a bucket boundary belongs
    to the bucket beginning at that boundary. Probability 1.0 belongs
    to the final bucket.

    The function calculates:

    - Brier score
    - mean predicted probability per non-empty bucket
    - observed positive frequency per non-empty bucket
    - sample count per non-empty bucket

    This module evaluates probability quality only. It does not claim
    profitability, trading edge, or future performance.
    """
    _validate_inputs(actual, probabilities, bucket_count)

    if not actual:
        return CalibrationResult(
            brier_score=0.0,
            buckets=(),
        )

    brier_score = sum(
        (float(probability) - int(outcome)) ** 2
        for outcome, probability in zip(actual, probabilities)
    ) / len(actual)

    buckets: list[CalibrationBucket] = []

    for bucket_index in range(bucket_count):
        lower_bound = bucket_index / bucket_count
        upper_bound = (bucket_index + 1) / bucket_count

        bucket_items = []

        for outcome, probability in zip(actual, probabilities):
            probability = float(probability)

            if bucket_index == bucket_count - 1:
                belongs_to_bucket = (
                    lower_bound <= probability <= upper_bound
                )
            else:
                belongs_to_bucket = (
                    lower_bound <= probability < upper_bound
                )

            if belongs_to_bucket:
                bucket_items.append(
                    (int(outcome), probability)
                )

        if not bucket_items:
            continue

        predicted_probability = sum(
            probability
            for _, probability in bucket_items
        ) / len(bucket_items)

        observed_frequency = sum(
            outcome
            for outcome, _ in bucket_items
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
        brier_score=float(brier_score),
        buckets=tuple(buckets),
    )


def _validate_inputs(
    actual: Sequence[int],
    probabilities: Sequence[float],
    bucket_count: int,
) -> None:
    """Validate calibration inputs."""

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
            "bucket_count must be a positive integer, "
            f"got {bucket_count!r}"
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

        probability = float(probability)

        if not math.isfinite(probability):
            raise ValueError(
                f"probability at index {index} must be finite, "
                f"got {probability!r}"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability at index {index} must be between "
                f"0 and 1, got {probability!r}"
            )
