"""Evaluation metrics for binary quantitative research models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class BinaryClassificationResult:
    """Evaluation result for a binary classifier."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float


def evaluate_binary_classification(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> BinaryClassificationResult:
    """Evaluate binary predictions against known labels.

    Labels must be binary integers:
        0 -> negative class
        1 -> positive class

    Metrics use standard binary-classification definitions.

    Degenerate metric conventions:
        - precision = 0.0 when there are no predicted positives
        - recall = 0.0 when there are no actual positives
        - accuracy = 0.0 when there are no observations
    """
    actual_labels = _validate_labels(actual, "actual")
    predicted_labels = _validate_labels(predicted, "predicted")

    if len(actual_labels) != len(predicted_labels):
        raise ValueError(
            "actual and predicted must contain the same number "
            "of observations"
        )

    if not actual_labels:
        return BinaryClassificationResult(
            true_positive=0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
        )

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for actual_label, predicted_label in zip(
        actual_labels,
        predicted_labels,
    ):
        if actual_label == 1 and predicted_label == 1:
            true_positive += 1
        elif actual_label == 0 and predicted_label == 0:
            true_negative += 1
        elif actual_label == 0 and predicted_label == 1:
            false_positive += 1
        else:
            false_negative += 1

    total = len(actual_labels)

    accuracy = (true_positive + true_negative) / total

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative

    precision = (
        true_positive / predicted_positive
        if predicted_positive > 0
        else 0.0
    )

    recall = (
        true_positive / actual_positive
        if actual_positive > 0
        else 0.0
    )

    return BinaryClassificationResult(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
    )


def evaluate_probability_predictions(
    actual: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> BinaryClassificationResult:
    """Evaluate probability predictions after applying a threshold.

    A probability greater than or equal to ``threshold`` becomes class 1.
    A probability below the threshold becomes class 0.
    """
    if not isinstance(threshold, (int, float)) or isinstance(
        threshold,
        bool,
    ):
        raise ValueError("threshold must be numeric")

    threshold_value = float(threshold)

    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite")

    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    validated_probabilities = _validate_probabilities(probabilities)

    if len(actual) != len(validated_probabilities):
        raise ValueError(
            "actual and probabilities must contain the same number "
            "of observations"
        )

    predicted = [
        1 if probability >= threshold_value else 0
        for probability in validated_probabilities
    ]

    return evaluate_binary_classification(actual, predicted)


def _validate_labels(
    labels: Sequence[int],
    name: str,
) -> list[int]:
    """Validate binary integer labels."""
    validated: list[int] = []

    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"{name} label at index {index} must be an integer 0 or 1"
            )

        if label not in (0, 1):
            raise ValueError(
                f"{name} label at index {index} must be 0 or 1, "
                f"got {label!r}"
            )

        validated.append(label)

    return validated


def _validate_probabilities(
    probabilities: Sequence[float],
) -> list[float]:
    """Validate probability predictions."""
    validated: list[float] = []

    for index, probability in enumerate(probabilities):
        if isinstance(probability, bool) or not isinstance(
            probability,
            (int, float),
        ):
            raise ValueError(
                f"probability at index {index} must be numeric, "
                f"got {probability!r}"
            )

        value = float(probability)

        if not math.isfinite(value):
            raise ValueError(
                f"probability at index {index} must be finite, "
                f"got {probability!r}"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"probability at index {index} must be between 0 and 1, "
                f"got {probability!r}"
            )

        validated.append(value)

    return validated
