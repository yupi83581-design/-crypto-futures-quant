"""Deterministic baseline models for quantitative research."""

from __future__ import annotations

import math
from typing import Sequence


class MajorityClassBaseline:
    """Predicts the class observed most often during training."""

    def __init__(self) -> None:
        self._majority_class: int | None = None

    def fit(self, labels: Sequence[int]) -> "MajorityClassBaseline":
        """Fit the model using binary training labels."""
        validated = _validate_labels(labels)

        if not validated:
            raise ValueError("labels must not be empty")

        ones = sum(validated)
        zeros = len(validated) - ones

        self._majority_class = 1 if ones > zeros else 0
        return self

    def predict(self, count: int) -> list[int]:
        """Predict the fitted majority class."""
        self._require_fitted()

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")

        return [self._majority_class] * count  # type: ignore[list-item]

    def predict_proba(self, count: int) -> list[float]:
        """Return probability of class 1 for each prediction."""
        self._require_fitted()

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")

        probability = float(self._majority_class)
        return [probability] * count

    def _require_fitted(self) -> None:
        if self._majority_class is None:
            raise RuntimeError("model must be fitted before prediction")


class LogisticRegressionBaseline:
    """Small deterministic one-feature logistic regression model.

    The model accepts one numeric feature per observation and predicts the
    probability of binary class 1.

    Training uses batch gradient descent with L2 regularization.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        iterations: int = 5000,
        regularization: float = 0.01,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")

        if regularization < 0:
            raise ValueError("regularization must be non-negative")

        self.learning_rate = float(learning_rate)
        self.iterations = iterations
        self.regularization = float(regularization)

        self._weight: float | None = None
        self._bias: float | None = None
        self._feature_mean: float | None = None
        self._feature_scale: float | None = None

    def fit(
        self,
        features: Sequence[float],
        labels: Sequence[int],
    ) -> "LogisticRegressionBaseline":
        """Fit logistic regression using training features and labels."""
        x = _validate_features(features)
        y = _validate_labels(labels)

        if len(x) != len(y):
            raise ValueError(
                "features and labels must contain the same number of observations"
            )

        if not x:
            raise ValueError("features and labels must not be empty")

        feature_mean = sum(x) / len(x)
        variance = sum(
            (value - feature_mean) ** 2 for value in x
        ) / len(x)

        feature_scale = math.sqrt(variance)

        if feature_scale == 0.0:
            feature_scale = 1.0

        standardized = [
            (value - feature_mean) / feature_scale
            for value in x
        ]

        weight = 0.0
        bias = 0.0
        n = float(len(standardized))

        for _ in range(self.iterations):
            probabilities = [
                _sigmoid(weight * value + bias)
                for value in standardized
            ]

            weight_gradient = sum(
                (probability - target) * value
                for probability, target, value in zip(
                    probabilities,
                    y,
                    standardized,
                )
            ) / n

            bias_gradient = sum(
                probability - target
                for probability, target in zip(probabilities, y)
            ) / n

            if self.regularization > 0.0:
                weight_gradient += self.regularization * weight

            weight -= self.learning_rate * weight_gradient
            bias -= self.learning_rate * bias_gradient

        self._weight = weight
        self._bias = bias
        self._feature_mean = feature_mean
        self._feature_scale = feature_scale

        return self

    def predict_proba(
        self,
        features: Sequence[float],
    ) -> list[float]:
        """Return predicted probability of class 1."""
        self._require_fitted()

        x = _validate_features(features)

        return [
            self._predict_probability(value)
            for value in x
        ]

    def predict(
        self,
        features: Sequence[float],
        threshold: float = 0.5,
    ) -> list[int]:
        """Convert predicted probabilities into binary classes."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        probabilities = self.predict_proba(features)

        return [
            1 if probability >= threshold else 0
            for probability in probabilities
        ]

    def _predict_probability(self, feature: float) -> float:
        assert self._weight is not None
        assert self._bias is not None
        assert self._feature_mean is not None
        assert self._feature_scale is not None

        standardized = (
            feature - self._feature_mean
        ) / self._feature_scale

        return _sigmoid(
            self._weight * standardized + self._bias
        )

    def _require_fitted(self) -> None:
        if (
            self._weight is None
            or self._bias is None
            or self._feature_mean is None
            or self._feature_scale is None
        ):
            raise RuntimeError("model must be fitted before prediction")


def _validate_features(features: Sequence[float]) -> list[float]:
    """Validate and normalize numeric feature values."""
    validated: list[float] = []

    for index, value in enumerate(features):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"feature at index {index} must be numeric, got {value!r}"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                f"feature at index {index} must be finite, got {value!r}"
            )

        validated.append(numeric)

    return validated


def _validate_labels(labels: Sequence[int]) -> list[int]:
    """Validate binary labels."""
    validated: list[int] = []

    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(
                f"label at index {index} must be an integer 0 or 1"
            )

        if label not in (0, 1):
            raise ValueError(
                f"label at index {index} must be 0 or 1, got {label!r}"
            )

        validated.append(label)

    return validated


def _sigmoid(value: float) -> float:
    """Numerically stable sigmoid function."""
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)

    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
