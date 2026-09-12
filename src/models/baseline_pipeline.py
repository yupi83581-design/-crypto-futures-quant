"""Baseline RSI research pipeline.

Connects the verified RSI feature, direction label, baseline logistic
regression model, and binary evaluation engine.

This module is strictly for quantitative research and evaluation.
It contains no trading, order, leverage, or execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.features.rsi import DEFAULT_PERIOD, compute_rsi
from src.labels.direction import compute_direction_labels
from src.models.baseline import LogisticRegressionBaseline
from src.models.evaluation import (
    BinaryClassificationResult,
    evaluate_probability_predictions,
)


@dataclass(frozen=True)
class BaselinePipelineResult:
    """Complete out-of-sample evaluation result."""

    probabilities: tuple[float, ...]
    predictions: tuple[int, ...]
    actual_labels: tuple[int, ...]
    evaluation: BinaryClassificationResult


@dataclass(frozen=True)
class _PreparedDataset:
    """Internal feature/label dataset after temporal alignment."""

    features: tuple[float, ...]
    labels: tuple[int, ...]


class BaselineRSIPipeline:
    """Train on one temporal dataset and evaluate on another.

    The pipeline deliberately requires training and evaluation records
    separately. This prevents accidental random splitting and makes the
    temporal boundary explicit.

    RSI is computed independently inside each supplied dataset. Therefore
    the caller must provide enough historical warm-up records in the
    evaluation dataset when RSI values are required from its beginning.

    Parameters
    ----------
    rsi_period:
        RSI lookback period. Defaults to 14.
    label_horizon:
        Number of records ahead used by the direction label.
    threshold:
        Probability threshold used to convert class-1 probabilities into
        binary predictions.
    model:
        Optional preconfigured LogisticRegressionBaseline instance.
        If omitted, a default model is created.
    """

    def __init__(
        self,
        rsi_period: int = DEFAULT_PERIOD,
        label_horizon: int = 3,
        threshold: float = 0.5,
        model: LogisticRegressionBaseline | None = None,
    ) -> None:
        if (
            not isinstance(rsi_period, int)
            or isinstance(rsi_period, bool)
            or rsi_period < 1
        ):
            raise ValueError(
                f"rsi_period must be a positive integer, got {rsi_period!r}"
            )

        if (
            not isinstance(label_horizon, int)
            or isinstance(label_horizon, bool)
            or label_horizon < 1
        ):
            raise ValueError(
                "label_horizon must be a positive integer, "
                f"got {label_horizon!r}"
            )

        if isinstance(threshold, bool) or not isinstance(
            threshold,
            (int, float),
        ):
            raise ValueError("threshold must be numeric")

        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.rsi_period = rsi_period
        self.label_horizon = label_horizon
        self.threshold = float(threshold)
        self.model = model or LogisticRegressionBaseline()

        self._fitted = False

    def fit(
        self,
        training_records: Sequence[dict[str, Any]],
    ) -> "BaselineRSIPipeline":
        """Fit the baseline model using only training records.

        No evaluation records are accepted by this method.
        """
        dataset = self._prepare_dataset(training_records)

        if not dataset.features:
            raise ValueError(
                "training data does not contain enough valid "
                "RSI/label observations"
            )

        self.model.fit(
            dataset.features,
            dataset.labels,
        )

        self._fitted = True
        return self

    def evaluate(
        self,
        evaluation_records: Sequence[dict[str, Any]],
    ) -> BaselinePipelineResult:
        """Evaluate the already-fitted model on separate records."""
        if not self._fitted:
            raise RuntimeError(
                "pipeline must be fitted before evaluation"
            )

        dataset = self._prepare_dataset(evaluation_records)

        if not dataset.features:
            raise ValueError(
                "evaluation data does not contain enough valid "
                "RSI/label observations"
            )

        probabilities = self.model.predict_proba(
            dataset.features
        )

        evaluation = evaluate_probability_predictions(
            actual=dataset.labels,
            probabilities=probabilities,
            threshold=self.threshold,
        )

        predictions = tuple(
            1 if probability >= self.threshold else 0
            for probability in probabilities
        )

        return BaselinePipelineResult(
            probabilities=tuple(probabilities),
            predictions=predictions,
            actual_labels=dataset.labels,
            evaluation=evaluation,
        )

    def fit_and_evaluate(
        self,
        training_records: Sequence[dict[str, Any]],
        evaluation_records: Sequence[dict[str, Any]],
    ) -> BaselinePipelineResult:
        """Fit on training data and evaluate on separate data."""
        self.fit(training_records)
        return self.evaluate(evaluation_records)

    def _prepare_dataset(
        self,
        records: Sequence[dict[str, Any]],
    ) -> _PreparedDataset:
        """Build temporally aligned RSI features and direction labels."""
        normalized_records = _validate_records(records)

        if not normalized_records:
            return _PreparedDataset(
                features=(),
                labels=(),
            )

        rsi_values = compute_rsi(
            list(normalized_records),
            period=self.rsi_period,
        )

        direction_labels = compute_direction_labels(
            list(normalized_records),
            horizon=self.label_horizon,
        )

        features: list[float] = []
        labels: list[int] = []

        for rsi, label in zip(
            rsi_values,
            direction_labels,
        ):
            # RSI None means insufficient historical warm-up.
            # Label None means insufficient future horizon.
            #
            # Both must exist before an observation can become a
            # supervised-learning example.
            if rsi is None or label is None:
                continue

            features.append(float(rsi))
            labels.append(int(label))

        return _PreparedDataset(
            features=tuple(features),
            labels=tuple(labels),
        )


def _validate_records(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate the minimum flat-dict market-record contract."""
    validated: list[dict[str, Any]] = []

    previous_event_time: str | None = None

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"record at index {index} must be a dictionary"
            )

        if "close" not in record:
            raise ValueError(
                f"record at index {index} is missing required 'close' field"
            )

        close = record["close"]

        if isinstance(close, bool) or not isinstance(
            close,
            (int, float),
        ):
            raise ValueError(
                f"record at index {index} has non-numeric "
                f"'close': {close!r}"
            )

        if "event_time" not in record or record["event_time"] is None:
            raise ValueError(
                f"record at index {index} is missing required "
                "'event_time' field"
            )

        event_time = record["event_time"]

        if not isinstance(event_time, str):
            raise ValueError(
                f"record at index {index} has non-string "
                f"'event_time': {event_time!r}"
            )

        if previous_event_time is not None and event_time < previous_event_time:
            raise ValueError(
                "records must be sorted ascending by event_time: "
                f"index {index - 1} ({previous_event_time!r}) comes "
                f"after index {index} ({event_time!r})"
            )

        previous_event_time = event_time

        validated.append(record)

    return tuple(validated)
