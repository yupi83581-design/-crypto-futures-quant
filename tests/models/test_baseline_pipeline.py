"""Tests for the RSI baseline research pipeline."""

from __future__ import annotations

import math

import pytest

from src.models.baseline import LogisticRegressionBaseline
from src.models.baseline_pipeline import (
    BaselinePipelineResult,
    BaselineRSIPipeline,
)


def make_records(closes: list[float]) -> list[dict[str, object]]:
    """Build minimal valid chronological market records."""
    return [
        {
            "event_time": f"2026-01-01T00:{index:02d}:00Z",
            "close": close,
        }
        for index, close in enumerate(closes)
    ]


def test_pipeline_fit_returns_self() -> None:
    records = make_records(
        [
            100,
            101,
            102,
            101,
            100,
            101,
            102,
            103,
            102,
            104,
            105,
            104,
            106,
            107,
            108,
            109,
            108,
            110,
            111,
        ]
    )

    pipeline = BaselineRSIPipeline()

    result = pipeline.fit(records)

    assert result is pipeline


def test_pipeline_evaluate_returns_expected_result_contract() -> None:
    training = make_records(
        [
            100,
            101,
            102,
            103,
            102,
            104,
            105,
            104,
            106,
            107,
            108,
            107,
            109,
            110,
            111,
            112,
            111,
            113,
            114,
            115,
            114,
            116,
            117,
            118,
        ]
    )

    evaluation = make_records(
        [
            200,
            201,
            202,
            201,
            203,
            204,
            203,
            205,
            206,
            207,
            206,
            208,
            209,
            210,
            211,
            210,
            212,
            213,
            214,
            215,
        ]
    )

    pipeline = BaselineRSIPipeline()

    result = pipeline.fit_and_evaluate(
        training,
        evaluation,
    )

    assert isinstance(result, BaselinePipelineResult)
    assert len(result.probabilities) == len(result.predictions)
    assert len(result.predictions) == len(result.actual_labels)

    assert all(
        0.0 <= probability <= 1.0
        for probability in result.probabilities
    )

    assert all(
        prediction in (0, 1)
        for prediction in result.predictions
    )

    assert all(
        label in (0, 1)
        for label in result.actual_labels
    )


def test_pipeline_probability_predictions_use_configured_threshold() -> None:
    training = make_records(
        [
            100,
            101,
            102,
            103,
            104,
            103,
            105,
            106,
            107,
            106,
            108,
            109,
            110,
            111,
            112,
            113,
            112,
            114,
            115,
            116,
        ]
    )

    evaluation = make_records(
        [
            200,
            201,
            202,
            203,
            202,
            204,
            205,
            206,
            205,
            207,
            208,
            209,
            210,
            211,
            210,
            212,
            213,
            214,
            215,
        ]
    )

    pipeline = BaselineRSIPipeline(
        threshold=0.7,
    )

    result = pipeline.fit_and_evaluate(
        training,
        evaluation,
    )

    expected_predictions = tuple(
        1 if probability >= 0.7 else 0
        for probability in result.probabilities
    )

    assert result.predictions == expected_predictions


def test_pipeline_requires_fit_before_evaluation() -> None:
    records = make_records(
        [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
        ]
    )

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        RuntimeError,
        match="must be fitted before evaluation",
    ):
        pipeline.evaluate(records)


def test_pipeline_rejects_invalid_rsi_period() -> None:
    with pytest.raises(
        ValueError,
        match="rsi_period must be a positive integer",
    ):
        BaselineRSIPipeline(rsi_period=0)


def test_pipeline_rejects_boolean_rsi_period() -> None:
    with pytest.raises(
        ValueError,
        match="rsi_period must be a positive integer",
    ):
        BaselineRSIPipeline(rsi_period=True)


def test_pipeline_rejects_invalid_label_horizon() -> None:
    with pytest.raises(
        ValueError,
        match="label_horizon must be a positive integer",
    ):
        BaselineRSIPipeline(label_horizon=0)


def test_pipeline_rejects_boolean_label_horizon() -> None:
    with pytest.raises(
        ValueError,
        match="label_horizon must be a positive integer",
    ):
        BaselineRSIPipeline(label_horizon=True)


def test_pipeline_rejects_invalid_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
        BaselineRSIPipeline(threshold=1.1)


def test_pipeline_rejects_boolean_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be numeric",
    ):
        BaselineRSIPipeline(threshold=True)


def test_pipeline_accepts_boundary_thresholds() -> None:
    assert BaselineRSIPipeline(threshold=0.0).threshold == 0.0
    assert BaselineRSIPipeline(threshold=1.0).threshold == 1.0


def test_pipeline_rejects_missing_close() -> None:
    records = [
        {
            "event_time": "2026-01-01T00:00:00Z",
        }
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="missing required 'close' field",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_nonnumeric_close() -> None:
    records = [
        {
            "event_time": "2026-01-01T00:00:00Z",
            "close": "100",
        }
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="non-numeric 'close'",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_boolean_close() -> None:
    records = [
        {
            "event_time": "2026-01-01T00:00:00Z",
            "close": True,
        }
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="non-numeric 'close'",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_missing_event_time() -> None:
    records = [
        {
            "close": 100,
        }
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="missing required 'event_time' field",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_non_string_event_time() -> None:
    records = [
        {
            "event_time": 123,
            "close": 100,
        }
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="non-string 'event_time'",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_unsorted_event_times() -> None:
    records = [
        {
            "event_time": "2026-01-01T00:05:00Z",
            "close": 100,
        },
        {
            "event_time": "2026-01-01T00:00:00Z",
            "close": 101,
        },
    ]

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="records must be sorted ascending",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_insufficient_training_data() -> None:
    records = make_records([100, 101, 102])

    pipeline = BaselineRSIPipeline()

    with pytest.raises(
        ValueError,
        match="training data does not contain enough",
    ):
        pipeline.fit(records)


def test_pipeline_rejects_insufficient_evaluation_data() -> None:
    training = make_records(
        [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
        ]
    )

    evaluation = make_records([200, 201, 202])

    pipeline = BaselineRSIPipeline()

    pipeline.fit(training)

    with pytest.raises(
        ValueError,
        match="evaluation data does not contain enough",
    ):
        pipeline.evaluate(evaluation)


def test_pipeline_respects_custom_model_configuration() -> None:
    model = LogisticRegressionBaseline(
        learning_rate=0.01,
        iterations=100,
        regularization=0.1,
    )

    pipeline = BaselineRSIPipeline(model=model)

    assert pipeline.model is model


def test_pipeline_does_not_train_model_in_constructor() -> None:
    model = LogisticRegressionBaseline()

    pipeline = BaselineRSIPipeline(model=model)

    assert pipeline.model is model
    assert not pipeline._fitted


def test_pipeline_fit_changes_fitted_state() -> None:
    records = make_records(
        [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
        ]
    )

    pipeline = BaselineRSIPipeline()

    assert not pipeline._fitted

    pipeline.fit(records)

    assert pipeline._fitted


def test_pipeline_output_is_deterministic() -> None:
    training = make_records(
        [
            100,
            101,
            102,
            101,
            103,
            104,
            103,
            105,
            106,
            107,
            106,
            108,
            109,
            110,
            111,
            110,
            112,
            113,
            114,
            115,
        ]
    )

    evaluation = make_records(
        [
            200,
            201,
            200,
            202,
            203,
            204,
            203,
            205,
            206,
            207,
            206,
            208,
            209,
            210,
            211,
            210,
            212,
            213,
            214,
            215,
        ]
    )

    first = BaselineRSIPipeline().fit_and_evaluate(
        training,
        evaluation,
    )

    second = BaselineRSIPipeline().fit_and_evaluate(
        training,
        evaluation,
    )

    assert first == second


def test_pipeline_has_no_future_label_for_final_horizon_records() -> None:
    records = make_records(
        [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
        ]
    )

    pipeline = BaselineRSIPipeline(
        label_horizon=3,
    )

    dataset = pipeline._prepare_dataset(records)

    expected_observations = len(records) - 14 - 3

    assert len(dataset.features) == expected_observations
    assert len(dataset.labels) == expected_observations


def test_pipeline_does_not_use_future_records_for_rsi_features() -> None:
    prefix = make_records(
        [
            100,
            101,
            102,
            101,
            103,
            104,
            105,
            104,
            106,
            107,
            108,
            107,
            109,
            110,
            111,
            112,
            111,
            113,
            114,
            115,
        ]
    )

    extended = prefix + make_records(
        [
            1000,
            2000,
            3000,
        ]
    )

    prefix_dataset = BaselineRSIPipeline()._prepare_dataset(prefix)
    extended_dataset = BaselineRSIPipeline()._prepare_dataset(extended)

    prefix_feature_count = len(prefix_dataset.features)

    assert prefix_feature_count > 0

    assert prefix_dataset.features == extended_dataset.features[:prefix_feature_count]


def test_pipeline_probability_values_are_finite() -> None:
    training = make_records(
        [
            100,
            101,
            102,
            103,
            102,
            104,
            105,
            106,
            105,
            107,
            108,
            109,
            110,
            109,
            111,
            112,
            113,
            114,
            113,
            115,
        ]
    )

    evaluation = make_records(
        [
            200,
            201,
            202,
            201,
            203,
            204,
            205,
            204,
            206,
            207,
            208,
            207,
            209,
            210,
            211,
            212,
            211,
            213,
            214,
            215,
        ]
    )

    result = BaselineRSIPipeline().fit_and_evaluate(
        training,
        evaluation,
    )

    assert all(math.isfinite(value) for value in result.probabilities)
