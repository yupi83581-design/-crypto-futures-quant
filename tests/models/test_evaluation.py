"""Tests for binary model evaluation metrics."""

from __future__ import annotations

import math

import pytest

from src.models.evaluation import (
    BinaryClassificationResult,
    evaluate_binary_classification,
    evaluate_probability_predictions,
)


class TestEvaluateBinaryClassification:
    def test_perfect_classification(self) -> None:
        result = evaluate_binary_classification(
            actual=[0, 0, 1, 1],
            predicted=[0, 0, 1, 1],
        )

        assert result == BinaryClassificationResult(
            true_positive=2,
            true_negative=2,
            false_positive=0,
            false_negative=0,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
        )

    def test_all_predictions_wrong(self) -> None:
        result = evaluate_binary_classification(
            actual=[0, 0, 1, 1],
            predicted=[1, 1, 0, 0],
        )

        assert result == BinaryClassificationResult(
            true_positive=0,
            true_negative=0,
            false_positive=2,
            false_negative=2,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
        )

    def test_confusion_matrix_counts_are_correct(self) -> None:
        result = evaluate_binary_classification(
            actual=[1, 1, 1, 0, 0, 0],
            predicted=[1, 0, 1, 1, 0, 0],
        )

        assert result.true_positive == 2
        assert result.true_negative == 2
        assert result.false_positive == 1
        assert result.false_negative == 1

    def test_accuracy_is_correct(self) -> None:
        result = evaluate_binary_classification(
            actual=[1, 1, 1, 0, 0],
            predicted=[1, 0, 1, 1, 0],
        )

        assert result.accuracy == pytest.approx(0.6)

    def test_precision_is_correct(self) -> None:
        result = evaluate_binary_classification(
            actual=[1, 1, 0, 0],
            predicted=[1, 0, 1, 0],
        )

        assert result.precision == pytest.approx(0.5)

    def test_recall_is_correct(self) -> None:
        result = evaluate_binary_classification(
            actual=[1, 1, 0, 0],
            predicted=[1, 0, 1, 0],
        )

        assert result.recall == pytest.approx(0.5)

    def test_no_predicted_positive_gives_zero_precision(self) -> None:
        result = evaluate_binary_classification(
            actual=[0, 0, 1, 1],
            predicted=[0, 0, 0, 0],
        )

        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_no_actual_positive_gives_zero_recall(self) -> None:
        result = evaluate_binary_classification(
            actual=[0, 0, 0, 0],
            predicted=[0, 0, 0, 1],
        )

        assert result.recall == 0.0
        assert result.precision == 0.0

    def test_empty_inputs_return_zero_metrics(self) -> None:
        result = evaluate_binary_classification(
            actual=[],
            predicted=[],
        )

        assert result == BinaryClassificationResult(
            true_positive=0,
            true_negative=0,
            false_positive=0,
            false_negative=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
        )

    def test_length_mismatch_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            evaluate_binary_classification(
                actual=[0, 1],
                predicted=[0],
            )

    def test_invalid_actual_label_fails(self) -> None:
        with pytest.raises(ValueError, match="actual label"):
            evaluate_binary_classification(
                actual=[0, 2],
                predicted=[0, 1],
            )

    def test_invalid_predicted_label_fails(self) -> None:
        with pytest.raises(ValueError, match="predicted label"):
            evaluate_binary_classification(
                actual=[0, 1],
                predicted=[0, 2],
            )

    def test_boolean_actual_label_fails(self) -> None:
        with pytest.raises(ValueError, match="integer 0 or 1"):
            evaluate_binary_classification(
                actual=[False, True],
                predicted=[0, 1],
            )

    def test_boolean_predicted_label_fails(self) -> None:
        with pytest.raises(ValueError, match="integer 0 or 1"):
            evaluate_binary_classification(
                actual=[0, 1],
                predicted=[False, True],
            )


class TestEvaluateProbabilityPredictions:
    def test_default_threshold_is_point_five(self) -> None:
        result = evaluate_probability_predictions(
            actual=[0, 0, 1, 1],
            probabilities=[0.1, 0.4, 0.6, 0.9],
        )

        assert result.true_positive == 2
        assert result.true_negative == 2
        assert result.false_positive == 0
        assert result.false_negative == 0

    def test_custom_threshold_changes_predictions(self) -> None:
        result = evaluate_probability_predictions(
            actual=[0, 1, 1, 1],
            probabilities=[0.2, 0.4, 0.6, 0.8],
            threshold=0.7,
        )

        assert result.true_positive == 1
        assert result.true_negative == 1
        assert result.false_positive == 0
        assert result.false_negative == 2

    def test_probability_equal_to_threshold_is_positive(self) -> None:
        result = evaluate_probability_predictions(
            actual=[1],
            probabilities=[0.5],
            threshold=0.5,
        )

        assert result.true_positive == 1

    def test_zero_threshold_predicts_everything_positive(self) -> None:
        result = evaluate_probability_predictions(
            actual=[0, 1],
            probabilities=[0.0, 0.0],
            threshold=0.0,
        )

        assert result.false_positive == 1
        assert result.true_positive == 1

    def test_one_threshold_requires_probability_one(self) -> None:
        result = evaluate_probability_predictions(
            actual=[0, 1],
            probabilities=[1.0, 0.9],
            threshold=1.0,
        )

        assert result.false_positive == 1
        assert result.false_negative == 1

    def test_probability_length_mismatch_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            evaluate_probability_predictions(
                actual=[0, 1],
                probabilities=[0.5],
            )

    def test_probability_below_zero_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            evaluate_probability_predictions(
                actual=[0],
                probabilities=[-0.1],
            )

    def test_probability_above_one_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[1.1],
            )

    def test_nan_probability_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="must be finite",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[math.nan],
            )

    def test_infinite_probability_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="must be finite",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[math.inf],
            )

    def test_boolean_probability_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="must be numeric",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[True],
            )

    def test_invalid_threshold_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[0.5],
                threshold=1.1,
            )

    def test_negative_threshold_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[0.5],
                threshold=-0.1,
            )

    def test_nan_threshold_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="finite",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[0.5],
                threshold=math.nan,
            )

    def test_boolean_threshold_fails(self) -> None:
        with pytest.raises(
            ValueError,
            match="threshold must be numeric",
        ):
            evaluate_probability_predictions(
                actual=[1],
                probabilities=[0.5],
                threshold=True,
            )

    def test_empty_probability_input(self) -> None:
        result = evaluate_probability_predictions(
            actual=[],
            probabilities=[],
        )

        assert result.accuracy == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_probability_metrics_match_binary_evaluation(self) -> None:
        actual = [0, 1, 1, 0, 1]
        probabilities = [0.2, 0.8, 0.7, 0.6, 0.3]

        probability_result = evaluate_probability_predictions(
            actual=actual,
            probabilities=probabilities,
            threshold=0.5,
        )

        binary_result = evaluate_binary_classification(
            actual=actual,
            predicted=[0, 1, 1, 1, 0],
        )

        assert probability_result == binary_result
