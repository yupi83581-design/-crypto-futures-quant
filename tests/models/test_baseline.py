"""Tests for deterministic baseline models."""

from __future__ import annotations

import pytest

from src.models.baseline import (
    LogisticRegressionBaseline,
    MajorityClassBaseline,
)


class TestMajorityClassBaseline:
    def test_predicts_majority_class_zero(self) -> None:
        model = MajorityClassBaseline()

        model.fit([0, 0, 0, 1])

        assert model.predict(4) == [0, 0, 0, 0]
        assert model.predict_proba(4) == [0.0, 0.0, 0.0, 0.0]

    def test_predicts_majority_class_one(self) -> None:
        model = MajorityClassBaseline()

        model.fit([1, 1, 1, 0])

        assert model.predict(3) == [1, 1, 1]
        assert model.predict_proba(3) == [1.0, 1.0, 1.0]

    def test_tie_deterministically_selects_zero(self) -> None:
        model = MajorityClassBaseline()

        model.fit([0, 1])

        assert model.predict(2) == [0, 0]
        assert model.predict_proba(2) == [0.0, 0.0]

    def test_fit_returns_self(self) -> None:
        model = MajorityClassBaseline()

        result = model.fit([0, 1, 1])

        assert result is model

    def test_predict_before_fit_fails(self) -> None:
        model = MajorityClassBaseline()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(1)

    def test_predict_proba_before_fit_fails(self) -> None:
        model = MajorityClassBaseline()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_proba(1)

    def test_empty_training_labels_fail(self) -> None:
        model = MajorityClassBaseline()

        with pytest.raises(ValueError, match="must not be empty"):
            model.fit([])

    def test_invalid_labels_fail(self) -> None:
        model = MajorityClassBaseline()

        with pytest.raises(ValueError, match="0 or 1"):
            model.fit([0, 2])

    def test_boolean_labels_fail(self) -> None:
        model = MajorityClassBaseline()

        with pytest.raises(ValueError, match="integer"):
            model.fit([0, True])

    def test_negative_prediction_count_fails(self) -> None:
        model = MajorityClassBaseline()
        model.fit([0, 1, 1])

        with pytest.raises(ValueError, match="non-negative integer"):
            model.predict(-1)

    def test_boolean_prediction_count_fails(self) -> None:
        model = MajorityClassBaseline()
        model.fit([0, 1, 1])

        with pytest.raises(ValueError, match="non-negative integer"):
            model.predict(True)


class TestLogisticRegressionBaseline:
    def test_fit_returns_self(self) -> None:
        model = LogisticRegressionBaseline()

        result = model.fit(
            [-2.0, -1.0, 1.0, 2.0],
            [0, 0, 1, 1],
        )

        assert result is model

    def test_learns_positive_relationship(self) -> None:
        model = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
            regularization=0.01,
        )

        model.fit(
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [0, 0, 0, 1, 1, 1],
        )

        probabilities = model.predict_proba([-2.0, 2.0])

        assert probabilities[0] < 0.5
        assert probabilities[1] > 0.5
        assert probabilities[0] < probabilities[1]

    def test_predict_uses_default_threshold(self) -> None:
        model = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
        )

        model.fit(
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [0, 0, 0, 1, 1, 1],
        )

        predictions = model.predict([-2.0, 2.0])

        assert predictions == [0, 1]

    def test_predict_supports_custom_threshold(self) -> None:
        model = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
        )

        model.fit(
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [0, 0, 0, 1, 1, 1],
        )

        probabilities = model.predict_proba([0.0])

        assert model.predict([0.0], threshold=0.0) == [1]
        assert model.predict([0.0], threshold=1.0) == [0]

        assert 0.0 < probabilities[0] < 1.0

    def test_probabilities_are_between_zero_and_one(self) -> None:
        model = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
        )

        model.fit(
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [0, 0, 0, 1, 1, 1],
        )

        probabilities = model.predict_proba(
            [-100.0, -1.0, 0.0, 1.0, 100.0]
        )

        assert all(0.0 <= value <= 1.0 for value in probabilities)

    def test_prediction_length_matches_input(self) -> None:
        model = LogisticRegressionBaseline()

        model.fit(
            [-2.0, -1.0, 1.0, 2.0],
            [0, 0, 1, 1],
        )

        features = [-3.0, -1.0, 0.0, 1.0, 3.0]

        assert len(model.predict(features)) == len(features)
        assert len(model.predict_proba(features)) == len(features)

    def test_predict_before_fit_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict([1.0])

    def test_predict_proba_before_fit_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_proba([1.0])

    def test_feature_label_length_mismatch_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(
            ValueError,
            match="same number of observations",
        ):
            model.fit([1.0, 2.0], [0])

    def test_empty_training_data_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="must not be empty"):
            model.fit([], [])

    def test_invalid_feature_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="must be numeric"):
            model.fit([1.0, "bad"], [0, 1])  # type: ignore[list-item]

    def test_boolean_feature_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="must be numeric"):
            model.fit([1.0, True], [0, 1])

    def test_non_finite_feature_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="must be finite"):
            model.fit([1.0, float("nan")], [0, 1])

    def test_invalid_label_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="0 or 1"):
            model.fit([1.0, 2.0], [0, 2])

    def test_boolean_label_fails(self) -> None:
        model = LogisticRegressionBaseline()

        with pytest.raises(ValueError, match="integer"):
            model.fit([1.0, 2.0], [0, True])

    def test_constant_features_are_supported(self) -> None:
        model = LogisticRegressionBaseline()

        model.fit(
            [5.0, 5.0, 5.0, 5.0],
            [0, 0, 1, 1],
        )

        probabilities = model.predict_proba([5.0, 5.0])

        assert len(probabilities) == 2
        assert all(0.0 <= value <= 1.0 for value in probabilities)

    def test_invalid_learning_rate_fails(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            LogisticRegressionBaseline(learning_rate=0.0)

    def test_invalid_iterations_fail(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            LogisticRegressionBaseline(iterations=0)

    def test_boolean_iterations_fail(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            LogisticRegressionBaseline(iterations=True)

    def test_negative_regularization_fails(self) -> None:
        with pytest.raises(ValueError, match="regularization"):
            LogisticRegressionBaseline(regularization=-1.0)

    def test_invalid_threshold_fails(self) -> None:
        model = LogisticRegressionBaseline()

        model.fit(
            [-1.0, 1.0],
            [0, 1],
        )

        with pytest.raises(ValueError, match="threshold"):
            model.predict([0.0], threshold=-0.1)

        with pytest.raises(ValueError, match="threshold"):
            model.predict([0.0], threshold=1.1)

    def test_training_is_deterministic(self) -> None:
        features = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        labels = [0, 0, 0, 1, 1, 1]

        first = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
            regularization=0.01,
        )
        second = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
            regularization=0.01,
        )

        first.fit(features, labels)
        second.fit(features, labels)

        assert first.predict_proba(features) == second.predict_proba(features)
        assert first.predict(features) == second.predict(features)

    def test_prediction_does_not_depend_on_future_features(self) -> None:
        model_a = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
        )
        model_b = LogisticRegressionBaseline(
            learning_rate=0.1,
            iterations=3000,
        )

        training_features = [-3.0, -2.0, -1.0, 1.0]
        training_labels = [0, 0, 0, 1]

        model_a.fit(training_features, training_labels)
        model_b.fit(training_features, training_labels)

        historical_features = [-2.0, -1.0, 0.0, 1.0]

        assert (
            model_a.predict_proba(historical_features)
            == model_b.predict_proba(historical_features)
        )

    def test_predict_proba_accepts_empty_prediction_input(self) -> None:
        model = LogisticRegressionBaseline()

        model.fit(
            [-1.0, 1.0],
            [0, 1],
        )

        assert model.predict_proba([]) == []
        assert model.predict([]) == []
