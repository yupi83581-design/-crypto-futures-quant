"""Tests for advanced model research utilities."""

from __future__ import annotations

import math

import pytest

from src.models.advanced import (
    AdvancedModelResult,
    evaluate_advanced_model,
)


def test_returns_expected_result_type() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.7, 0.9, 0.6],
        baseline_probabilities=[0.6, 0.6, 0.7, 0.5],
        outcomes=[1, 1, 1, 0],
    )

    assert isinstance(result, AdvancedModelResult)


def test_observation_count_is_correct() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.7, 0.9],
        baseline_probabilities=[0.6, 0.6, 0.7],
        outcomes=[1, 1, 0],
    )

    assert result.observations == 3


def test_candidate_mean_probability_is_calculated() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.6],
        baseline_probabilities=[0.5, 0.5],
        outcomes=[1, 0],
    )

    assert result.candidate_mean_probability == pytest.approx(0.7)


def test_baseline_mean_probability_is_calculated() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.6],
        baseline_probabilities=[0.4, 0.6],
        outcomes=[1, 0],
    )

    assert result.baseline_mean_probability == pytest.approx(0.5)


def test_mean_probability_uplift_is_candidate_minus_baseline() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.6],
        baseline_probabilities=[0.4, 0.4],
        outcomes=[1, 0],
    )

    assert result.mean_probability_uplift == pytest.approx(0.3)


def test_perfect_predictions_have_zero_brier_score() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[1.0, 0.0, 1.0, 0.0],
        baseline_probabilities=[0.5, 0.5, 0.5, 0.5],
        outcomes=[1, 0, 1, 0],
    )

    assert result.candidate_brier_score == pytest.approx(0.0)


def test_brier_score_matches_manual_calculation() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.2],
        baseline_probabilities=[0.5, 0.5],
        outcomes=[1, 0],
    )

    expected = ((0.8 - 1) ** 2 + (0.2 - 0) ** 2) / 2

    assert result.candidate_brier_score == pytest.approx(expected)


def test_baseline_brier_score_matches_manual_calculation() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.2],
        baseline_probabilities=[0.5, 0.5],
        outcomes=[1, 0],
    )

    expected = ((0.5 - 1) ** 2 + (0.5 - 0) ** 2) / 2

    assert result.baseline_brier_score == pytest.approx(expected)


def test_brier_improvement_is_baseline_minus_candidate() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1],
        baseline_probabilities=[0.6, 0.4],
        outcomes=[1, 0],
    )

    assert result.brier_score_improvement > 0.0


def test_accuracy_is_calculated_from_half_threshold() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1, 0.8, 0.2],
        baseline_probabilities=[0.6, 0.4, 0.4, 0.6],
        outcomes=[1, 0, 1, 0],
    )

    assert result.candidate_accuracy == pytest.approx(1.0)
    assert result.baseline_accuracy == pytest.approx(0.5)


def test_probability_exactly_half_predicts_positive() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.5],
        baseline_probabilities=[0.4],
        outcomes=[1],
    )

    assert result.candidate_accuracy == pytest.approx(1.0)


def test_accuracy_uplift_is_candidate_minus_baseline() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1],
        baseline_probabilities=[0.6, 0.6],
        outcomes=[1, 0],
    )

    assert result.accuracy_uplift == pytest.approx(0.5)


def test_passes_when_all_thresholds_are_met() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1] * 20,
        baseline_probabilities=[0.6, 0.4] * 20,
        outcomes=[1, 0] * 20,
        minimum_observations=30,
        minimum_probability_uplift=0.0,
        minimum_brier_improvement=0.0,
        minimum_accuracy_uplift=0.0,
    )

    assert result.passed is True


def test_fails_when_observations_are_below_minimum() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1] * 5,
        baseline_probabilities=[0.6, 0.4] * 5,
        outcomes=[1, 0] * 5,
        minimum_observations=30,
    )

    assert result.passed is False


def test_fails_when_probability_uplift_is_below_threshold() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.55, 0.45] * 20,
        baseline_probabilities=[0.50, 0.40] * 20,
        outcomes=[1, 0] * 20,
        minimum_observations=30,
        minimum_probability_uplift=0.10,
    )

    assert result.passed is False


def test_fails_when_brier_improvement_is_below_threshold() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.6, 0.4] * 20,
        baseline_probabilities=[0.5, 0.5] * 20,
        outcomes=[1, 0] * 20,
        minimum_observations=30,
        minimum_brier_improvement=0.20,
    )

    assert result.passed is False


def test_fails_when_accuracy_uplift_is_below_threshold() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.6, 0.4] * 20,
        baseline_probabilities=[0.5, 0.5] * 20,
        outcomes=[1, 0] * 20,
        minimum_observations=30,
        minimum_accuracy_uplift=0.10,
    )

    assert result.passed is False


def test_thresholds_are_inclusive() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[1.0] * 30,
        baseline_probabilities=[0.5] * 30,
        outcomes=[1] * 30,
        minimum_observations=30,
        minimum_probability_uplift=0.5,
        minimum_brier_improvement=0.25,
        minimum_accuracy_uplift=0.0,
    )

    assert result.passed is True


def test_result_statistics_are_finite() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.8, 0.2],
        baseline_probabilities=[0.6, 0.4],
        outcomes=[1, 0],
    )

    values = [
        result.candidate_mean_probability,
        result.baseline_mean_probability,
        result.mean_probability_uplift,
        result.candidate_brier_score,
        result.baseline_brier_score,
        result.brier_score_improvement,
        result.candidate_accuracy,
        result.baseline_accuracy,
        result.accuracy_uplift,
    ]

    assert all(math.isfinite(value) for value in values)


def test_validation_is_deterministic() -> None:
    candidate = [0.8, 0.2] * 10
    baseline = [0.6, 0.4] * 10
    outcomes = [1, 0] * 10

    first = evaluate_advanced_model(
        candidate_probabilities=candidate,
        baseline_probabilities=baseline,
        outcomes=outcomes,
    )

    second = evaluate_advanced_model(
        candidate_probabilities=candidate,
        baseline_probabilities=baseline,
        outcomes=outcomes,
    )

    assert first == second


def test_candidate_and_baseline_must_have_equal_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        evaluate_advanced_model(
            candidate_probabilities=[0.8, 0.2],
            baseline_probabilities=[0.6],
            outcomes=[1, 0],
        )


def test_probabilities_and_outcomes_must_have_equal_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        evaluate_advanced_model(
            candidate_probabilities=[0.8, 0.2],
            baseline_probabilities=[0.6, 0.4],
            outcomes=[1],
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [],
        [True],
        ["bad"],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [-0.1],
        [1.1],
    ],
)
def test_invalid_candidate_probabilities_are_rejected(
    bad_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        evaluate_advanced_model(
            candidate_probabilities=bad_values,
            baseline_probabilities=[0.5] * len(bad_values),
            outcomes=[1] * len(bad_values),
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [],
        [True],
        ["bad"],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [-0.1],
        [1.1],
    ],
)
def test_invalid_baseline_probabilities_are_rejected(
    bad_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        evaluate_advanced_model(
            candidate_probabilities=[0.5] * len(bad_values),
            baseline_probabilities=bad_values,
            outcomes=[1] * len(bad_values),
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [],
        [True],
        [2],
        [-1],
        [0.5],
        ["bad"],
    ],
)
def test_invalid_outcomes_are_rejected(
    bad_values: list[int],
) -> None:
    with pytest.raises(ValueError):
        evaluate_advanced_model(
            candidate_probabilities=[0.5] * max(1, len(bad_values)),
            baseline_probabilities=[0.5] * max(1, len(bad_values)),
            outcomes=bad_values,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_minimum_observations_are_rejected(
    value: int,
) -> None:
    with pytest.raises(ValueError):
        evaluate_advanced_model(
            candidate_probabilities=[0.5],
            baseline_probabilities=[0.5],
            outcomes=[1],
            minimum_observations=value,
        )


@pytest.mark.parametrize(
    "parameter_name",
    [
        "minimum_probability_uplift",
        "minimum_brier_improvement",
        "minimum_accuracy_uplift",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "bad",
    ],
)
def test_invalid_thresholds_are_rejected(
    parameter_name: str,
    value: float,
) -> None:
    kwargs = {
        "minimum_probability_uplift": 0.0,
        "minimum_brier_improvement": 0.0,
        "minimum_accuracy_uplift": 0.0,
    }
    kwargs[parameter_name] = value

    with pytest.raises(ValueError):
        evaluate_advanced_model(
            candidate_probabilities=[0.5],
            baseline_probabilities=[0.5],
            outcomes=[1],
            **kwargs,
        )


def test_candidate_can_have_worse_brier_score() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.51, 0.49],
        baseline_probabilities=[0.99, 0.01],
        outcomes=[1, 0],
    )

    assert result.brier_score_improvement < 0.0


def test_candidate_can_have_worse_accuracy() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.4, 0.6],
        baseline_probabilities=[0.9, 0.1],
        outcomes=[1, 0],
    )

    assert result.accuracy_uplift < 0.0


def test_brier_score_improvement_uses_lower_is_better_semantics() -> None:
    result = evaluate_advanced_model(
        candidate_probabilities=[0.9, 0.1],
        baseline_probabilities=[0.5, 0.5],
        outcomes=[1, 0],
    )

    assert result.candidate_brier_score < result.baseline_brier_score
    assert result.brier_score_improvement > 0.0
