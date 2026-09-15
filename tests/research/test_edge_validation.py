"""Tests for deterministic research edge validation."""

from __future__ import annotations

import math

import pytest

from src.research.edge_validation import (
    EdgeValidationResult,
    validate_edge,
)


def _candidate_returns() -> list[float]:
    return [
        0.010,
        0.020,
        -0.005,
        0.015,
        0.010,
        0.020,
        -0.005,
        0.015,
        0.010,
        0.020,
    ]


def _baseline_returns() -> list[float]:
    return [
        0.005,
        0.010,
        -0.010,
        0.005,
        0.005,
        0.010,
        -0.010,
        0.005,
        0.005,
        0.010,
    ]


def test_returns_expected_result_type() -> None:
    result = validate_edge(
        candidate_returns=_candidate_returns(),
        baseline_returns=_baseline_returns(),
    )

    assert isinstance(result, EdgeValidationResult)


def test_candidate_mean_return_is_calculated() -> None:
    result = validate_edge(
        candidate_returns=[0.10, 0.20],
        baseline_returns=[0.05, 0.05],
    )

    assert result.candidate_mean_return == pytest.approx(0.15)


def test_baseline_mean_return_is_calculated() -> None:
    result = validate_edge(
        candidate_returns=[0.10, 0.20],
        baseline_returns=[0.05, 0.15],
    )

    assert result.baseline_mean_return == pytest.approx(0.10)


def test_mean_return_uplift_is_candidate_minus_baseline() -> None:
    result = validate_edge(
        candidate_returns=[0.10, 0.20],
        baseline_returns=[0.05, 0.10],
    )

    assert result.mean_return_uplift == pytest.approx(0.075)


def test_positive_rate_is_calculated() -> None:
    result = validate_edge(
        candidate_returns=[0.10, -0.10, 0.20, -0.20],
        baseline_returns=[0.10, 0.10, -0.10, -0.10],
    )

    assert result.candidate_positive_rate == pytest.approx(0.50)
    assert result.baseline_positive_rate == pytest.approx(0.50)


def test_positive_rate_uplift_is_calculated() -> None:
    result = validate_edge(
        candidate_returns=[0.10, 0.20, -0.10, -0.20],
        baseline_returns=[0.10, -0.10, -0.20, -0.30],
    )

    assert result.positive_rate_uplift == pytest.approx(0.25)


def test_passes_when_all_thresholds_are_met() -> None:
    candidate = [0.10] * 10
    baseline = [0.05] * 10

    result = validate_edge(
        candidate_returns=candidate,
        baseline_returns=baseline,
        minimum_observations=10,
        minimum_mean_uplift=0.05,
        minimum_positive_rate_uplift=0.0,
    )

    assert result.passed is True


def test_fails_when_observation_count_is_too_small() -> None:
    result = validate_edge(
        candidate_returns=[0.10] * 5,
        baseline_returns=[0.05] * 5,
        minimum_observations=10,
    )

    assert result.passed is False


def test_fails_when_mean_uplift_is_below_threshold() -> None:
    result = validate_edge(
        candidate_returns=[0.06] * 10,
        baseline_returns=[0.05] * 10,
        minimum_observations=10,
        minimum_mean_uplift=0.02,
    )

    assert result.passed is False


def test_fails_when_positive_rate_uplift_is_below_threshold() -> None:
    result = validate_edge(
        candidate_returns=[0.10, -0.10] * 5,
        baseline_returns=[0.10, -0.10] * 5,
        minimum_observations=10,
        minimum_positive_rate_uplift=0.01,
    )

    assert result.passed is False


def test_thresholds_are_inclusive() -> None:
    result = validate_edge(
        candidate_returns=[0.10] * 10,
        baseline_returns=[0.05] * 10,
        minimum_observations=10,
        minimum_mean_uplift=0.05,
        minimum_positive_rate_uplift=0.0,
    )

    assert result.passed is True


def test_zero_return_is_not_positive() -> None:
    result = validate_edge(
        candidate_returns=[0.0, 0.10, -0.10],
        baseline_returns=[0.0, 0.10, -0.10],
    )

    assert result.candidate_positive_rate == pytest.approx(1 / 3)
    assert result.baseline_positive_rate == pytest.approx(1 / 3)


def test_candidate_and_baseline_are_allowed_to_have_different_lengths() -> None:
    result = validate_edge(
        candidate_returns=[0.10] * 10,
        baseline_returns=[0.05] * 20,
        minimum_observations=10,
    )

    assert result.observations == 10
    assert result.passed is True


def test_result_contains_configured_thresholds() -> None:
    result = validate_edge(
        candidate_returns=[0.10] * 10,
        baseline_returns=[0.05] * 10,
        minimum_observations=10,
        minimum_mean_uplift=0.02,
        minimum_positive_rate_uplift=0.01,
    )

    assert result.minimum_observations == 10
    assert result.minimum_mean_uplift == pytest.approx(0.02)
    assert result.minimum_positive_rate_uplift == pytest.approx(
        0.01
    )


def test_result_statistics_are_finite() -> None:
    result = validate_edge(
        candidate_returns=_candidate_returns(),
        baseline_returns=_baseline_returns(),
    )

    values = [
        result.candidate_mean_return,
        result.baseline_mean_return,
        result.mean_return_uplift,
        result.candidate_positive_rate,
        result.baseline_positive_rate,
        result.positive_rate_uplift,
    ]

    assert all(math.isfinite(value) for value in values)


@pytest.mark.parametrize(
    "bad_values",
    [
        [],
        [True],
        ["bad"],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
    ],
)
def test_invalid_candidate_returns_are_rejected(
    bad_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_edge(
            candidate_returns=bad_values,
            baseline_returns=[0.01],
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
    ],
)
def test_invalid_baseline_returns_are_rejected(
    bad_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_edge(
            candidate_returns=[0.01],
            baseline_returns=bad_values,
        )


@pytest.mark.parametrize(
    "minimum_observations",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_minimum_observations_are_rejected(
    minimum_observations: int,
) -> None:
    with pytest.raises(ValueError):
        validate_edge(
            candidate_returns=[0.01],
            baseline_returns=[0.01],
            minimum_observations=minimum_observations,
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
def test_invalid_mean_uplift_threshold_is_rejected(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        validate_edge(
            candidate_returns=[0.01],
            baseline_returns=[0.01],
            minimum_mean_uplift=value,
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
def test_invalid_positive_rate_threshold_is_rejected(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        validate_edge(
            candidate_returns=[0.01],
            baseline_returns=[0.01],
            minimum_positive_rate_uplift=value,
        )


def test_negative_mean_uplift_can_fail_validation() -> None:
    result = validate_edge(
        candidate_returns=[0.01] * 10,
        baseline_returns=[0.02] * 10,
        minimum_observations=10,
    )

    assert result.mean_return_uplift == pytest.approx(-0.01)
    assert result.passed is False


def test_negative_positive_rate_uplift_can_fail_validation() -> None:
    result = validate_edge(
        candidate_returns=[0.10, -0.10] * 5,
        baseline_returns=[0.10] * 10,
        minimum_observations=10,
    )

    assert result.positive_rate_uplift == pytest.approx(-0.50)
    assert result.passed is False


def test_validation_is_deterministic() -> None:
    candidate = _candidate_returns()
    baseline = _baseline_returns()

    first = validate_edge(
        candidate_returns=candidate,
        baseline_returns=baseline,
    )

    second = validate_edge(
        candidate_returns=candidate,
        baseline_returns=baseline,
    )

    assert first == second
