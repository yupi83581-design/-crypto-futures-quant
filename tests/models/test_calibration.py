"""Tests for probability calibration metrics."""

import math

import pytest

from src.models.calibration import (
    CalibrationBucket,
    CalibrationResult,
    evaluate_calibration,
)


def test_perfect_probabilities_have_zero_brier_score():
    result = evaluate_calibration(
        actual=[0, 1, 0, 1],
        probabilities=[0.0, 1.0, 0.0, 1.0],
    )

    assert result.brier_score == pytest.approx(0.0)


def test_brier_score_is_mean_squared_probability_error():
    result = evaluate_calibration(
        actual=[1, 0],
        probabilities=[0.8, 0.2],
        bucket_count=2,
    )

    expected = (
        (0.8 - 1.0) ** 2
        + (0.2 - 0.0) ** 2
    ) / 2

    assert result.brier_score == pytest.approx(expected)


def test_brier_score_is_finite_and_non_negative():
    result = evaluate_calibration(
        actual=[0, 1, 1, 0],
        probabilities=[0.1, 0.9, 0.7, 0.2],
    )

    assert result.brier_score >= 0.0
    assert math.isfinite(result.brier_score)


def test_calibration_bucket_reports_statistics():
    result = evaluate_calibration(
        actual=[1, 1, 0, 0],
        probabilities=[0.8, 0.9, 0.1, 0.2],
        bucket_count=2,
    )

    assert len(result.buckets) == 2

    low = result.buckets[0]
    high = result.buckets[1]

    assert low.lower_bound == pytest.approx(0.0)
    assert low.upper_bound == pytest.approx(0.5)
    assert low.predicted_probability == pytest.approx(0.15)
    assert low.observed_frequency == pytest.approx(0.0)
    assert low.sample_count == 2

    assert high.lower_bound == pytest.approx(0.5)
    assert high.upper_bound == pytest.approx(1.0)
    assert high.predicted_probability == pytest.approx(0.85)
    assert high.observed_frequency == pytest.approx(1.0)
    assert high.sample_count == 2


def test_boundary_probability_belongs_to_upper_bucket():
    result = evaluate_calibration(
        actual=[0, 1, 0],
        probabilities=[0.25, 0.5, 1.0],
        bucket_count=2,
    )

    assert len(result.buckets) == 2

    low = result.buckets[0]
    high = result.buckets[1]

    assert low.lower_bound == pytest.approx(0.0)
    assert low.upper_bound == pytest.approx(0.5)
    assert low.predicted_probability == pytest.approx(0.25)
    assert low.observed_frequency == pytest.approx(0.0)
    assert low.sample_count == 1

    assert high.lower_bound == pytest.approx(0.5)
    assert high.upper_bound == pytest.approx(1.0)
    assert high.predicted_probability == pytest.approx(0.75)
    assert high.observed_frequency == pytest.approx(0.5)
    assert high.sample_count == 2


def test_probability_zero_belongs_to_first_bucket():
    result = evaluate_calibration(
        actual=[0],
        probabilities=[0.0],
        bucket_count=10,
    )

    assert len(result.buckets) == 1

    bucket = result.buckets[0]

    assert bucket.lower_bound == pytest.approx(0.0)
    assert bucket.upper_bound == pytest.approx(0.1)
    assert bucket.sample_count == 1


def test_probability_one_belongs_to_last_bucket():
    result = evaluate_calibration(
        actual=[1],
        probabilities=[1.0],
        bucket_count=10,
    )

    assert len(result.buckets) == 1

    bucket = result.buckets[0]

    assert bucket.lower_bound == pytest.approx(0.9)
    assert bucket.upper_bound == pytest.approx(1.0)
    assert bucket.sample_count == 1


def test_empty_inputs_return_empty_result():
    result = evaluate_calibration(
        actual=[],
        probabilities=[],
    )

    assert isinstance(result, CalibrationResult)
    assert result.brier_score == pytest.approx(0.0)
    assert result.buckets == ()


def test_result_buckets_are_tuple():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.2, 0.8],
    )

    assert isinstance(result.buckets, tuple)


def test_bucket_is_frozen_dataclass():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.2, 0.8],
    )

    bucket = result.buckets[0]

    assert isinstance(bucket, CalibrationBucket)

    with pytest.raises(AttributeError):
        bucket.sample_count = 99


def test_sample_counts_sum_to_total_observations():
    actual = [0, 1, 0, 1, 1, 0]
    probabilities = [0.05, 0.15, 0.35, 0.55, 0.75, 0.95]

    result = evaluate_calibration(
        actual=actual,
        probabilities=probabilities,
        bucket_count=5,
    )

    total_samples = sum(
        bucket.sample_count
        for bucket in result.buckets
    )

    assert total_samples == len(actual)


def test_bucket_predicted_probability_is_mean():
    result = evaluate_calibration(
        actual=[0, 1, 1],
        probabilities=[0.1, 0.2, 0.3],
        bucket_count=2,
    )

    bucket = result.buckets[0]

    assert bucket.predicted_probability == pytest.approx(0.2)


def test_bucket_observed_frequency_is_mean_label():
    result = evaluate_calibration(
        actual=[0, 1, 1],
        probabilities=[0.1, 0.2, 0.3],
        bucket_count=2,
    )

    bucket = result.buckets[0]

    assert bucket.observed_frequency == pytest.approx(2 / 3)


def test_empty_buckets_are_not_returned():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.1, 0.9],
        bucket_count=10,
    )

    assert len(result.buckets) == 2

    assert all(
        bucket.sample_count > 0
        for bucket in result.buckets
    )


def test_custom_bucket_count_changes_bucket_ranges():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.2, 0.8],
        bucket_count=4,
    )

    assert len(result.buckets) == 2

    assert result.buckets[0].lower_bound == pytest.approx(0.0)
    assert result.buckets[0].upper_bound == pytest.approx(0.25)

    assert result.buckets[1].lower_bound == pytest.approx(0.75)
    assert result.buckets[1].upper_bound == pytest.approx(1.0)


def test_same_inputs_produce_same_result():
    actual = [1, 0, 1, 0, 1]
    probabilities = [0.7, 0.2, 0.8, 0.4, 0.9]

    first = evaluate_calibration(
        actual,
        probabilities,
    )

    second = evaluate_calibration(
        actual,
        probabilities,
    )

    assert first == second


def test_actual_and_probability_lengths_must_match():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_calibration(
            actual=[0, 1],
            probabilities=[0.5],
        )


@pytest.mark.parametrize(
    "bucket_count",
    [0, -1, 1.5, True],
)
def test_bucket_count_must_be_positive_integer(
    bucket_count,
):
    with pytest.raises(
        ValueError,
        match="bucket_count",
    ):
        evaluate_calibration(
            actual=[0],
            probabilities=[0.5],
            bucket_count=bucket_count,
        )


@pytest.mark.parametrize(
    "actual",
    [
        [2],
        [-1],
        [0, 2],
        [True],
        [False],
    ],
)
def test_actual_values_must_be_binary(
    actual,
):
    with pytest.raises(
        ValueError,
        match="binary",
    ):
        evaluate_calibration(
            actual=actual,
            probabilities=[0.5] * len(actual),
        )


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_probability_must_be_valid(
    probability,
):
    with pytest.raises(
        ValueError,
        match="probability",
    ):
        evaluate_calibration(
            actual=[0],
            probabilities=[probability],
        )


def test_probability_must_be_numeric():
    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        evaluate_calibration(
            actual=[0],
            probabilities=["0.5"],
        )


def test_multiple_observations_in_same_bucket_are_aggregated():
    result = evaluate_calibration(
        actual=[0, 1, 1, 0],
        probabilities=[0.21, 0.22, 0.23, 0.24],
        bucket_count=2,
    )

    assert len(result.buckets) == 1

    bucket = result.buckets[0]

    assert bucket.sample_count == 4
    assert bucket.predicted_probability == pytest.approx(0.225)
    assert bucket.observed_frequency == pytest.approx(0.5)


def test_probability_one_is_included_even_when_bucket_count_is_one():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.0, 1.0],
        bucket_count=1,
    )

    assert len(result.buckets) == 1

    bucket = result.buckets[0]

    assert bucket.lower_bound == pytest.approx(0.0)
    assert bucket.upper_bound == pytest.approx(1.0)
    assert bucket.predicted_probability == pytest.approx(0.5)
    assert bucket.observed_frequency == pytest.approx(0.5)
    assert bucket.sample_count == 2


def test_all_probabilities_can_share_final_bucket():
    result = evaluate_calibration(
        actual=[0, 1, 1],
        probabilities=[0.8, 0.9, 1.0],
        bucket_count=2,
    )

    assert len(result.buckets) == 1

    bucket = result.buckets[0]

    assert bucket.lower_bound == pytest.approx(0.5)
    assert bucket.upper_bound == pytest.approx(1.0)
    assert bucket.sample_count == 3


def test_calibration_does_not_modify_inputs():
    actual = [0, 1, 1]
    probabilities = [0.2, 0.7, 0.9]

    original_actual = actual.copy()
    original_probabilities = probabilities.copy()

    evaluate_calibration(
        actual,
        probabilities,
    )

    assert actual == original_actual
    assert probabilities == original_probabilities
