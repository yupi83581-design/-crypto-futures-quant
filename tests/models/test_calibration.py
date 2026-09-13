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

    assert result.brier_score == 0.0


def test_brier_score_is_mean_squared_probability_error():
    result = evaluate_calibration(
        actual=[1, 0],
        probabilities=[0.8, 0.2],
        bucket_count=2,
    )

    expected = ((0.8 - 1) ** 2 + (0.2 - 0) ** 2) / 2

    assert result.brier_score == pytest.approx(expected)


def test_calibration_bucket_reports_predicted_and_observed_frequency():
    result = evaluate_calibration(
        actual=[1, 1, 0, 0],
        probabilities=[0.8, 0.9, 0.1, 0.2],
        bucket_count=2,
    )

    assert len(result.buckets) == 2

    low = result.buckets[0]
    high = result.buckets[1]

    assert low.lower_bound == 0.0
    assert low.upper_bound == 0.5
    assert low.predicted_probability == pytest.approx(0.15)
    assert low.observed_frequency == pytest.approx(0.0)
    assert low.sample_count == 2

    assert high.lower_bound == 0.5
    assert high.upper_bound == 1.0
    assert high.predicted_probability == pytest.approx(0.85)
    assert high.observed_frequency == pytest.approx(1.0)
    assert high.sample_count == 2


def test_probability_equal_to_bucket_boundary_belongs_to_upper_bucket():
    result = evaluate_calibration(
        actual=[0, 1, 0],
        probabilities=[0.25, 0.5, 1.0],
        bucket_count=2,
    )

    assert len(result.buckets) == 2

    low = result.buckets[0]
    high = result.buckets[1]

    assert low.predicted_probability == pytest.approx(0.25)
    assert low.observed_frequency == pytest.approx(0.0)
    assert low.sample_count == 1

    assert high.predicted_probability == pytest.approx(0.75)
    assert high.observed_frequency == pytest.approx(0.5)
    assert high.sample_count == 2
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.5, 1.0],
        bucket_count=2,
    )

    assert len(result.buckets) == 2

    assert result.buckets[0].predicted_probability == pytest.approx(0.5)
    assert result.buckets[0].observed_frequency == pytest.approx(0.0)

    assert result.buckets[1].predicted_probability == pytest.approx(1.0)
    assert result.buckets[1].observed_frequency == pytest.approx(1.0)


def test_probability_zero_is_in_first_bucket():
    result = evaluate_calibration(
        actual=[0],
        probabilities=[0.0],
        bucket_count=10,
    )

    assert len(result.buckets) == 1
    assert result.buckets[0].lower_bound == 0.0
    assert result.buckets[0].sample_count == 1


def test_probability_one_is_in_last_bucket():
    result = evaluate_calibration(
        actual=[1],
        probabilities=[1.0],
        bucket_count=10,
    )

    assert len(result.buckets) == 1
    assert result.buckets[0].upper_bound == 1.0
    assert result.buckets[0].sample_count == 1


def test_empty_inputs_return_empty_result():
    result = evaluate_calibration(
        actual=[],
        probabilities=[],
    )

    assert isinstance(result, CalibrationResult)
    assert result.brier_score == 0.0
    assert result.buckets == ()


def test_output_buckets_are_immutable():
    result = evaluate_calibration(
        actual=[0, 1],
        probabilities=[0.2, 0.8],
    )

    assert isinstance(result.buckets, tuple)
    assert isinstance(result.buckets[0], CalibrationBucket)


def test_actual_and_probability_length_must_match():
    with pytest.raises(ValueError, match="same length"):
        evaluate_calibration(
            actual=[0, 1],
            probabilities=[0.5],
        )


@pytest.mark.parametrize("bucket_count", [0, -1, 1.5, True])
def test_bucket_count_must_be_positive_integer(bucket_count):
    with pytest.raises(ValueError, match="bucket_count"):
        evaluate_calibration(
            actual=[0],
            probabilities=[0.5],
            bucket_count=bucket_count,
        )


@pytest.mark.parametrize("actual", [[2], [-1], [0, 2], [True], [False]])
def test_actual_values_must_be_binary_integers(actual):
    with pytest.raises(ValueError, match="binary"):
        evaluate_calibration(
            actual=actual,
            probabilities=[0.5] * len(actual),
        )


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True],
)
def test_probability_must_be_finite_and_between_zero_and_one(probability):
    with pytest.raises(ValueError, match="probability"):
        evaluate_calibration(
            actual=[0],
            probabilities=[probability],
        )


def test_probability_must_be_numeric():
    with pytest.raises(ValueError, match="numeric"):
        evaluate_calibration(
            actual=[0],
            probabilities=["0.5"],
        )


def test_brier_score_is_always_non_negative_and_finite():
    result = evaluate_calibration(
        actual=[0, 1, 1, 0],
        probabilities=[0.1, 0.9, 0.7, 0.2],
    )

    assert result.brier_score >= 0.0
    assert math.isfinite(result.brier_score)


def test_bucket_sample_counts_sum_to_observation_count():
    actual = [0, 1, 0, 1, 1, 0]
    probabilities = [0.05, 0.15, 0.35, 0.55, 0.75, 0.95]

    result = evaluate_calibration(
        actual=actual,
        probabilities=probabilities,
        bucket_count=5,
    )

    assert sum(bucket.sample_count for bucket in result.buckets) == len(actual)


def test_bucket_predicted_probability_is_mean_of_members():
    result = evaluate_calibration(
        actual=[0, 1, 1],
        probabilities=[0.1, 0.2, 0.3],
        bucket_count=2,
    )

    assert result.buckets[0].predicted_probability == pytest.approx(0.2)
    assert result.buckets[0].observed_frequency == pytest.approx(2 / 3)


def test_calibration_result_is_deterministic():
    actual = [1, 0, 1, 0, 1]
    probabilities = [0.7, 0.2, 0.8, 0.4, 0.9]

    first = evaluate_calibration(actual, probabilities)
    second = evaluate_calibration(actual, probabilities)

    assert first == second
