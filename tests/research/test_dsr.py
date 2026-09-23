import math
import pytest
from src.research.dsr import deflated_sharpe_ratio

def test_normal_returns_have_finite_dsr():
    result = deflated_sharpe_ratio([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, -0.004, 0.009], trial_count=7)
    assert result.sample_size == 8
    assert result.trial_count == 7
    assert math.isfinite(result.observed_sharpe)
    assert math.isfinite(result.expected_max_sharpe)
    assert 0.0 <= result.deflated_sharpe_probability <= 1.0

def test_trial_count_changes_expected_maximum():
    a = deflated_sharpe_ratio([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, -0.004, 0.009], trial_count=2)
    b = deflated_sharpe_ratio([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, -0.004, 0.009], trial_count=20)
    assert b.expected_max_sharpe > a.expected_max_sharpe
    assert b.deflated_sharpe_probability <= a.deflated_sharpe_probability

def test_rejects_invalid_trial_count():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.1, 0.2, 0.3], trial_count=0)

def test_rejects_insufficient_sample():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.1], trial_count=1)

def test_deterministic():
    values=[0.01, -0.01, 0.02, 0.0, 0.005, -0.004]
    assert deflated_sharpe_ratio(values, trial_count=4) == deflated_sharpe_ratio(values, trial_count=4)


def test_non_normality_statistics_are_exposed():
    values = [-0.03, -0.01, 0.0, 0.01, 0.02, 0.08, 0.01, -0.005]
    result = deflated_sharpe_ratio(values, trial_count=5)
    assert math.isfinite(result.skewness)
    assert math.isfinite(result.kurtosis)
    assert result.kurtosis >= 1.0


def test_requires_enough_observations_for_shape_statistics():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.1, -0.1], trial_count=2)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.1, -0.1, 0.05], trial_count=2)


def test_explicit_trial_count_is_part_of_result():
    result = deflated_sharpe_ratio([0.01, -0.01, 0.02, 0.0, 0.005], trial_count=11)
    assert result.trial_count == 11
