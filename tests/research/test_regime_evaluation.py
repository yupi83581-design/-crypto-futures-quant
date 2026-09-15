"""Tests for regime forward-return evaluation."""

import math

import pytest

from src.research.regime import RegimeConfig
from src.research.regime_evaluation import (
    RegimeEvaluationResult,
    RegimePerformance,
    evaluate_regime_impact,
)


def test_returns_evaluation_result() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    assert isinstance(result, RegimeEvaluationResult)


def test_total_observations_matches_evaluation_windows() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    assert result.total_observations == 6


def test_overall_statistics_are_finite() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    assert math.isfinite(
        result.overall_mean_forward_return
    )
    assert math.isfinite(
        result.overall_positive_rate
    )


def test_overall_positive_rate_is_between_zero_and_one() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    assert 0.0 <= result.overall_positive_rate <= 1.0


def test_regime_performance_objects_are_returned() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    assert isinstance(
        result.trend_up,
        RegimePerformance,
    )
    assert isinstance(
        result.trend_down,
        RegimePerformance,
    )
    assert isinstance(
        result.high_volatility,
        RegimePerformance,
    )
    assert isinstance(
        result.range,
        RegimePerformance,
    )


def test_regime_observation_counts_sum_to_total() -> None:
    prices = [
        100.0 + index
        for index in range(30)
    ]

    result = evaluate_regime_impact(prices)

    regime_count = (
        result.trend_up.observations
        + result.trend_down.observations
        + result.high_volatility.observations
        + result.range.observations
    )

    assert regime_count == result.total_observations


def test_positive_rate_is_valid_for_each_regime() -> None:
    prices = [
        100.0 + index
        for index in range(30)
    ]

    result = evaluate_regime_impact(prices)

    performances = [
        result.trend_up,
        result.trend_down,
        result.high_volatility,
        result.range,
    ]

    for performance in performances:
        assert 0.0 <= performance.positive_rate <= 1.0


def test_empty_regime_has_zero_statistics() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = evaluate_regime_impact(prices)

    performances = [
        result.trend_down,
        result.high_volatility,
        result.range,
    ]

    for performance in performances:
        if performance.observations == 0:
            assert performance.mean_forward_return == 0.0
            assert performance.positive_rate == 0.0


def test_constant_prices_produce_zero_forward_returns() -> None:
    prices = [100.0] * 25

    result = evaluate_regime_impact(prices)

    assert result.overall_mean_forward_return == 0.0
    assert result.overall_positive_rate == 0.0
    assert result.range.observations == 6
    assert result.range.mean_forward_return == 0.0
    assert result.range.positive_rate == 0.0


def test_monotonic_up_prices_have_positive_overall_forward_return() -> None:
    prices = [
        100.0 + (index * 2.0)
        for index in range(30)
    ]

    result = evaluate_regime_impact(prices)

    assert result.overall_mean_forward_return > 0.0
    assert result.overall_positive_rate == 1.0
    assert result.trend_up.observations > 0


def test_monotonic_down_prices_have_negative_overall_forward_return() -> None:
    prices = [
        200.0 - (index * 2.0)
        for index in range(30)
    ]

    result = evaluate_regime_impact(prices)

    assert result.overall_mean_forward_return < 0.0
    assert result.overall_positive_rate == 0.0
    assert result.trend_down.observations > 0


def test_custom_minimum_observations_changes_window_count() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    config = RegimeConfig(
        minimum_observations=10,
    )

    result = evaluate_regime_impact(
        prices,
        config=config,
    )

    assert result.total_observations == 16


def test_rejects_insufficient_prices_for_evaluation() -> None:
    prices = [100.0] * 20

    with pytest.raises(
        ValueError,
        match="insufficient prices for forward-return evaluation",
    ):
        evaluate_regime_impact(prices)


def test_rejects_too_few_prices() -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain at least 2 observations",
    ):
        evaluate_regime_impact([100.0])


def test_rejects_non_numeric_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="prices must be a numeric sequence",
    ):
        evaluate_regime_impact(
            "100,101,102"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, 0.0] + [100.0] * 19,
        [100.0, -1.0] + [100.0] * 19,
    ],
)
def test_rejects_non_positive_prices(
    prices: list[float],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must be positive",
    ):
        evaluate_regime_impact(prices)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, float("nan")] + [100.0] * 19,
        [100.0, float("inf")] + [100.0] * 19,
        [100.0, float("-inf")] + [100.0] * 19,
    ],
)
def test_rejects_non_finite_prices(
    prices: list[float],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain only finite values",
    ):
        evaluate_regime_impact(prices)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, True] + [100.0] * 19,
        [100.0, "101"] + [100.0] * 19,  # type: ignore[list-item]
    ],
)
def test_rejects_non_numeric_values(
    prices: list[object],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain only numeric values",
    ):
        evaluate_regime_impact(
            prices  # type: ignore[arg-type]
        )


def test_evaluation_is_deterministic() -> None:
    prices = [
        100.0,
        101.0,
        100.5,
        102.0,
        103.0,
        102.5,
        104.0,
        105.0,
        106.0,
        105.5,
        107.0,
        108.0,
        109.0,
        108.5,
        110.0,
        111.0,
        112.0,
        111.5,
        113.0,
        114.0,
        115.0,
        114.5,
        116.0,
        117.0,
        118.0,
    ]

    first = evaluate_regime_impact(prices)
    second = evaluate_regime_impact(prices)

    assert first == second


def test_price_scale_does_not_change_regime_distribution() -> None:
    prices = [
        100.0 + (index * 1.5)
        for index in range(30)
    ]

    scaled_prices = [
        price * 1000.0
        for price in prices
    ]

    first = evaluate_regime_impact(prices)
    second = evaluate_regime_impact(scaled_prices)

    assert first.total_observations == second.total_observations

    assert (
        first.trend_up.observations
        == second.trend_up.observations
    )
    assert (
        first.trend_down.observations
        == second.trend_down.observations
    )
    assert (
        first.high_volatility.observations
        == second.high_volatility.observations
    )
    assert (
        first.range.observations
        == second.range.observations
    )


def test_forward_return_is_based_on_next_price() -> None:
    prices = [
        100.0 + index
        for index in range(21)
    ]

    config = RegimeConfig(
        minimum_observations=20,
        trend_threshold=0.0,
        volatility_threshold=1.0,
    )

    result = evaluate_regime_impact(
        prices,
        config=config,
    )

    expected_return = (
        prices[-1] / prices[-2]
    ) - 1.0

    assert result.total_observations == 1
    assert math.isclose(
        result.overall_mean_forward_return,
        expected_return,
    )


def test_forward_outcome_is_not_counted_as_historical_price() -> None:
    historical = [
        100.0 + index
        for index in range(20)
    ]

    prices_a = historical + [121.0]
    prices_b = historical + [100.0]

    config = RegimeConfig(
        minimum_observations=20,
        trend_threshold=0.0,
        volatility_threshold=1.0,
    )

    result_a = evaluate_regime_impact(
        prices_a,
        config=config,
    )
    result_b = evaluate_regime_impact(
        prices_b,
        config=config,
    )

    assert result_a.total_observations == 1
    assert result_b.total_observations == 1

    assert result_a.trend_up.observations == 1
    assert result_b.trend_up.observations == 1
