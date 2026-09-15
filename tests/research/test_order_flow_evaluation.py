"""Tests for order-flow forward-return evaluation."""

import math

import pytest

from src.research.order_flow_evaluation import (
    OrderFlowEvaluationResult,
    OrderFlowPerformance,
    evaluate_order_flow_impact,
)


def test_returns_evaluation_result() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 60.0, 80.0],
        sell_volumes=[30.0, 40.0, 20.0],
        prices=[100.0, 101.0, 102.0],
    )

    assert isinstance(result, OrderFlowEvaluationResult)


def test_total_observations_equals_price_count_minus_one() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]

    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 60.0, 80.0, 75.0, 65.0],
        sell_volumes=[30.0, 40.0, 20.0, 25.0, 35.0],
        prices=prices,
    )

    assert result.total_observations == len(prices) - 1
    assert result.total_observations == 4


def test_regime_bucket_counts_sum_to_total() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[
            90.0,
            50.0,
            10.0,
            70.0,
            50.0,
        ],
        sell_volumes=[
            10.0,
            50.0,
            90.0,
            30.0,
            50.0,
        ],
        prices=[
            100.0,
            101.0,
            100.0,
            99.0,
            100.0,
        ],
    )

    bucket_count = (
        result.buy_dominant.observations
        + result.balanced.observations
        + result.sell_dominant.observations
    )

    assert bucket_count == result.total_observations


def test_returns_are_aligned_to_next_price() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 70.0],
        sell_volumes=[30.0, 30.0],
        prices=[100.0, 110.0],
    )

    assert result.total_observations == 1
    assert math.isclose(
        result.overall_mean_forward_return,
        0.10,
    )


def test_multiple_forward_returns_have_correct_mean() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 70.0, 30.0],
        sell_volumes=[30.0, 30.0, 70.0],
        prices=[100.0, 110.0, 99.0],
    )

    expected_mean = (
        ((110.0 / 100.0) - 1.0)
        + ((99.0 / 110.0) - 1.0)
    ) / 2.0

    assert math.isclose(
        result.overall_mean_forward_return,
        expected_mean,
    )


def test_positive_rate_counts_only_positive_returns() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 70.0, 70.0, 70.0],
        sell_volumes=[30.0, 30.0, 30.0, 30.0],
        prices=[100.0, 110.0, 100.0, 100.0],
    )

    assert result.total_observations == 3
    assert math.isclose(
        result.overall_positive_rate,
        1.0 / 3.0,
    )


def test_zero_forward_return_is_not_positive() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 70.0],
        sell_volumes=[30.0, 30.0],
        prices=[100.0, 100.0],
    )

    assert result.overall_mean_forward_return == 0.0
    assert result.overall_positive_rate == 0.0


def test_buy_dominant_bucket_uses_positive_threshold() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[60.0],
        sell_volumes=[40.0],
        prices=[100.0, 101.0],
        imbalance_threshold=0.20,
    )

    assert result.buy_dominant.observations == 1
    assert result.balanced.observations == 0
    assert result.sell_dominant.observations == 0


def test_sell_dominant_bucket_uses_negative_threshold() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[40.0],
        sell_volumes=[60.0],
        prices=[100.0, 99.0],
        imbalance_threshold=0.20,
    )

    assert result.sell_dominant.observations == 1
    assert result.balanced.observations == 0
    assert result.buy_dominant.observations == 0


def test_balanced_bucket_is_between_thresholds() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[55.0],
        sell_volumes=[45.0],
        prices=[100.0, 101.0],
        imbalance_threshold=0.20,
    )

    assert result.balanced.observations == 1
    assert result.buy_dominant.observations == 0
    assert result.sell_dominant.observations == 0


def test_threshold_boundary_is_buy_dominant() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[60.0],
        sell_volumes=[40.0],
        prices=[100.0, 101.0],
        imbalance_threshold=0.20,
    )

    assert result.buy_dominant.observations == 1


def test_threshold_boundary_is_sell_dominant() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[40.0],
        sell_volumes=[60.0],
        prices=[100.0, 99.0],
        imbalance_threshold=0.20,
    )

    assert result.sell_dominant.observations == 1


def test_performance_objects_are_returned() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 50.0],
        sell_volumes=[30.0, 50.0],
        prices=[100.0, 101.0, 100.0],
    )

    assert isinstance(
        result.buy_dominant,
        OrderFlowPerformance,
    )
    assert isinstance(
        result.balanced,
        OrderFlowPerformance,
    )
    assert isinstance(
        result.sell_dominant,
        OrderFlowPerformance,
    )


def test_empty_bucket_has_zero_statistics() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 70.0],
        sell_volumes=[30.0, 30.0],
        prices=[100.0, 101.0, 102.0],
    )

    assert result.sell_dominant.observations == 0
    assert result.sell_dominant.mean_forward_return == 0.0
    assert result.sell_dominant.positive_rate == 0.0


def test_bucket_positive_rate_is_valid() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 50.0, 30.0],
        sell_volumes=[30.0, 50.0, 70.0],
        prices=[100.0, 101.0, 100.0, 99.0],
    )

    performances = [
        result.buy_dominant,
        result.balanced,
        result.sell_dominant,
    ]

    for performance in performances:
        assert 0.0 <= performance.positive_rate <= 1.0


def test_overall_positive_rate_is_bounded() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[70.0, 50.0, 30.0],
        sell_volumes=[30.0, 50.0, 70.0],
        prices=[100.0, 101.0, 100.0, 99.0],
    )

    assert 0.0 <= result.overall_positive_rate <= 1.0


def test_all_buy_dominant_observations_are_bucketed_correctly() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[90.0, 80.0, 70.0],
        sell_volumes=[10.0, 20.0, 30.0],
        prices=[100.0, 101.0, 102.0, 103.0],
    )

    assert result.buy_dominant.observations == 3
    assert result.balanced.observations == 0
    assert result.sell_dominant.observations == 0


def test_all_sell_dominant_observations_are_bucketed_correctly() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[10.0, 20.0, 30.0],
        sell_volumes=[90.0, 80.0, 70.0],
        prices=[100.0, 99.0, 98.0, 97.0],
    )

    assert result.sell_dominant.observations == 3
    assert result.balanced.observations == 0
    assert result.buy_dominant.observations == 0


def test_all_balanced_observations_are_bucketed_correctly() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[55.0, 52.0, 51.0],
        sell_volumes=[45.0, 48.0, 49.0],
        prices=[100.0, 101.0, 100.0, 102.0],
    )

    assert result.balanced.observations == 3
    assert result.buy_dominant.observations == 0
    assert result.sell_dominant.observations == 0


def test_future_price_does_not_change_order_flow_bucket() -> None:
    common_buy = [80.0, 80.0]
    common_sell = [20.0, 20.0]

    result_up = evaluate_order_flow_impact(
        buy_volumes=common_buy,
        sell_volumes=common_sell,
        prices=[100.0, 101.0, 120.0],
    )

    result_down = evaluate_order_flow_impact(
        buy_volumes=common_buy,
        sell_volumes=common_sell,
        prices=[100.0, 101.0, 80.0],
    )

    assert result_up.buy_dominant.observations == 2
    assert result_down.buy_dominant.observations == 2

    assert result_up.balanced.observations == 0
    assert result_down.balanced.observations == 0

    assert result_up.sell_dominant.observations == 0
    assert result_down.sell_dominant.observations == 0

    assert (
        result_up.overall_mean_forward_return
        != result_down.overall_mean_forward_return
    )


def test_custom_threshold_changes_bucket_assignment() -> None:
    result_default = evaluate_order_flow_impact(
        buy_volumes=[60.0],
        sell_volumes=[40.0],
        prices=[100.0, 101.0],
        imbalance_threshold=0.20,
    )

    result_strict = evaluate_order_flow_impact(
        buy_volumes=[60.0],
        sell_volumes=[40.0],
        prices=[100.0, 101.0],
        imbalance_threshold=0.30,
    )

    assert result_default.buy_dominant.observations == 1
    assert result_strict.balanced.observations == 1


def test_results_are_deterministic() -> None:
    kwargs = {
        "buy_volumes": [80.0, 50.0, 20.0, 70.0],
        "sell_volumes": [20.0, 50.0, 80.0, 30.0],
        "prices": [100.0, 101.0, 100.0, 99.0, 100.0],
    }

    first = evaluate_order_flow_impact(**kwargs)
    second = evaluate_order_flow_impact(**kwargs)

    assert first == second


def test_result_statistics_are_finite() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[80.0, 50.0, 20.0],
        sell_volumes=[20.0, 50.0, 80.0],
        prices=[100.0, 101.0, 100.0, 99.0],
    )

    values = [
        result.overall_mean_forward_return,
        result.overall_positive_rate,
        result.buy_dominant.mean_forward_return,
        result.buy_dominant.positive_rate,
        result.balanced.mean_forward_return,
        result.balanced.positive_rate,
        result.sell_dominant.mean_forward_return,
        result.sell_dominant.positive_rate,
    ]

    for value in values:
        assert math.isfinite(value)


def test_rejects_mismatched_buy_and_sell_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="buy_volumes and sell_volumes must have equal length",
    ):
        evaluate_order_flow_impact(
            buy_volumes=[70.0, 60.0],
            sell_volumes=[30.0],
            prices=[100.0, 101.0],
        )


def test_rejects_mismatched_volume_and_price_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="volume and price sequences must have equal length",
    ):
        evaluate_order_flow_impact(
            buy_volumes=[70.0, 60.0],
            sell_volumes=[30.0, 40.0],
            prices=[100.0, 101.0, 102.0],
        )


def test_rejects_insufficient_prices() -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain at least 2 observations",
    ):
        evaluate_order_flow_impact(
            buy_volumes=[70.0],
            sell_volumes=[30.0],
            prices=[100.0],
        )


@pytest.mark.parametrize(
    "buy_volumes",
    [
        [70.0, "60"],
        [70.0, True],
        [70.0, float("nan")],
        [70.0, float("inf")],
        [70.0, -1.0],
    ],
)
def test_rejects_invalid_buy_volumes(
    buy_volumes: list[object],
) -> None:
    with pytest.raises(ValueError):
        evaluate_order_flow_impact(
            buy_volumes=buy_volumes,  # type: ignore[arg-type]
            sell_volumes=[30.0, 40.0],
            prices=[100.0, 101.0],
        )


@pytest.mark.parametrize(
    "sell_volumes",
    [
        [30.0, "40"],
        [30.0, True],
        [30.0, float("nan")],
        [30.0, float("inf")],
        [30.0, -1.0],
    ],
)
def test_rejects_invalid_sell_volumes(
    sell_volumes: list[object],
) -> None:
    with pytest.raises(ValueError):
        evaluate_order_flow_impact(
            buy_volumes=[70.0, 60.0],
            sell_volumes=sell_volumes,  # type: ignore[arg-type]
            prices=[100.0, 101.0],
        )


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, "101"],
        [100.0, True],
        [100.0, float("nan")],
        [100.0, float("inf")],
        [100.0, 0.0],
        [100.0, -1.0],
    ],
)
def test_rejects_invalid_prices(
    prices: list[object],
) -> None:
    with pytest.raises(ValueError):
        evaluate_order_flow_impact(
            buy_volumes=[70.0, 60.0],
            sell_volumes=[30.0, 40.0],
            prices=prices,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "0.2",
    ],
)
def test_rejects_invalid_imbalance_threshold(
    threshold: object,
) -> None:
    with pytest.raises(ValueError):
        evaluate_order_flow_impact(
            buy_volumes=[70.0, 60.0],
            sell_volumes=[30.0, 40.0],
            prices=[100.0, 101.0],
            imbalance_threshold=threshold,  # type: ignore[arg-type]
        )


def test_rejects_zero_total_order_flow() -> None:
    with pytest.raises(
        ValueError,
        match="each order-flow observation must have positive total volume",
    ):
        evaluate_order_flow_impact(
            buy_volumes=[0.0, 70.0],
            sell_volumes=[0.0, 30.0],
            prices=[100.0, 101.0],
        )


def test_accepts_zero_buy_volume_when_sell_volume_exists() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[0.0],
        sell_volumes=[100.0],
        prices=[100.0, 99.0],
    )

    assert result.sell_dominant.observations == 1


def test_accepts_zero_sell_volume_when_buy_volume_exists() -> None:
    result = evaluate_order_flow_impact(
        buy_volumes=[100.0],
        sell_volumes=[0.0],
        prices=[100.0, 101.0],
    )

    assert result.buy_dominant.observations == 1
