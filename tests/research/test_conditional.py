"""Tests for conditional regime and order-flow research."""

from __future__ import annotations

import math

import pytest

from src.research.conditional import (
    ConditionalEvaluationResult,
    ConditionalPerformance,
    evaluate_conditional_impact,
)
from src.research.regime import RegimeConfig


def _base_inputs() -> tuple[list[float], list[float], list[float]]:
    prices = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        104.0,
        103.0,
        102.0,
        101.0,
        100.0,
        100.5,
        101.0,
        101.5,
        102.0,
        102.5,
        103.0,
        103.5,
        104.0,
        104.5,
        105.0,
        106.0,
        107.0,
        108.0,
    ]

    buy_volumes = [
        80.0,
        80.0,
        80.0,
        80.0,
        80.0,
        80.0,
        20.0,
        20.0,
        20.0,
        20.0,
        50.0,
        50.0,
        50.0,
        50.0,
        80.0,
        80.0,
        80.0,
        80.0,
        20.0,
        20.0,
        20.0,
        80.0,
        80.0,
        80.0,
    ]

    sell_volumes = [
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        80.0,
        80.0,
        80.0,
        80.0,
        50.0,
        50.0,
        50.0,
        50.0,
        20.0,
        20.0,
        20.0,
        20.0,
        80.0,
        80.0,
        80.0,
        20.0,
        20.0,
        20.0,
    ]

    return buy_volumes, sell_volumes, prices


def _config() -> RegimeConfig:
    return RegimeConfig(
        trend_threshold=0.01,
        volatility_threshold=0.50,
        minimum_observations=5,
    )


def test_returns_conditional_evaluation_result() -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    assert isinstance(result, ConditionalEvaluationResult)


def test_total_observations_matches_forward_returns() -> None:
    buy, sell, prices = _base_inputs()
    config = _config()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
    )

    assert result.total_observations == len(prices) - (
        config.minimum_observations
    )


def test_overall_statistics_are_finite() -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    assert math.isfinite(result.overall_mean_forward_return)
    assert math.isfinite(result.overall_positive_rate)


def test_overall_positive_rate_is_between_zero_and_one() -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    assert 0.0 <= result.overall_positive_rate <= 1.0


@pytest.mark.parametrize(
    "field_name",
    [
        "trend_up_buy_dominant",
        "trend_up_balanced",
        "trend_up_sell_dominant",
        "trend_down_buy_dominant",
        "trend_down_balanced",
        "trend_down_sell_dominant",
        "high_volatility_buy_dominant",
        "high_volatility_balanced",
        "high_volatility_sell_dominant",
        "range_buy_dominant",
        "range_balanced",
        "range_sell_dominant",
    ],
)
def test_all_conditional_buckets_have_expected_type(
    field_name: str,
) -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    performance = getattr(result, field_name)

    assert isinstance(performance, ConditionalPerformance)
    assert performance.observations >= 0
    assert math.isfinite(performance.mean_forward_return)
    assert 0.0 <= performance.positive_rate <= 1.0


def test_bucket_observations_sum_to_total_observations() -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    fields = [
        "trend_up_buy_dominant",
        "trend_up_balanced",
        "trend_up_sell_dominant",
        "trend_down_buy_dominant",
        "trend_down_balanced",
        "trend_down_sell_dominant",
        "high_volatility_buy_dominant",
        "high_volatility_balanced",
        "high_volatility_sell_dominant",
        "range_buy_dominant",
        "range_balanced",
        "range_sell_dominant",
    ]

    bucket_total = sum(
        getattr(result, field).observations
        for field in fields
    )

    assert bucket_total == result.total_observations


def test_empty_buckets_return_zero_statistics() -> None:
    buy, sell, prices = _base_inputs()

    result = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=_config(),
    )

    fields = [
        "trend_up_buy_dominant",
        "trend_up_balanced",
        "trend_up_sell_dominant",
        "trend_down_buy_dominant",
        "trend_down_balanced",
        "trend_down_sell_dominant",
        "high_volatility_buy_dominant",
        "high_volatility_balanced",
        "high_volatility_sell_dominant",
        "range_buy_dominant",
        "range_balanced",
        "range_sell_dominant",
    ]

    for field in fields:
        performance = getattr(result, field)

        if performance.observations == 0:
            assert performance.mean_forward_return == 0.0
            assert performance.positive_rate == 0.0


def test_result_is_deterministic() -> None:
    buy, sell, prices = _base_inputs()
    config = _config()

    first = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
    )

    second = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
    )

    assert first == second


def test_future_price_changes_outcome_not_bucket_assignment() -> None:
    buy, sell, prices = _base_inputs()

    changed_prices = prices.copy()
    changed_prices[-1] = 1000.0

    config = _config()

    original = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
    )

    changed = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=changed_prices,
        regime_config=config,
    )

    assert original.total_observations == changed.total_observations

    fields = [
        "trend_up_buy_dominant",
        "trend_up_balanced",
        "trend_up_sell_dominant",
        "trend_down_buy_dominant",
        "trend_down_balanced",
        "trend_down_sell_dominant",
        "high_volatility_buy_dominant",
        "high_volatility_balanced",
        "high_volatility_sell_dominant",
        "range_buy_dominant",
        "range_balanced",
        "range_sell_dominant",
    ]

    for field in fields:
        assert getattr(original, field).observations == getattr(
            changed,
            field,
        ).observations


def test_future_price_changes_forward_return_statistics() -> None:
    buy, sell, prices = _base_inputs()

    changed_prices = prices.copy()
    changed_prices[-1] = 1000.0

    config = _config()

    original = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
    )

    changed = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=changed_prices,
        regime_config=config,
    )

    assert (
        original.overall_mean_forward_return
        != changed.overall_mean_forward_return
    )


def test_default_regime_config_is_supported() -> None:
    buy, sell, prices = _base_inputs()

    prices = prices + [109.0]

    result = evaluate_conditional_impact(
        buy_volumes=buy + [80.0],
        sell_volumes=sell + [20.0],
        prices=prices,
    )

    assert result.total_observations == (
        len(prices) - RegimeConfig().minimum_observations
    )


def test_custom_imbalance_threshold_changes_bucket_assignment() -> None:
    buy, sell, prices = _base_inputs()

    config = _config()

    low_threshold = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
        imbalance_threshold=0.20,
    )

    high_threshold = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=config,
        imbalance_threshold=0.90,
    )

    assert low_threshold != high_threshold


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
    ],
)
def test_invalid_imbalance_threshold_is_rejected(
    threshold: float,
) -> None:
    buy, sell, prices = _base_inputs()

    with pytest.raises(ValueError):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config=_config(),
            imbalance_threshold=threshold,
        )


def test_mismatched_buy_and_sell_lengths_are_rejected() -> None:
    buy, sell, prices = _base_inputs()

    with pytest.raises(ValueError, match="equal length"):
        evaluate_conditional_impact(
            buy_volumes=buy[:-1],
            sell_volumes=sell,
            prices=prices,
            regime_config=_config(),
        )


def test_mismatched_volume_and_price_lengths_are_rejected() -> None:
    buy, sell, prices = _base_inputs()

    with pytest.raises(ValueError, match="equal length"):
        evaluate_conditional_impact(
            buy_volumes=buy[:-1],
            sell_volumes=sell[:-1],
            prices=prices,
            regime_config=_config(),
        )


def test_insufficient_prices_are_rejected() -> None:
    buy = [50.0, 50.0, 50.0, 50.0, 50.0]
    sell = [50.0, 50.0, 50.0, 50.0, 50.0]
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]

    config = RegimeConfig(minimum_observations=5)

    with pytest.raises(
        ValueError,
        match="insufficient prices",
    ):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config=config,
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [50.0, 50.0, "bad", 50.0, 50.0],
        [50.0, 50.0, True, 50.0, 50.0],
        [50.0, 50.0, float("nan"), 50.0, 50.0],
        [50.0, 50.0, float("inf"), 50.0, 50.0],
    ],
)
def test_invalid_buy_volume_values_are_rejected(
    bad_values: list[float],
) -> None:
    _, sell, prices = _base_inputs()

    with pytest.raises(ValueError):
        evaluate_conditional_impact(
            buy_volumes=bad_values,
            sell_volumes=sell[:5],
            prices=prices[:5],
            regime_config=RegimeConfig(minimum_observations=3),
        )


@pytest.mark.parametrize(
    "bad_values",
    [
        [50.0, 50.0, "bad", 50.0, 50.0],
        [50.0, 50.0, True, 50.0, 50.0],
        [50.0, 50.0, float("nan"), 50.0, 50.0],
        [50.0, 50.0, float("inf"), 50.0, 50.0],
    ],
)
def test_invalid_sell_volume_values_are_rejected(
    bad_values: list[float],
) -> None:
    buy, _, prices = _base_inputs()

    with pytest.raises(ValueError):
        evaluate_conditional_impact(
            buy_volumes=buy[:5],
            sell_volumes=bad_values,
            prices=prices[:5],
            regime_config=RegimeConfig(minimum_observations=3),
        )


@pytest.mark.parametrize(
    "bad_prices",
    [
        [100.0, 101.0, 0.0, 103.0, 104.0],
        [100.0, 101.0, -1.0, 103.0, 104.0],
    ],
)
def test_non_positive_prices_are_rejected(
    bad_prices: list[float],
) -> None:
    buy = [50.0] * 5
    sell = [50.0] * 5

    with pytest.raises(ValueError, match="positive"):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=bad_prices,
            regime_config=RegimeConfig(minimum_observations=3),
        )


def test_zero_total_order_flow_is_rejected() -> None:
    buy, sell, prices = _base_inputs()

    buy[5] = 0.0
    sell[5] = 0.0

    with pytest.raises(
        ValueError,
        match="positive total volume",
    ):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config=_config(),
        )


def test_negative_buy_volume_is_rejected() -> None:
    buy, sell, prices = _base_inputs()
    buy[5] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config=_config(),
        )


def test_negative_sell_volume_is_rejected() -> None:
    buy, sell, prices = _base_inputs()
    sell[5] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config=_config(),
        )


def test_invalid_regime_config_type_is_rejected() -> None:
    buy, sell, prices = _base_inputs()

    with pytest.raises(
        ValueError,
        match="regime_config",
    ):
        evaluate_conditional_impact(
            buy_volumes=buy,
            sell_volumes=sell,
            prices=prices,
            regime_config="invalid",  # type: ignore[arg-type]
        )


def test_regime_configuration_changes_evaluation_window() -> None:
    buy, sell, prices = _base_inputs()

    short_window = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=RegimeConfig(
            minimum_observations=5,
            trend_threshold=0.01,
            volatility_threshold=0.50,
        ),
    )

    longer_window = evaluate_conditional_impact(
        buy_volumes=buy,
        sell_volumes=sell,
        prices=prices,
        regime_config=RegimeConfig(
            minimum_observations=10,
            trend_threshold=0.01,
            volatility_threshold=0.50,
        ),
    )

    assert short_window.total_observations == len(prices) - 5
    assert longer_window.total_observations == len(prices) - 10
