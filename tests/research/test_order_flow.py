"""Tests for order-flow research features."""

import math

import pytest

from src.research.order_flow import (
    OrderFlowResult,
    calculate_order_flow,
)


def test_returns_order_flow_result() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert isinstance(result, OrderFlowResult)


def test_preserves_buy_volume() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert result.buy_volume == 70.0


def test_preserves_sell_volume() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert result.sell_volume == 30.0


def test_total_volume_is_sum_of_buy_and_sell() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert result.total_volume == 100.0


def test_imbalance_is_calculated_correctly() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert math.isclose(
        result.imbalance,
        0.40,
    )


def test_buy_pressure_is_calculated_correctly() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert math.isclose(
        result.buy_pressure,
        0.70,
    )


def test_sell_pressure_is_calculated_correctly() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert math.isclose(
        result.sell_pressure,
        0.30,
    )


def test_pressures_sum_to_one() -> None:
    result = calculate_order_flow(
        buy_volume=70.0,
        sell_volume=30.0,
    )

    assert math.isclose(
        result.buy_pressure + result.sell_pressure,
        1.0,
    )


def test_imbalance_equals_buy_pressure_minus_sell_pressure() -> None:
    result = calculate_order_flow(
        buy_volume=80.0,
        sell_volume=20.0,
    )

    assert math.isclose(
        result.imbalance,
        result.buy_pressure - result.sell_pressure,
    )


def test_equal_volume_produces_zero_imbalance() -> None:
    result = calculate_order_flow(
        buy_volume=50.0,
        sell_volume=50.0,
    )

    assert result.imbalance == 0.0
    assert result.buy_pressure == 0.5
    assert result.sell_pressure == 0.5


def test_buy_dominant_volume_produces_positive_imbalance() -> None:
    result = calculate_order_flow(
        buy_volume=90.0,
        sell_volume=10.0,
    )

    assert result.imbalance > 0.0


def test_sell_dominant_volume_produces_negative_imbalance() -> None:
    result = calculate_order_flow(
        buy_volume=10.0,
        sell_volume=90.0,
    )

    assert result.imbalance < 0.0


def test_imbalance_is_bounded_between_minus_one_and_one() -> None:
    test_cases = [
        (100.0, 0.0),
        (90.0, 10.0),
        (50.0, 50.0),
        (10.0, 90.0),
        (0.0, 100.0),
    ]

    for buy_volume, sell_volume in test_cases:
        result = calculate_order_flow(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
        )

        assert -1.0 <= result.imbalance <= 1.0


def test_all_buy_volume_produces_maximum_positive_imbalance() -> None:
    result = calculate_order_flow(
        buy_volume=100.0,
        sell_volume=0.0,
    )

    assert result.imbalance == 1.0
    assert result.buy_pressure == 1.0
    assert result.sell_pressure == 0.0


def test_all_sell_volume_produces_maximum_negative_imbalance() -> None:
    result = calculate_order_flow(
        buy_volume=0.0,
        sell_volume=100.0,
    )

    assert result.imbalance == -1.0
    assert result.buy_pressure == 0.0
    assert result.sell_pressure == 1.0


def test_zero_buy_volume_is_valid_when_sell_volume_is_positive() -> None:
    result = calculate_order_flow(
        buy_volume=0.0,
        sell_volume=100.0,
    )

    assert result.buy_volume == 0.0
    assert result.total_volume == 100.0


def test_zero_sell_volume_is_valid_when_buy_volume_is_positive() -> None:
    result = calculate_order_flow(
        buy_volume=100.0,
        sell_volume=0.0,
    )

    assert result.sell_volume == 0.0
    assert result.total_volume == 100.0


def test_integer_inputs_are_supported() -> None:
    result = calculate_order_flow(
        buy_volume=70,
        sell_volume=30,
    )

    assert result.total_volume == 100.0
    assert result.imbalance == 0.4


def test_results_are_deterministic() -> None:
    first = calculate_order_flow(
        buy_volume=73.0,
        sell_volume=27.0,
    )

    second = calculate_order_flow(
        buy_volume=73.0,
        sell_volume=27.0,
    )

    assert first == second


def test_result_values_are_finite() -> None:
    result = calculate_order_flow(
        buy_volume=73.0,
        sell_volume=27.0,
    )

    values = [
        result.buy_volume,
        result.sell_volume,
        result.total_volume,
        result.imbalance,
        result.buy_pressure,
        result.sell_pressure,
    ]

    for value in values:
        assert math.isfinite(value)


def test_rejects_zero_total_volume() -> None:
    with pytest.raises(
        ValueError,
        match="total volume must be greater than zero",
    ):
        calculate_order_flow(
            buy_volume=0.0,
            sell_volume=0.0,
        )


@pytest.mark.parametrize(
    "buy_volume,sell_volume",
    [
        (-1.0, 10.0),
        (10.0, -1.0),
        (-1.0, -1.0),
    ],
)
def test_rejects_negative_volume(
    buy_volume: float,
    sell_volume: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        calculate_order_flow(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
        )


@pytest.mark.parametrize(
    "buy_volume,sell_volume",
    [
        (True, 10.0),
        (10.0, True),
        (False, 10.0),
        (10.0, False),
    ],
)
def test_rejects_boolean_volume(
    buy_volume: object,
    sell_volume: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        calculate_order_flow(
            buy_volume=buy_volume,  # type: ignore[arg-type]
            sell_volume=sell_volume,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "buy_volume,sell_volume",
    [
        ("70", 30.0),
        (70.0, "30"),
    ],
)
def test_rejects_string_volume(
    buy_volume: object,
    sell_volume: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        calculate_order_flow(
            buy_volume=buy_volume,  # type: ignore[arg-type]
            sell_volume=sell_volume,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "buy_volume,sell_volume",
    [
        (float("nan"), 30.0),
        (70.0, float("nan")),
        (float("inf"), 30.0),
        (70.0, float("inf")),
        (float("-inf"), 30.0),
        (70.0, float("-inf")),
    ],
)
def test_rejects_non_finite_volume(
    buy_volume: float,
    sell_volume: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        calculate_order_flow(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
        )
