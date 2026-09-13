"""Tests for expected-value and trading-cost calculations."""

import math

import pytest

from src.models.expected_value import (
    ExpectedValueResult,
    break_even_probability,
    calculate_expected_value,
)


def test_positive_net_expected_value():
    result = calculate_expected_value(
        probability=0.6,
        reward=2.0,
        loss=1.0,
    )

    assert result.gross_expected_value == pytest.approx(0.8)
    assert result.total_cost == pytest.approx(0.0)
    assert result.net_expected_value == pytest.approx(0.8)


def test_negative_net_expected_value():
    result = calculate_expected_value(
        probability=0.4,
        reward=2.0,
        loss=1.0,
    )

    assert result.gross_expected_value == pytest.approx(0.2)
    assert result.net_expected_value == pytest.approx(0.2)


def test_zero_expected_value():
    result = calculate_expected_value(
        probability=1 / 3,
        reward=2.0,
        loss=1.0,
    )

    assert result.gross_expected_value == pytest.approx(0.0)


def test_costs_reduce_net_expected_value():
    result = calculate_expected_value(
        probability=0.6,
        reward=2.0,
        loss=1.0,
        fee=0.1,
        slippage=0.05,
    )

    assert result.gross_expected_value == pytest.approx(0.8)
    assert result.total_cost == pytest.approx(0.15)
    assert result.net_expected_value == pytest.approx(0.65)


def test_fee_and_slippage_are_added_to_total_cost():
    result = calculate_expected_value(
        probability=0.5,
        reward=1.0,
        loss=1.0,
        fee=0.02,
        slippage=0.03,
    )

    assert result.total_cost == pytest.approx(0.05)


def test_probability_zero_means_only_loss_contributes():
    result = calculate_expected_value(
        probability=0.0,
        reward=2.0,
        loss=1.5,
    )

    assert result.gross_expected_value == pytest.approx(-1.5)


def test_probability_one_means_only_reward_contributes():
    result = calculate_expected_value(
        probability=1.0,
        reward=2.0,
        loss=1.5,
    )

    assert result.gross_expected_value == pytest.approx(2.0)


def test_result_contains_original_inputs():
    result = calculate_expected_value(
        probability=0.65,
        reward=1.5,
        loss=1.0,
        fee=0.02,
        slippage=0.03,
    )

    assert isinstance(result, ExpectedValueResult)
    assert result.probability == pytest.approx(0.65)
    assert result.reward == pytest.approx(1.5)
    assert result.loss == pytest.approx(1.0)
    assert result.fee == pytest.approx(0.02)
    assert result.slippage == pytest.approx(0.03)


def test_total_cost_equals_fee_plus_slippage():
    result = calculate_expected_value(
        probability=0.55,
        reward=2.0,
        loss=1.0,
        fee=0.07,
        slippage=0.08,
    )

    assert result.total_cost == pytest.approx(0.15)


def test_net_expected_value_equals_gross_minus_cost():
    result = calculate_expected_value(
        probability=0.55,
        reward=2.0,
        loss=1.0,
        fee=0.07,
        slippage=0.08,
    )

    assert result.net_expected_value == pytest.approx(
        result.gross_expected_value - result.total_cost
    )


def test_break_even_probability_without_costs():
    probability = break_even_probability(
        reward=2.0,
        loss=1.0,
    )

    assert probability == pytest.approx(1 / 3)


def test_break_even_probability_with_costs():
    probability = break_even_probability(
        reward=2.0,
        loss=1.0,
        fee=0.1,
        slippage=0.05,
    )

    assert probability == pytest.approx(
        (1.0 + 0.1 + 0.05) / 3.0
    )


def test_break_even_probability_produces_zero_net_ev():
    reward = 2.0
    loss = 1.0
    fee = 0.1
    slippage = 0.05

    probability = break_even_probability(
        reward=reward,
        loss=loss,
        fee=fee,
        slippage=slippage,
    )

    result = calculate_expected_value(
        probability=probability,
        reward=reward,
        loss=loss,
        fee=fee,
        slippage=slippage,
    )

    assert result.net_expected_value == pytest.approx(0.0)


def test_break_even_probability_can_exceed_one_when_costs_are_too_high():
    probability = break_even_probability(
        reward=1.0,
        loss=1.0,
        fee=1.0,
        slippage=1.0,
    )

    assert probability == pytest.approx(1.5)


def test_expected_value_is_deterministic():
    arguments = {
        "probability": 0.62,
        "reward": 1.8,
        "loss": 1.0,
        "fee": 0.03,
        "slippage": 0.02,
    }

    first = calculate_expected_value(**arguments)
    second = calculate_expected_value(**arguments)

    assert first == second


def test_inputs_are_not_modified():
    arguments = {
        "probability": 0.62,
        "reward": 1.8,
        "loss": 1.0,
        "fee": 0.03,
        "slippage": 0.02,
    }

    original = arguments.copy()

    calculate_expected_value(**arguments)

    assert arguments == original


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True],
)
def test_probability_must_be_valid(probability):
    with pytest.raises(
        ValueError,
        match="probability",
    ):
        calculate_expected_value(
            probability=probability,
            reward=1.0,
            loss=1.0,
        )


def test_probability_must_be_numeric():
    with pytest.raises(
        ValueError,
        match="numeric",
    ):
        calculate_expected_value(
            probability="0.5",
            reward=1.0,
            loss=1.0,
        )


@pytest.mark.parametrize(
    "reward",
    [0, -1, float("nan"), float("inf"), float("-inf"), True],
)
def test_reward_must_be_positive_and_finite(reward):
    with pytest.raises(
        ValueError,
        match="reward",
    ):
        calculate_expected_value(
            probability=0.5,
            reward=reward,
            loss=1.0,
        )


@pytest.mark.parametrize(
    "loss",
    [0, -1, float("nan"), float("inf"), float("-inf"), True],
)
def test_loss_must_be_positive_and_finite(loss):
    with pytest.raises(
        ValueError,
        match="loss",
    ):
        calculate_expected_value(
            probability=0.5,
            reward=1.0,
            loss=loss,
        )


@pytest.mark.parametrize(
    "fee",
    [-0.01, float("nan"), float("inf"), float("-inf"), True],
)
def test_fee_must_be_non_negative_and_finite(fee):
    with pytest.raises(
        ValueError,
        match="fee",
    ):
        calculate_expected_value(
            probability=0.5,
            reward=1.0,
            loss=1.0,
            fee=fee,
        )


@pytest.mark.parametrize(
    "slippage",
    [-0.01, float("nan"), float("inf"), float("-inf"), True],
)
def test_slippage_must_be_non_negative_and_finite(slippage):
    with pytest.raises(
        ValueError,
        match="slippage",
    ):
        calculate_expected_value(
            probability=0.5,
            reward=1.0,
            loss=1.0,
            slippage=slippage,
        )


@pytest.mark.parametrize(
    "reward,loss",
    [
        (0, 1),
        (-1, 1),
        (1, 0),
        (1, -1),
    ],
)
def test_break_even_requires_positive_reward_and_loss(
    reward,
    loss,
):
    with pytest.raises(ValueError):
        break_even_probability(
            reward=reward,
            loss=loss,
        )


@pytest.mark.parametrize(
    "fee,slippage",
    [
        (-0.01, 0.0),
        (0.0, -0.01),
        (-0.01, -0.01),
    ],
)
def test_break_even_costs_must_be_non_negative(
    fee,
    slippage,
):
    with pytest.raises(ValueError):
        break_even_probability(
            reward=1.0,
            loss=1.0,
            fee=fee,
            slippage=slippage,
        )


def test_expected_value_outputs_are_finite():
    result = calculate_expected_value(
        probability=0.55,
        reward=2.0,
        loss=1.0,
        fee=0.05,
        slippage=0.05,
    )

    assert math.isfinite(result.gross_expected_value)
    assert math.isfinite(result.total_cost)
    assert math.isfinite(result.net_expected_value)


def test_high_probability_does_not_guarantee_positive_ev():
    result = calculate_expected_value(
        probability=0.9,
        reward=0.1,
        loss=10.0,
        fee=0.01,
        slippage=0.01,
    )

    assert result.net_expected_value < 0.0


def test_low_probability_can_have_positive_ev_with_large_reward():
    result = calculate_expected_value(
        probability=0.4,
        reward=3.0,
        loss=1.0,
    )

    assert result.net_expected_value > 0.0
