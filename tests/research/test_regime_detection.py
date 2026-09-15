"""Tests for deterministic market-regime detection."""

import math

import pytest

from src.research.regime import (
    RegimeConfig,
    RegimeResult,
    detect_regime,
)


def test_returns_regime_result() -> None:
    prices = [
        100.0 + index
        for index in range(20)
    ]

    result = detect_regime(prices)

    assert isinstance(result, RegimeResult)


def test_detects_uptrend() -> None:
    prices = [
        100.0 + (index * 2.0)
        for index in range(20)
    ]

    result = detect_regime(prices)

    assert result.regime == "TREND_UP"
    assert result.trend_strength > 0.01
    assert result.return_mean > 0.0
    assert result.observations == 20


def test_detects_downtrend() -> None:
    prices = [
        200.0 - (index * 2.0)
        for index in range(20)
    ]

    result = detect_regime(prices)

    assert result.regime == "TREND_DOWN"
    assert result.trend_strength > 0.01
    assert result.return_mean < 0.0
    assert result.observations == 20


def test_detects_range() -> None:
    prices = [
        100.0,
        100.05,
        99.95,
        100.04,
        99.96,
        100.03,
        99.97,
        100.02,
        99.98,
        100.01,
        99.99,
        100.0,
        100.02,
        99.98,
        100.01,
        99.99,
        100.0,
        100.01,
        99.99,
        100.0,
    ]

    result = detect_regime(prices)

    assert result.regime == "RANGE"


def test_detects_high_volatility() -> None:
    prices = [
        100.0,
        110.0,
        90.0,
        115.0,
        85.0,
        120.0,
        80.0,
        125.0,
        75.0,
        130.0,
        70.0,
        135.0,
        65.0,
        140.0,
        60.0,
        145.0,
        55.0,
        150.0,
        50.0,
        155.0,
    ]

    result = detect_regime(prices)

    assert result.regime == "HIGH_VOLATILITY"
    assert result.volatility >= 0.02


def test_uses_percentage_returns() -> None:
    prices_a = [
        100.0 + index
        for index in range(20)
    ]

    prices_b = [
        price * 10.0
        for price in prices_a
    ]

    result_a = detect_regime(prices_a)
    result_b = detect_regime(prices_b)

    assert result_a.regime == result_b.regime
    assert math.isclose(
        result_a.volatility,
        result_b.volatility,
    )
    assert math.isclose(
        result_a.return_mean,
        result_b.return_mean,
    )


def test_observation_count_matches_input() -> None:
    prices = [
        100.0 + index
        for index in range(25)
    ]

    result = detect_regime(prices)

    assert result.observations == 25


def test_custom_config_changes_thresholds() -> None:
    prices = [
        100.0 + (index * 0.1)
        for index in range(20)
    ]

    config = RegimeConfig(
        trend_threshold=0.001,
        volatility_threshold=1.0,
        minimum_observations=20,
    )

    result = detect_regime(
        prices,
        config=config,
    )

    assert result.regime == "TREND_UP"


def test_rejects_insufficient_observations() -> None:
    prices = [100.0] * 19

    with pytest.raises(
        ValueError,
        match="insufficient price observations",
    ):
        detect_regime(prices)


def test_rejects_string_prices() -> None:
    with pytest.raises(
        ValueError,
        match="prices must be a numeric sequence",
    ):
        detect_regime("100,101,102")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, 0.0] + [100.0] * 18,
        [100.0, -1.0] + [100.0] * 18,
    ],
)
def test_rejects_non_positive_prices(
    prices: list[float],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must be positive",
    ):
        detect_regime(prices)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, float("nan")] + [100.0] * 18,
        [100.0, float("inf")] + [100.0] * 18,
        [100.0, float("-inf")] + [100.0] * 18,
    ],
)
def test_rejects_non_finite_prices(
    prices: list[float],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain only finite values",
    ):
        detect_regime(prices)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, True] + [100.0] * 18,
        [100.0, "101"] + [100.0] * 18,  # type: ignore[list-item]
    ],
)
def test_rejects_non_numeric_price_values(
    prices: list[object],
) -> None:
    with pytest.raises(
        ValueError,
        match="prices must contain only numeric values",
    ):
        detect_regime(prices)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        RegimeConfig(trend_threshold=-0.01),
        RegimeConfig(volatility_threshold=-0.01),
    ],
)
def test_rejects_negative_thresholds(
    config: RegimeConfig,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        detect_regime(
            [100.0] * 20,
            config=config,
        )


@pytest.mark.parametrize(
    "config",
    [
        RegimeConfig(trend_threshold=float("nan")),
        RegimeConfig(trend_threshold=float("inf")),
        RegimeConfig(volatility_threshold=float("nan")),
        RegimeConfig(volatility_threshold=float("inf")),
    ],
)
def test_rejects_non_finite_thresholds(
    config: RegimeConfig,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        detect_regime(
            [100.0] * 20,
            config=config,
        )


@pytest.mark.parametrize(
    "minimum",
    [
        True,
        1,
        0,
        -1,
        2.5,
    ],
)
def test_rejects_invalid_minimum_observations(
    minimum: object,
) -> None:
    config = RegimeConfig(
        minimum_observations=minimum,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        detect_regime(
            [100.0] * 20,
            config=config,
        )


def test_constant_prices_are_range() -> None:
    prices = [100.0] * 20

    result = detect_regime(prices)

    assert result.regime == "RANGE"
    assert result.trend_strength == 0.0
    assert result.volatility == 0.0
    assert result.return_mean == 0.0


def test_result_values_are_finite() -> None:
    prices = [
        100.0 + index
        for index in range(20)
    ]

    result = detect_regime(prices)

    assert math.isfinite(result.trend_strength)
    assert math.isfinite(result.volatility)
    assert math.isfinite(result.return_mean)


def test_result_is_deterministic() -> None:
    prices = [
        100.0,
        101.0,
        102.5,
        101.5,
        103.0,
        104.0,
        105.5,
        106.0,
        107.5,
        108.0,
        109.0,
        110.0,
        111.5,
        112.0,
        113.0,
        114.5,
        115.0,
        116.0,
        117.5,
        118.0,
    ]

    first = detect_regime(prices)
    second = detect_regime(prices)

    assert first == second


def test_different_price_scales_preserve_trend_strength() -> None:
    prices = [
        100.0 + (index * 2.0)
        for index in range(20)
    ]

    scaled_prices = [
        price * 1000.0
        for price in prices
    ]

    first = detect_regime(prices)
    second = detect_regime(scaled_prices)

    assert math.isclose(
        first.trend_strength,
        second.trend_strength,
    )
    assert first.regime == second.regime
