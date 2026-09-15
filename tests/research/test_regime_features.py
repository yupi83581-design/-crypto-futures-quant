"""Tests for market-regime feature representation."""

import math

import pytest

from src.research.regime import RegimeResult
from src.research.regime_features import (
    RegimeFeatures,
    build_regime_features,
)


def make_result(regime: str) -> RegimeResult:
    return RegimeResult(
        regime=regime,
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )


@pytest.mark.parametrize(
    ("regime", "flag"),
    [
        ("TREND_UP", "is_trend_up"),
        ("TREND_DOWN", "is_trend_down"),
        ("HIGH_VOLATILITY", "is_high_volatility"),
        ("RANGE", "is_range"),
    ],
)
def test_builds_features_for_each_regime(
    regime: str,
    flag: str,
) -> None:
    result = make_result(regime)

    features = build_regime_features(result)

    assert isinstance(features, RegimeFeatures)
    assert getattr(features, flag) == 1


@pytest.mark.parametrize(
    "regime",
    [
        "TREND_UP",
        "TREND_DOWN",
        "HIGH_VOLATILITY",
        "RANGE",
    ],
)
def test_exactly_one_regime_flag_is_active(
    regime: str,
) -> None:
    features = build_regime_features(
        make_result(regime)
    )

    flags = [
        features.is_trend_up,
        features.is_trend_down,
        features.is_high_volatility,
        features.is_range,
    ]

    assert sum(flags) == 1
    assert all(flag in (0, 1) for flag in flags)


def test_preserves_numeric_regime_values() -> None:
    result = RegimeResult(
        regime="TREND_UP",
        trend_strength=0.123,
        volatility=0.045,
        return_mean=0.006,
        observations=50,
    )

    features = build_regime_features(result)

    assert math.isclose(
        features.trend_strength,
        0.123,
    )
    assert math.isclose(
        features.volatility,
        0.045,
    )
    assert math.isclose(
        features.return_mean,
        0.006,
    )


def test_converts_numeric_values_to_float() -> None:
    result = RegimeResult(
        regime="RANGE",
        trend_strength=1,
        volatility=2,
        return_mean=3,
        observations=20,
    )

    features = build_regime_features(result)

    assert isinstance(
        features.trend_strength,
        float,
    )
    assert isinstance(
        features.volatility,
        float,
    )
    assert isinstance(
        features.return_mean,
        float,
    )


def test_result_is_deterministic() -> None:
    result = make_result("TREND_DOWN")

    first = build_regime_features(result)
    second = build_regime_features(result)

    assert first == second


def test_rejects_non_regime_result() -> None:
    with pytest.raises(
        ValueError,
        match="result must be a RegimeResult",
    ):
        build_regime_features(
            "TREND_UP"  # type: ignore[arg-type]
        )


def test_rejects_unsupported_regime() -> None:
    result = RegimeResult(
        regime="UNKNOWN",
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )

    with pytest.raises(
        ValueError,
        match="result contains an unsupported regime",
    ):
        build_regime_features(result)


def test_flags_are_integers() -> None:
    features = build_regime_features(
        make_result("HIGH_VOLATILITY")
    )

    assert isinstance(features.is_trend_up, int)
    assert isinstance(features.is_trend_down, int)
    assert isinstance(
        features.is_high_volatility,
        int,
    )
    assert isinstance(features.is_range, int)


def test_range_has_only_range_flag() -> None:
    features = build_regime_features(
        make_result("RANGE")
    )

    assert features.is_trend_up == 0
    assert features.is_trend_down == 0
    assert features.is_high_volatility == 0
    assert features.is_range == 1


def test_trend_up_has_only_trend_up_flag() -> None:
    features = build_regime_features(
        make_result("TREND_UP")
    )

    assert features.is_trend_up == 1
    assert features.is_trend_down == 0
    assert features.is_high_volatility == 0
    assert features.is_range == 0


def test_trend_down_has_only_trend_down_flag() -> None:
    features = build_regime_features(
        make_result("TREND_DOWN")
    )

    assert features.is_trend_up == 0
    assert features.is_trend_down == 1
    assert features.is_high_volatility == 0
    assert features.is_range == 0


def test_high_volatility_has_only_high_volatility_flag() -> None:
    features = build_regime_features(
        make_result("HIGH_VOLATILITY")
    )

    assert features.is_trend_up == 0
    assert features.is_trend_down == 0
    assert features.is_high_volatility == 1
    assert features.is_range == 0
