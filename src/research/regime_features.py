"""Feature representation for market-regime research."""

from __future__ import annotations

from dataclasses import dataclass

from src.research.regime import RegimeResult


@dataclass(frozen=True)
class RegimeFeatures:
    """Numerical representation of a detected market regime."""

    trend_strength: float
    volatility: float
    return_mean: float
    is_trend_up: int
    is_trend_down: int
    is_high_volatility: int
    is_range: int


def build_regime_features(
    result: RegimeResult,
) -> RegimeFeatures:
    """Convert a regime result into model-ready numerical features.

    The transformation is deterministic and contains no forecasting,
    future-data access, or trading execution logic.
    """
    if not isinstance(result, RegimeResult):
        raise ValueError(
            "result must be a RegimeResult"
        )

    regime_flags = {
        "TREND_UP": 0,
        "TREND_DOWN": 0,
        "HIGH_VOLATILITY": 0,
        "RANGE": 0,
    }

    if result.regime not in regime_flags:
        raise ValueError(
            "result contains an unsupported regime"
        )

    regime_flags[result.regime] = 1

    return RegimeFeatures(
        trend_strength=float(result.trend_strength),
        volatility=float(result.volatility),
        return_mean=float(result.return_mean),
        is_trend_up=regime_flags["TREND_UP"],
        is_trend_down=regime_flags["TREND_DOWN"],
        is_high_volatility=regime_flags["HIGH_VOLATILITY"],
        is_range=regime_flags["RANGE"],
    )
