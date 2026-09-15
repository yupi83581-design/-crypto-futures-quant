"""Final validation for the Phase 12 advanced research stack."""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.models.advanced import AdvancedModelResult
from src.research.conditional import (
    ConditionalEvaluationResult,
    ConditionalPerformance,
)
from src.research.edge_validation import EdgeValidationResult
from src.research.order_flow import OrderFlowResult
from src.research.regime import RegimeResult
from src.research.regime_features import RegimeFeatures


@dataclass(frozen=True)
class Phase12ValidationResult:
    """Validation status for every Phase 12 research component."""

    regime_valid: bool
    regime_features_valid: bool
    order_flow_valid: bool
    conditional_valid: bool
    edge_validation_valid: bool
    advanced_model_valid: bool
    passed: bool


def validate_phase12_stack(
    *,
    regime: RegimeResult,
    regime_features: RegimeFeatures,
    order_flow: OrderFlowResult,
    conditional: ConditionalEvaluationResult,
    edge_validation: EdgeValidationResult,
    advanced_model: AdvancedModelResult,
) -> Phase12ValidationResult:
    """Validate the complete Phase 12 research stack.

    This validates object types, numerical ranges, and structural
    integrity only.

    It does not claim that a market edge is profitable or
    statistically significant.
    """
    regime_valid = _validate_regime(regime)
    regime_features_valid = _validate_regime_features(regime_features)
    order_flow_valid = _validate_order_flow(order_flow)
    conditional_valid = _validate_conditional(conditional)
    edge_validation_valid = _validate_edge_validation(edge_validation)
    advanced_model_valid = _validate_advanced_model(advanced_model)

    passed = all(
        (
            regime_valid,
            regime_features_valid,
            order_flow_valid,
            conditional_valid,
            edge_validation_valid,
            advanced_model_valid,
        )
    )

    return Phase12ValidationResult(
        regime_valid=regime_valid,
        regime_features_valid=regime_features_valid,
        order_flow_valid=order_flow_valid,
        conditional_valid=conditional_valid,
        edge_validation_valid=edge_validation_valid,
        advanced_model_valid=advanced_model_valid,
        passed=passed,
    )


def _validate_regime(result: RegimeResult) -> bool:
    if not isinstance(result, RegimeResult):
        raise ValueError("regime must be a RegimeResult")

    if not isinstance(result.regime, str) or not result.regime:
        raise ValueError("regime.regime must be a non-empty string")

    _validate_finite(
        result.trend_strength,
        "regime.trend_strength",
    )
    _validate_finite(
        result.volatility,
        "regime.volatility",
    )
    _validate_finite(
        result.return_mean,
        "regime.return_mean",
    )

    if result.trend_strength < 0.0:
        raise ValueError(
            "regime.trend_strength must be non-negative"
        )

    if result.volatility < 0.0:
        raise ValueError(
            "regime.volatility must be non-negative"
        )

    _validate_positive_integer(
        result.observations,
        "regime.observations",
    )

    return True


def _validate_regime_features(result: RegimeFeatures) -> bool:
    if not isinstance(result, RegimeFeatures):
        raise ValueError(
            "regime_features must be a RegimeFeatures"
        )

    for name in (
        "trend_strength",
        "volatility",
        "return_mean",
    ):
        _validate_finite(
            getattr(result, name),
            f"regime_features.{name}",
        )

    if result.trend_strength < 0.0:
        raise ValueError(
            "regime_features.trend_strength must be non-negative"
        )

    if result.volatility < 0.0:
        raise ValueError(
            "regime_features.volatility must be non-negative"
        )

    for name in (
        "is_trend_up",
        "is_trend_down",
        "is_high_volatility",
        "is_range",
    ):
        if not isinstance(getattr(result, name), bool):
            raise ValueError(
                f"regime_features.{name} must be boolean"
            )

    return True


def _validate_order_flow(result: OrderFlowResult) -> bool:
    if not isinstance(result, OrderFlowResult):
        raise ValueError(
            "order_flow must be an OrderFlowResult"
        )

    for name in (
        "buy_volume",
        "sell_volume",
        "total_volume",
        "imbalance",
        "buy_pressure",
        "sell_pressure",
    ):
        _validate_finite(
            getattr(result, name),
            f"order_flow.{name}",
        )

    if result.buy_volume < 0.0:
        raise ValueError(
            "order_flow.buy_volume must be non-negative"
        )

    if result.sell_volume < 0.0:
        raise ValueError(
            "order_flow.sell_volume must be non-negative"
        )

    if result.total_volume <= 0.0:
        raise ValueError(
            "order_flow.total_volume must be positive"
        )

    if not -1.0 <= result.imbalance <= 1.0:
        raise ValueError(
            "order_flow.imbalance must be between -1 and 1"
        )

    if not 0.0 <= result.buy_pressure <= 1.0:
        raise ValueError(
            "order_flow.buy_pressure must be between 0 and 1"
        )

    if not 0.0 <= result.sell_pressure <= 1.0:
        raise ValueError(
            "order_flow.sell_pressure must be between 0 and 1"
        )

    return True


def _validate_conditional(
    result: ConditionalEvaluationResult,
) -> bool:
    if not isinstance(result, ConditionalEvaluationResult):
        raise ValueError(
            "conditional must be a ConditionalEvaluationResult"
        )

    _validate_non_negative_integer(
        result.total_observations,
        "conditional.total_observations",
    )

    _validate_finite(
        result.overall_mean_forward_return,
        "conditional.overall_mean_forward_return",
    )

    _validate_probability(
        result.overall_positive_rate,
        "conditional.overall_positive_rate",
    )

    performance_fields = (
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
    )

    for name in performance_fields:
        performance = getattr(result, name)

        if not isinstance(performance, ConditionalPerformance):
            raise ValueError(
                f"conditional.{name} must be a ConditionalPerformance"
            )

        _validate_non_negative_integer(
            performance.observations,
            f"conditional.{name}.observations",
        )

        _validate_finite(
            performance.mean_forward_return,
            f"conditional.{name}.mean_forward_return",
        )

        _validate_probability(
            performance.positive_rate,
            f"conditional.{name}.positive_rate",
        )

    return True


def _validate_edge_validation(
    result: EdgeValidationResult,
) -> bool:
    if not isinstance(result, EdgeValidationResult):
        raise ValueError(
            "edge_validation must be an EdgeValidationResult"
        )

    _validate_non_negative_integer(
        result.observations,
        "edge_validation.observations",
    )

    for name in (
        "candidate_mean_return",
        "baseline_mean_return",
        "mean_return_uplift",
        "candidate_positive_rate",
        "baseline_positive_rate",
        "positive_rate_uplift",
        "minimum_mean_uplift",
        "minimum_positive_rate_uplift",
    ):
        _validate_finite(
            getattr(result, name),
            f"edge_validation.{name}",
        )

    _validate_probability(
        result.candidate_positive_rate,
        "edge_validation.candidate_positive_rate",
    )

    _validate_probability(
        result.baseline_positive_rate,
        "edge_validation.baseline_positive_rate",
    )

    _validate_positive_integer(
        result.minimum_observations,
        "edge_validation.minimum_observations",
    )

    if result.minimum_mean_uplift < 0.0:
        raise ValueError(
            "edge_validation.minimum_mean_uplift "
            "must be non-negative"
        )

    if result.minimum_positive_rate_uplift < 0.0:
        raise ValueError(
            "edge_validation.minimum_positive_rate_uplift "
            "must be non-negative"
        )

    if not isinstance(result.passed, bool):
        raise ValueError(
            "edge_validation.passed must be boolean"
        )

    return True


def _validate_advanced_model(
    result: AdvancedModelResult,
) -> bool:
    if not isinstance(result, AdvancedModelResult):
        raise ValueError(
            "advanced_model must be an AdvancedModelResult"
        )

    _validate_non_negative_integer(
        result.observations,
        "advanced_model.observations",
    )

    for name in (
        "candidate_mean_probability",
        "baseline_mean_probability",
        "mean_probability_uplift",
        "candidate_brier_score",
        "baseline_brier_score",
        "brier_score_improvement",
        "candidate_accuracy",
        "baseline_accuracy",
        "accuracy_uplift",
    ):
        _validate_finite(
            getattr(result, name),
            f"advanced_model.{name}",
        )

    _validate_probability(
        result.candidate_mean_probability,
        "advanced_model.candidate_mean_probability",
    )

    _validate_probability(
        result.baseline_mean_probability,
        "advanced_model.baseline_mean_probability",
    )

    _validate_probability(
        result.candidate_accuracy,
        "advanced_model.candidate_accuracy",
    )

    _validate_probability(
        result.baseline_accuracy,
        "advanced_model.baseline_accuracy",
    )

    if result.candidate_brier_score < 0.0:
        raise ValueError(
            "advanced_model.candidate_brier_score "
            "must be non-negative"
        )

    if result.baseline_brier_score < 0.0:
        raise ValueError(
            "advanced_model.baseline_brier_score "
            "must be non-negative"
        )

    if not isinstance(result.passed, bool):
        raise ValueError(
            "advanced_model.passed must be boolean"
        )

    return True


def _validate_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(f"{name} must be numeric")

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")


def _validate_probability(value: object, name: str) -> None:
    _validate_finite(value, name)

    numeric = float(value)

    if not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


def _validate_non_negative_integer(
    value: object,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer"
        )


def _validate_positive_integer(
    value: object,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a positive integer"
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be a positive integer"
        )
