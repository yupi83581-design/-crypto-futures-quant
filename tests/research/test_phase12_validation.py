"""Tests for the final Phase 12 research-stack validation."""

from dataclasses import FrozenInstanceError

import pytest

from src.models.advanced import AdvancedModelResult
from src.research.conditional import (
    ConditionalEvaluationResult,
    ConditionalPerformance,
)
from src.research.edge_validation import EdgeValidationResult
from src.research.order_flow import OrderFlowResult
from src.research.phase12_validation import (
    Phase12ValidationResult,
    validate_phase12_stack,
)
from src.research.regime import RegimeResult
from src.research.regime_features import RegimeFeatures


def _valid_regime() -> RegimeResult:
    return RegimeResult(
        regime="TREND_UP",
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )


def _valid_regime_features() -> RegimeFeatures:
    return RegimeFeatures(
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        is_trend_up=True,
        is_trend_down=False,
        is_high_volatility=False,
        is_range=False,
    )


def _valid_order_flow() -> OrderFlowResult:
    return OrderFlowResult(
        buy_volume=60.0,
        sell_volume=40.0,
        total_volume=100.0,
        imbalance=0.20,
        buy_pressure=0.60,
        sell_pressure=0.40,
    )


def _valid_performance() -> ConditionalPerformance:
    return ConditionalPerformance(
        observations=10,
        mean_forward_return=0.01,
        positive_rate=0.60,
    )


def _valid_conditional() -> ConditionalEvaluationResult:
    performance = _valid_performance()

    return ConditionalEvaluationResult(
        total_observations=120,
        overall_mean_forward_return=0.002,
        overall_positive_rate=0.53,
        trend_up_buy_dominant=performance,
        trend_up_balanced=performance,
        trend_up_sell_dominant=performance,
        trend_down_buy_dominant=performance,
        trend_down_balanced=performance,
        trend_down_sell_dominant=performance,
        high_volatility_buy_dominant=performance,
        high_volatility_balanced=performance,
        high_volatility_sell_dominant=performance,
        range_buy_dominant=performance,
        range_balanced=performance,
        range_sell_dominant=performance,
    )


def _valid_edge_validation() -> EdgeValidationResult:
    return EdgeValidationResult(
        observations=100,
        candidate_mean_return=0.012,
        baseline_mean_return=0.006,
        mean_return_uplift=0.006,
        candidate_positive_rate=0.58,
        baseline_positive_rate=0.52,
        positive_rate_uplift=0.06,
        minimum_observations=30,
        minimum_mean_uplift=0.001,
        minimum_positive_rate_uplift=0.02,
        passed=True,
    )


def _valid_advanced_model() -> AdvancedModelResult:
    return AdvancedModelResult(
        observations=100,
        candidate_mean_probability=0.58,
        baseline_mean_probability=0.52,
        mean_probability_uplift=0.06,
        candidate_brier_score=0.18,
        baseline_brier_score=0.22,
        brier_score_improvement=0.04,
        candidate_accuracy=0.70,
        baseline_accuracy=0.64,
        accuracy_uplift=0.06,
        passed=True,
    )


def _valid_inputs() -> dict:
    return {
        "regime": _valid_regime(),
        "regime_features": _valid_regime_features(),
        "order_flow": _valid_order_flow(),
        "conditional": _valid_conditional(),
        "edge_validation": _valid_edge_validation(),
        "advanced_model": _valid_advanced_model(),
    }


def test_valid_phase12_stack_passes():
    result = validate_phase12_stack(**_valid_inputs())

    assert isinstance(result, Phase12ValidationResult)
    assert result.regime_valid is True
    assert result.regime_features_valid is True
    assert result.order_flow_valid is True
    assert result.conditional_valid is True
    assert result.edge_validation_valid is True
    assert result.advanced_model_valid is True
    assert result.passed is True


def test_validation_is_deterministic():
    inputs = _valid_inputs()

    first = validate_phase12_stack(**inputs)
    second = validate_phase12_stack(**inputs)

    assert first == second


def test_result_is_immutable():
    result = validate_phase12_stack(**_valid_inputs())

    with pytest.raises(FrozenInstanceError):
        result.passed = False


@pytest.mark.parametrize(
    "field_name",
    [
        "regime",
        "regime_features",
        "order_flow",
        "conditional",
        "edge_validation",
        "advanced_model",
    ],
)
def test_rejects_wrong_component_type(field_name):
    inputs = _valid_inputs()
    inputs[field_name] = object()

    with pytest.raises(ValueError):
        validate_phase12_stack(**inputs)


def test_rejects_empty_regime_name():
    inputs = _valid_inputs()
    inputs["regime"] = RegimeResult(
        regime="",
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )

    with pytest.raises(ValueError, match="non-empty string"):
        validate_phase12_stack(**inputs)


def test_rejects_negative_regime_trend_strength():
    inputs = _valid_inputs()
    inputs["regime"] = RegimeResult(
        regime="TREND_UP",
        trend_strength=-0.01,
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )

    with pytest.raises(ValueError, match="trend_strength"):
        validate_phase12_stack(**inputs)


def test_rejects_negative_regime_volatility():
    inputs = _valid_inputs()
    inputs["regime"] = RegimeResult(
        regime="TREND_UP",
        trend_strength=0.05,
        volatility=-0.01,
        return_mean=0.002,
        observations=20,
    )

    with pytest.raises(ValueError, match="volatility"):
        validate_phase12_stack(**inputs)


def test_rejects_non_positive_regime_observations():
    inputs = _valid_inputs()
    inputs["regime"] = RegimeResult(
        regime="TREND_UP",
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        observations=0,
    )

    with pytest.raises(ValueError, match="observations"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_regime_feature_boolean():
    inputs = _valid_inputs()
    inputs["regime_features"] = RegimeFeatures(
        trend_strength=0.05,
        volatility=0.01,
        return_mean=0.002,
        is_trend_up=1,
        is_trend_down=False,
        is_high_volatility=False,
        is_range=False,
    )

    with pytest.raises(ValueError, match="is_trend_up"):
        validate_phase12_stack(**inputs)


def test_rejects_negative_order_flow_volume():
    inputs = _valid_inputs()
    inputs["order_flow"] = OrderFlowResult(
        buy_volume=-1.0,
        sell_volume=40.0,
        total_volume=39.0,
        imbalance=-1.05,
        buy_pressure=0.0,
        sell_pressure=1.0,
    )

    with pytest.raises(ValueError, match="buy_volume"):
        validate_phase12_stack(**inputs)


def test_rejects_zero_order_flow_total_volume():
    inputs = _valid_inputs()
    inputs["order_flow"] = OrderFlowResult(
        buy_volume=0.0,
        sell_volume=0.0,
        total_volume=0.0,
        imbalance=0.0,
        buy_pressure=0.0,
        sell_pressure=0.0,
    )

    with pytest.raises(ValueError, match="total_volume"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_order_flow_imbalance():
    inputs = _valid_inputs()
    inputs["order_flow"] = OrderFlowResult(
        buy_volume=60.0,
        sell_volume=40.0,
        total_volume=100.0,
        imbalance=1.5,
        buy_pressure=0.60,
        sell_pressure=0.40,
    )

    with pytest.raises(ValueError, match="imbalance"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_conditional_positive_rate():
    inputs = _valid_inputs()
    performance = ConditionalPerformance(
        observations=10,
        mean_forward_return=0.01,
        positive_rate=1.5,
    )

    inputs["conditional"] = ConditionalEvaluationResult(
        total_observations=120,
        overall_mean_forward_return=0.002,
        overall_positive_rate=0.53,
        trend_up_buy_dominant=performance,
        trend_up_balanced=_valid_performance(),
        trend_up_sell_dominant=_valid_performance(),
        trend_down_buy_dominant=_valid_performance(),
        trend_down_balanced=_valid_performance(),
        trend_down_sell_dominant=_valid_performance(),
        high_volatility_buy_dominant=_valid_performance(),
        high_volatility_balanced=_valid_performance(),
        high_volatility_sell_dominant=_valid_performance(),
        range_buy_dominant=_valid_performance(),
        range_balanced=_valid_performance(),
        range_sell_dominant=_valid_performance(),
    )

    with pytest.raises(ValueError, match="positive_rate"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_edge_positive_rate():
    inputs = _valid_inputs()
    inputs["edge_validation"] = EdgeValidationResult(
        observations=100,
        candidate_mean_return=0.012,
        baseline_mean_return=0.006,
        mean_return_uplift=0.006,
        candidate_positive_rate=1.2,
        baseline_positive_rate=0.52,
        positive_rate_uplift=0.68,
        minimum_observations=30,
        minimum_mean_uplift=0.001,
        minimum_positive_rate_uplift=0.02,
        passed=True,
    )

    with pytest.raises(ValueError, match="candidate_positive_rate"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_edge_minimum_observations():
    inputs = _valid_inputs()
    inputs["edge_validation"] = EdgeValidationResult(
        observations=100,
        candidate_mean_return=0.012,
        baseline_mean_return=0.006,
        mean_return_uplift=0.006,
        candidate_positive_rate=0.58,
        baseline_positive_rate=0.52,
        positive_rate_uplift=0.06,
        minimum_observations=0,
        minimum_mean_uplift=0.001,
        minimum_positive_rate_uplift=0.02,
        passed=True,
    )

    with pytest.raises(ValueError, match="minimum_observations"):
        validate_phase12_stack(**inputs)


def test_rejects_invalid_advanced_probability():
    inputs = _valid_inputs()
    inputs["advanced_model"] = AdvancedModelResult(
        observations=100,
        candidate_mean_probability=1.20,
        baseline_mean_probability=0.52,
        mean_probability_uplift=0.68,
        candidate_brier_score=0.18,
        baseline_brier_score=0.22,
        brier_score_improvement=0.04,
        candidate_accuracy=0.70,
        baseline_accuracy=0.64,
        accuracy_uplift=0.06,
        passed=True,
    )

    with pytest.raises(ValueError, match="candidate_mean_probability"):
        validate_phase12_stack(**inputs)


def test_rejects_negative_advanced_brier_score():
    inputs = _valid_inputs()
    inputs["advanced_model"] = AdvancedModelResult(
        observations=100,
        candidate_mean_probability=0.58,
        baseline_mean_probability=0.52,
        mean_probability_uplift=0.06,
        candidate_brier_score=-0.01,
        baseline_brier_score=0.22,
        brier_score_improvement=0.23,
        candidate_accuracy=0.70,
        baseline_accuracy=0.64,
        accuracy_uplift=0.06,
        passed=True,
    )

    with pytest.raises(ValueError, match="candidate_brier_score"):
        validate_phase12_stack(**inputs)


def test_rejects_nan_numeric_value():
    inputs = _valid_inputs()
    inputs["regime"] = RegimeResult(
        regime="TREND_UP",
        trend_strength=float("nan"),
        volatility=0.01,
        return_mean=0.002,
        observations=20,
    )

    with pytest.raises(ValueError, match="finite"):
        validate_phase12_stack(**inputs)


def test_rejects_infinite_numeric_value():
    inputs = _valid_inputs()
    inputs["order_flow"] = OrderFlowResult(
        buy_volume=float("inf"),
        sell_volume=40.0,
        total_volume=float("inf"),
        imbalance=0.20,
        buy_pressure=0.60,
        sell_pressure=0.40,
    )

    with pytest.raises(ValueError, match="finite"):
        validate_phase12_stack(**inputs)
