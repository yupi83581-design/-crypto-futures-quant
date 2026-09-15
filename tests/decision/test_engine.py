from __future__ import annotations

import pytest

from src.decision.engine import (
    DecisionConfig,
    DecisionResult,
    make_decision,
)


def test_valid_evidence_is_approved():
    result = make_decision(
        probability=0.65,
        expected_value=0.02,
        risk_approved=True,
    )

    assert isinstance(result, DecisionResult)
    assert result.approved is True
    assert result.reason == "decision criteria passed"
    assert result.probability == pytest.approx(0.65)
    assert result.expected_value == pytest.approx(0.02)
    assert result.risk_approved is True


def test_probability_below_threshold_is_rejected():
    result = make_decision(
        probability=0.49,
        expected_value=0.02,
        risk_approved=True,
    )

    assert result.approved is False
    assert result.reason == "probability below minimum threshold"


def test_probability_equal_to_threshold_is_allowed():
    result = make_decision(
        probability=0.50,
        expected_value=0.02,
        risk_approved=True,
    )

    assert result.approved is True


def test_zero_expected_value_is_rejected():
    result = make_decision(
        probability=0.65,
        expected_value=0.0,
        risk_approved=True,
    )

    assert result.approved is False
    assert result.reason == "expected value below minimum threshold"


def test_negative_expected_value_is_rejected():
    result = make_decision(
        probability=0.65,
        expected_value=-0.01,
        risk_approved=True,
    )

    assert result.approved is False
    assert result.reason == "expected value below minimum threshold"


def test_risk_rejection_overrides_otherwise_valid_evidence():
    result = make_decision(
        probability=0.80,
        expected_value=0.05,
        risk_approved=False,
    )

    assert result.approved is False
    assert result.reason == "risk engine rejected decision"
    assert result.risk_approved is False


def test_custom_probability_threshold():
    config = DecisionConfig(
        min_probability=0.70,
        min_expected_value=0.0,
    )

    rejected = make_decision(
        probability=0.69,
        expected_value=0.02,
        risk_approved=True,
        config=config,
    )

    approved = make_decision(
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        config=config,
    )

    assert rejected.approved is False
    assert approved.approved is True


def test_custom_expected_value_threshold():
    config = DecisionConfig(
        min_probability=0.50,
        min_expected_value=0.02,
    )

    rejected = make_decision(
        probability=0.70,
        expected_value=0.019,
        risk_approved=True,
        config=config,
    )

    approved = make_decision(
        probability=0.70,
        expected_value=0.02,
        risk_approved=True,
        config=config,
    )

    assert rejected.approved is False
    assert approved.approved is True


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError):
        make_decision(
            probability=probability,
            expected_value=0.02,
            risk_approved=True,
        )


@pytest.mark.parametrize(
    "expected_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_expected_value_is_rejected(expected_value):
    with pytest.raises(ValueError):
        make_decision(
            probability=0.65,
            expected_value=expected_value,
            risk_approved=True,
        )


@pytest.mark.parametrize(
    "risk_approved",
    [
        1,
        0,
        "true",
        "false",
        None,
    ],
)
def test_risk_approval_must_be_boolean(risk_approved):
    with pytest.raises(ValueError):
        make_decision(
            probability=0.65,
            expected_value=0.02,
            risk_approved=risk_approved,
        )


@pytest.mark.parametrize(
    "config",
    [
        DecisionConfig(min_probability=-0.01),
        DecisionConfig(min_probability=1.01),
        DecisionConfig(min_expected_value=float("nan")),
        DecisionConfig(min_expected_value=float("inf")),
        DecisionConfig(min_expected_value=float("-inf")),
    ],
)
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        make_decision(
            probability=0.65,
            expected_value=0.02,
            risk_approved=True,
            config=config,
        )


def test_boolean_probability_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            probability=True,
            expected_value=0.02,
            risk_approved=True,
        )


def test_boolean_expected_value_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            probability=0.65,
            expected_value=False,
            risk_approved=True,
        )


def test_decision_result_preserves_input_evidence():
    result = make_decision(
        probability=0.73,
        expected_value=0.031,
        risk_approved=True,
    )

    assert result.probability == pytest.approx(0.73)
    assert result.expected_value == pytest.approx(0.031)
    assert result.risk_approved is True
