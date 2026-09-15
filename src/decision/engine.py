"""Deterministic decision engine for quantitative research."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds used by the decision engine."""

    min_probability: float = 0.50
    min_expected_value: float = 0.0


@dataclass(frozen=True)
class DecisionResult:
    """Final research decision produced from validated evidence."""

    approved: bool
    reason: str
    probability: float
    expected_value: float
    risk_approved: bool


def make_decision(
    *,
    probability: float,
    expected_value: float,
    risk_approved: bool,
    config: DecisionConfig | None = None,
) -> DecisionResult:
    """Convert model probability, EV, and risk status into a decision.

    This function is deterministic and read-only.

    A decision is approved only when:
    1. probability is valid and meets the configured threshold;
    2. expected value is valid and meets the configured threshold;
    3. the risk engine has approved the decision.

    No order placement or execution is performed.
    """
    if config is None:
        config = DecisionConfig()

    _validate_config(config)
    _validate_probability(probability)
    _validate_expected_value(expected_value)

    if not isinstance(risk_approved, bool):
        raise ValueError("risk_approved must be boolean")

    if probability < config.min_probability:
        return DecisionResult(
            approved=False,
            reason="probability below minimum threshold",
            probability=probability,
            expected_value=expected_value,
            risk_approved=risk_approved,
        )

    if expected_value <= config.min_expected_value:
        return DecisionResult(
            approved=False,
            reason="expected value below minimum threshold",
            probability=probability,
            expected_value=expected_value,
            risk_approved=risk_approved,
        )

    if not risk_approved:
        return DecisionResult(
            approved=False,
            reason="risk engine rejected decision",
            probability=probability,
            expected_value=expected_value,
            risk_approved=False,
        )

    return DecisionResult(
        approved=True,
        reason="decision criteria passed",
        probability=probability,
        expected_value=expected_value,
        risk_approved=True,
    )


def _validate_config(config: DecisionConfig) -> None:
    _validate_probability(
        config.min_probability,
        name="min_probability",
    )
    _validate_finite(
        config.min_expected_value,
        name="min_expected_value",
    )

    if not 0.0 <= config.min_probability <= 1.0:
        raise ValueError(
            "min_probability must be between 0 and 1"
        )


def _validate_probability(
    value: float,
    name: str = "probability",
) -> None:
    _validate_finite(value, name)

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


def _validate_expected_value(
    value: float,
) -> None:
    _validate_finite(value, "expected_value")


def _validate_finite(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{name} must be finite"
        )
