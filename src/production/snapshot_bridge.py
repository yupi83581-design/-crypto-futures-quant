"""Read-only aggregation bridge for the Quant Command Center."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from src.production.command_center_contract import (
    QuantCommandCenterSnapshot,
    build_snapshot,
)


_UNAVAILABLE = {"status": "UNAVAILABLE"}


def _payload(value: Any) -> dict[str, Any]:
    """Serialize an existing result object without calculating new metrics."""
    if value is None:
        return dict(_UNAVAILABLE)
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(
        "snapshot source must be a mapping, dataclass result, or None"
    )


def _value_or_none(value: Any, name: str) -> Any:
    """Read one already-computed field; never derive a replacement value."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def build_command_center_snapshot(
    *,
    symbol: str,
    status: str,
    market_data: Any = None,
    inference: Any = None,
    calibration: Any = None,
    expected_value_cost: Any = None,
    risk: Any = None,
    decision: Any = None,
    backtest_oos: Any = None,
    walk_forward: Any = None,
    robustness: Any = None,
    paper_trading: Any = None,
    journal_monitoring: Any = None,
    final_validation: Any = None,
    validation_status: str = "INSUFFICIENT_EVIDENCE",
) -> QuantCommandCenterSnapshot:
    """Aggregate existing runtime/research results into the read-only contract.

    This function deliberately performs no quant calculations. Missing domains
    remain explicitly unavailable instead of being represented as numeric zeroes.
    """
    if market_data is None and inference is not None:
        market_payload: dict[str, Any] = {}
        observations = _value_or_none(inference, "observations")
        usable = _value_or_none(inference, "usable_observations")
        timeframe = _value_or_none(inference, "timeframe")
        if observations is not None:
            market_payload["observations"] = observations
        if usable is not None:
            market_payload["usable_observations"] = usable
        if timeframe is not None:
            market_payload["timeframe"] = timeframe
        if not market_payload:
            market_payload = dict(_UNAVAILABLE)
    else:
        market_payload = _payload(market_data)

    probability = _value_or_none(inference, "probability")
    decision_value = None
    if decision is not None:
        approved = _value_or_none(decision, "approved")
        if approved is not None:
            decision_value = "APPROVED" if approved else "REJECTED"

    return build_snapshot(
        symbol=symbol,
        status=status,
        market_data=market_payload,
        probability=probability,
        calibration=_payload(calibration),
        expected_value_cost=_payload(expected_value_cost),
        risk=_payload(risk),
        decision=decision_value,
        backtest_oos=_payload(backtest_oos),
        walk_forward=_payload(walk_forward),
        robustness=_payload(robustness),
        paper_trading=_payload(paper_trading),
        journal_monitoring=_payload(journal_monitoring),
        final_validation=_payload(final_validation),
        validation_status=validation_status,
    )


def build_from_orchestrator_result(
    result: Any,
    *,
    status: str,
    market_data: Any = None,
    calibration: Any = None,
    expected_value_cost: Any = None,
    backtest_oos: Any = None,
    walk_forward: Any = None,
    robustness: Any = None,
    final_validation: Any = None,
    validation_status: str = "INSUFFICIENT_EVIDENCE",
) -> QuantCommandCenterSnapshot:
    """Bridge an existing ProductionOrchestrator result without rerunning it."""
    required = ("symbol", "model_output", "decision", "risk")
    if result is None or any(not hasattr(result, name) for name in required):
        raise ValueError("incomplete orchestrator result")

    probability = _value_or_none(result.model_output, "probability")
    inference = {"probability": probability} if probability is not None else None

    return build_command_center_snapshot(
        symbol=result.symbol,
        status=status,
        market_data=market_data,
        inference=inference,
        calibration=calibration,
        expected_value_cost=expected_value_cost,
        risk=result.risk,
        decision=result.decision,
        backtest_oos=backtest_oos,
        walk_forward=walk_forward,
        robustness=robustness,
        paper_trading=result.paper_result,
        journal_monitoring=result.monitoring_result,
        final_validation=final_validation,
        validation_status=validation_status,
    )
