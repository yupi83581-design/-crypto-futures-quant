"""Presentation-neutral read-only contract for the future Quant Command Center.

This module intentionally contains no quant logic. It exposes a stable,
JSON-serializable snapshot boundary so a dashboard remains a consumer of
engine evidence rather than a second implementation of the research engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


MetricMap = Mapping[str, Any]


@dataclass(frozen=True)
class QuantCommandCenterSnapshot:
    symbol: str
    status: str
    market_data: MetricMap = field(default_factory=dict)
    probability: float | None = None
    calibration: MetricMap = field(default_factory=dict)
    expected_value_cost: MetricMap = field(default_factory=dict)
    risk: MetricMap = field(default_factory=dict)
    decision: str | None = None
    backtest_oos: MetricMap = field(default_factory=dict)
    walk_forward: MetricMap = field(default_factory=dict)
    robustness: MetricMap = field(default_factory=dict)
    paper_trading: MetricMap = field(default_factory=dict)
    journal_monitoring: MetricMap = field(default_factory=dict)
    final_validation: MetricMap = field(default_factory=dict)
    validation_status: str = "INSUFFICIENT_EVIDENCE"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, read-only presentation snapshot."""
        return asdict(self)


def build_snapshot(
    *,
    symbol: str,
    status: str,
    market_data: MetricMap | None = None,
    probability: float | None = None,
    calibration: MetricMap | None = None,
    expected_value_cost: MetricMap | None = None,
    risk: MetricMap | None = None,
    decision: str | None = None,
    backtest_oos: MetricMap | None = None,
    walk_forward: MetricMap | None = None,
    robustness: MetricMap | None = None,
    paper_trading: MetricMap | None = None,
    journal_monitoring: MetricMap | None = None,
    final_validation: MetricMap | None = None,
    validation_status: str = "INSUFFICIENT_EVIDENCE",
    # Compatibility fields retained for existing consumers.
    paper_equity: float | None = None,
    observations: int | None = None,
) -> QuantCommandCenterSnapshot:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if status not in {"READY", "NO_TRADE", "ERROR"}:
        raise ValueError("status must be READY, NO_TRADE, or ERROR")
    if validation_status not in {"PASS", "FAIL", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError("invalid validation_status")
    if probability is not None and not 0 <= float(probability) <= 1:
        raise ValueError("probability must be between 0 and 1")
    if observations is not None and observations < 0:
        raise ValueError("observations must be non-negative")

    paper_payload = dict(paper_trading or {})
    if paper_equity is not None:
        paper_payload.setdefault("equity", paper_equity)
    if observations is not None:
        market_payload = dict(market_data or {})
        market_payload.setdefault("observations", observations)
    else:
        market_payload = dict(market_data or {})

    return QuantCommandCenterSnapshot(
        symbol=symbol,
        status=status,
        market_data=market_payload,
        probability=probability,
        calibration=dict(calibration or {}),
        expected_value_cost=dict(expected_value_cost or {}),
        risk=dict(risk or {}),
        decision=decision,
        backtest_oos=dict(backtest_oos or {}),
        walk_forward=dict(walk_forward or {}),
        robustness=dict(robustness or {}),
        paper_trading=paper_payload,
        journal_monitoring=dict(journal_monitoring or {}),
        final_validation=dict(final_validation or {}),
        validation_status=validation_status,
    )
