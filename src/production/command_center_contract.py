"""Presentation-neutral data contract for the future Quant Command Center.

This module intentionally contains no quant logic. It exposes a stable,
JSON-serializable snapshot boundary so a dashboard remains a consumer rather
than a second implementation of the research engine.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class QuantCommandCenterSnapshot:
    symbol: str
    status: str
    probability: float | None
    decision: str | None
    risk_approved: bool | None
    paper_equity: float | None
    observations: int | None
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_snapshot(*, symbol: str, status: str, probability: float | None = None, decision: str | None = None, risk_approved: bool | None = None, paper_equity: float | None = None, observations: int | None = None, validation_status: str = "INSUFFICIENT_EVIDENCE") -> QuantCommandCenterSnapshot:
    if not isinstance(symbol, str) or not symbol.strip(): raise ValueError("symbol must be a non-empty string")
    if status not in {"READY", "NO_TRADE", "ERROR"}: raise ValueError("status must be READY, NO_TRADE, or ERROR")
    if validation_status not in {"PASS", "FAIL", "INSUFFICIENT_EVIDENCE"}: raise ValueError("invalid validation_status")
    if probability is not None and not 0 <= float(probability) <= 1: raise ValueError("probability must be between 0 and 1")
    if observations is not None and observations < 0: raise ValueError("observations must be non-negative")
    return QuantCommandCenterSnapshot(symbol, status, probability, decision, risk_approved, paper_equity, observations, validation_status)
