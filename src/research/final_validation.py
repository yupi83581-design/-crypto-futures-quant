"""Final acceptance gate for the quantitative research engine."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

REQUIRED_STAGES = (
    "data_quality",
    "model",
    "calibration",
    "ev_cost",
    "risk",
    "backtest",
    "walk_forward",
    "oos",
    "regime",
    "robustness",
    "paper_trading",
    "monitoring",
    "trading_performance",
    "dsr",
    "pbo_cscv",
    "lookahead_protection",
    "untouched_final_test",
    "risk_controls",
    "kill_switch",
    "state_persistence",
    "evidence_integrity",
    "reproducibility",
    "execution_lock",
    "security",
)

@dataclass(frozen=True)
class FinalValidationResult:
    status: GateStatus
    stage_status: tuple[tuple[str, GateStatus], ...]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS


def evaluate_final_gate(evidence: Mapping[str, GateStatus | str]) -> FinalValidationResult:
    missing = [s for s in REQUIRED_STAGES if s not in evidence]
    if missing:
        statuses = tuple((k, _coerce(v)) for k, v in evidence.items())
        return FinalValidationResult(GateStatus.INSUFFICIENT_EVIDENCE, statuses, (f"missing evidence: {', '.join(missing)}",))
    statuses = tuple((s, _coerce(evidence[s])) for s in REQUIRED_STAGES)
    failed = [s for s, status in statuses if status is GateStatus.FAIL]
    insufficient = [s for s, status in statuses if status is GateStatus.INSUFFICIENT_EVIDENCE]
    if failed: return FinalValidationResult(GateStatus.FAIL, statuses, tuple(f"failed stage: {s}" for s in failed))
    if insufficient: return FinalValidationResult(GateStatus.INSUFFICIENT_EVIDENCE, statuses, tuple(f"insufficient evidence: {s}" for s in insufficient))
    return FinalValidationResult(GateStatus.PASS, statuses, ())


def _coerce(value: GateStatus | str) -> GateStatus:
    try: return value if isinstance(value, GateStatus) else GateStatus(value)
    except ValueError as exc: raise ValueError(f"invalid gate status: {value!r}") from exc
