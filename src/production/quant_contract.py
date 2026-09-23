"""Explicit production quant-cycle contract from probability to paper decision."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from src.models.expected_value import ExpectedValueResult, calculate_expected_value
from src.risk.engine import RiskConfig, RiskResult, assess_risk
from src.decision.engine import DecisionConfig, DecisionResult, make_decision

@dataclass(frozen=True)
class QuantCycleResult:
    probability: float
    expected_value: ExpectedValueResult
    risk: RiskResult
    decision: DecisionResult

class ProductionQuantCycle:
    """Connect calibrated probability -> cost-aware EV -> risk -> decision.

    Inputs are already fitted/validated research artifacts. No training,
    exchange execution, or future-data access occurs in this cycle.
    """
    def __init__(self, *, calibrator: Any | None = None, risk_config: RiskConfig | None = None, decision_config: DecisionConfig | None = None) -> None:
        if calibrator is not None and not callable(getattr(calibrator, "predict", None)): raise TypeError("calibrator must provide predict")
        self.calibrator=calibrator; self.risk_config=risk_config; self.decision_config=decision_config

    def evaluate(self, *, raw_probability: float, reward: float, loss: float, fee: float, slippage: float, equity: float, entry_price: float, stop_price: float, current_drawdown_fraction: float=0.0) -> QuantCycleResult:
        probability=float(self.calibrator.predict([raw_probability])[0]) if self.calibrator else float(raw_probability)
        ev=calculate_expected_value(probability,reward,loss,fee,slippage)
        risk=assess_risk(equity=equity,entry_price=entry_price,stop_price=stop_price,net_expected_value=ev.net_expected_value,current_drawdown_fraction=current_drawdown_fraction,config=self.risk_config)
        decision=make_decision(probability=probability,expected_value=ev.net_expected_value,risk_approved=risk.approved,config=self.decision_config)
        return QuantCycleResult(probability,ev,risk,decision)
