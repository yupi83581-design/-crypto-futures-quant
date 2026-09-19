"""Deterministic production paper-cycle composition.

This module connects already-existing research components for one virtual
long-only cycle. It never places exchange orders. Missing market-integrity
evidence blocks approval rather than treating missing evidence as NORMAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.decision.engine import DecisionConfig, DecisionResult, make_decision
from src.market_integrity.detector import MarketIntegrityResult
from src.models.expected_value import ExpectedValueResult, calculate_expected_value
from src.monitoring.journal import Journal, JournalEntry
from src.paper.engine import PaperPosition, PaperTradingEngine
from src.risk.engine import RiskConfig, RiskResult, assess_risk


@dataclass(frozen=True)
class PaperCycleConfig:
    reward_multiple: float = 2.0
    reward_fraction: float = 0.02
    fee: float = 0.0
    slippage: float = 0.0
    decision: DecisionConfig = DecisionConfig()
    risk: RiskConfig = RiskConfig()


@dataclass(frozen=True)
class PaperCycleResult:
    probability: float
    expected_value: ExpectedValueResult | None
    risk: RiskResult | None
    decision: DecisionResult
    paper_position: PaperPosition | None
    journal_entry: JournalEntry
    market_integrity: MarketIntegrityResult | None


class ProductionPaperCycle:
    """Compose EV, integrity, risk, decision, paper execution and journal."""

    def __init__(
        self,
        *,
        paper_engine: PaperTradingEngine,
        journal: Journal,
        config: PaperCycleConfig | None = None,
    ) -> None:
        self.paper_engine = paper_engine
        self.journal = journal
        self.config = config or PaperCycleConfig()
        self._validate_config()

    def run(
        self,
        *,
        symbol: str,
        probability: float,
        entry_price: float,
        stop_price: float,
        signal: int = 1,
        market_integrity: MarketIntegrityResult | None = None,
    ) -> PaperCycleResult:
        """Evaluate one cycle and open a virtual long only when all gates pass."""
        if self.paper_engine.position is not None:
            blocked = DecisionResult(
                approved=False,
                reason="paper position already open",
                probability=float(probability),
                expected_value=0.0,
                risk_approved=False,
            )
            entry = self.journal.record(
                symbol=symbol,
                signal=signal,
                probability=probability,
                expected_value=0.0,
                risk_approved=False,
                approved=False,
            )
            return PaperCycleResult(
                probability=float(probability),
                expected_value=None,
                risk=None,
                decision=blocked,
                paper_position=self.paper_engine.position,
                journal_entry=entry,
                market_integrity=market_integrity,
            )

        if market_integrity is None:
            decision = make_decision(
                probability=probability,
                expected_value=0.0,
                risk_approved=False,
                config=self.config.decision,
            )
            blocked = DecisionResult(
                approved=False,
                reason="market integrity evidence unavailable",
                probability=decision.probability,
                expected_value=decision.expected_value,
                risk_approved=False,
            )
            entry = self.journal.record(
                symbol=symbol,
                signal=signal,
                probability=probability,
                expected_value=0.0,
                risk_approved=False,
                approved=False,
                entry_price=None,
            )
            return PaperCycleResult(
                probability=float(probability),
                expected_value=None,
                risk=None,
                decision=blocked,
                paper_position=None,
                journal_entry=entry,
                market_integrity=None,
            )

        if market_integrity.status != "NORMAL":
            blocked = DecisionResult(
                approved=False,
                reason="market integrity gate rejected cycle",
                probability=float(probability),
                expected_value=0.0,
                risk_approved=False,
            )
            entry = self.journal.record(
                symbol=symbol,
                signal=signal,
                probability=probability,
                expected_value=0.0,
                risk_approved=False,
                approved=False,
            )
            return PaperCycleResult(
                probability=float(probability),
                expected_value=None,
                risk=None,
                decision=blocked,
                paper_position=None,
                journal_entry=entry,
                market_integrity=market_integrity,
            )

        reward = float(entry_price) * self.config.reward_fraction
        loss = float(entry_price - stop_price)
        expected_value = calculate_expected_value(
            probability=probability,
            reward=reward,
            loss=loss,
            fee=self.config.fee,
            slippage=self.config.slippage,
        )
        risk = assess_risk(
            equity=self.paper_engine.equity,
            entry_price=entry_price,
            stop_price=stop_price,
            net_expected_value=expected_value.net_expected_value,
            config=self.config.risk,
        )
        decision = make_decision(
            probability=probability,
            expected_value=expected_value.net_expected_value,
            risk_approved=risk.approved,
            config=self.config.decision,
        )

        position = None
        if decision.approved:
            position = self.paper_engine.open_long(
                entry_price=entry_price,
                quantity=risk.position_notional / entry_price,
            )

        entry = self.journal.record(
            symbol=symbol,
            signal=signal,
            probability=probability,
            expected_value=expected_value.net_expected_value,
            risk_approved=risk.approved,
            approved=decision.approved,
            entry_price=entry_price if position else None,
        )

        return PaperCycleResult(
            probability=float(probability),
            expected_value=expected_value,
            risk=risk,
            decision=decision,
            paper_position=position,
            journal_entry=entry,
            market_integrity=market_integrity,
        )

    def _validate_config(self) -> None:
        if self.config.reward_multiple <= 0.0:
            raise ValueError("reward_multiple must be positive")
        if self.config.reward_fraction <= 0.0:
            raise ValueError("reward_fraction must be positive")
        if self.config.fee < 0.0 or self.config.slippage < 0.0:
            raise ValueError("fee and slippage must be non-negative")
