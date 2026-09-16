"""
Production orchestration layer.

Connects the research components into one deterministic pipeline:

real market data
    -> feature builder
    -> model
    -> decision
    -> risk
    -> paper execution
    -> monitoring

This module does NOT place real exchange orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class OrchestratorInput:
    """Validated input supplied to one production research cycle."""

    symbol: str
    prices: Sequence[float]
    market_data: Any = None

    def __post_init__(self) -> None:
        """Validate the input immediately when the object is created."""
        ProductionOrchestrator._validate_input(self)


@dataclass(frozen=True)
class OrchestratorResult:
    """Immutable result of one orchestration cycle."""

    symbol: str
    features: Any
    model_output: Any
    decision: Any
    risk: Any
    paper_result: Any = None
    monitoring_result: Any = None


class ProductionOrchestrator:
    """
    Coordinates the existing research components.

    The orchestrator owns pipeline sequencing only. Feature engineering,
    modelling, decision logic, risk logic, paper execution, and monitoring
    remain separate components.

    Real market data is supplied through ``OrchestratorInput.market_data``
    and/or ``prices``. This layer does not fabricate market data and does
    not place real exchange orders.
    """

    def __init__(
        self,
        *,
        feature_builder: Callable[[OrchestratorInput], Any],
        model: Callable[[Any], Any],
        decision_engine: Callable[[Any], Any],
        risk_engine: Callable[[Any], Any],
        paper_engine: Callable[[Any], Any] | None = None,
        monitor: Callable[[Any], Any] | None = None,
    ) -> None:
        self._feature_builder = self._validate_callable(
            feature_builder,
            "feature_builder",
        )
        self._model = self._validate_callable(
            model,
            "model",
        )
        self._decision_engine = self._validate_callable(
            decision_engine,
            "decision_engine",
        )
        self._risk_engine = self._validate_callable(
            risk_engine,
            "risk_engine",
        )

        if paper_engine is not None:
            self._validate_callable(
                paper_engine,
                "paper_engine",
            )

        if monitor is not None:
            self._validate_callable(
                monitor,
                "monitor",
            )

        self._paper_engine = paper_engine
        self._monitor = monitor

    def run(self, data: OrchestratorInput) -> OrchestratorResult:
        """
        Run exactly one deterministic research/paper cycle.

        Pipeline:

        input
            -> features
            -> model
            -> decision
            -> risk
            -> optional paper execution
            -> optional monitoring
        """
        self._validate_input(data)

        features = self._feature_builder(data)

        model_output = self._model(features)

        decision = self._decision_engine(model_output)

        risk = self._risk_engine(decision)

        paper_result = None
        if self._paper_engine is not None:
            paper_result = self._paper_engine(risk)

        monitoring_result = None
        if self._monitor is not None:
            monitoring_result = self._monitor(
                {
                    "input": data,
                    "features": features,
                    "model_output": model_output,
                    "decision": decision,
                    "risk": risk,
                    "paper_result": paper_result,
                }
            )

        return OrchestratorResult(
            symbol=data.symbol,
            features=features,
            model_output=model_output,
            decision=decision,
            risk=risk,
            paper_result=paper_result,
            monitoring_result=monitoring_result,
        )

    @staticmethod
    def _validate_callable(
        value: Callable[..., Any] | None,
        name: str,
    ) -> Callable[..., Any]:
        """Validate that a pipeline dependency is callable."""
        if not callable(value):
            raise TypeError(f"{name} must be callable")

        return value

    @staticmethod
    def _validate_input(data: OrchestratorInput) -> None:
        """Validate orchestration input before any pipeline stage runs."""
        if not isinstance(data, OrchestratorInput):
            raise TypeError("data must be an OrchestratorInput")

        if not isinstance(data.symbol, str):
            raise TypeError("symbol must be a string")

        if not data.symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        if isinstance(data.prices, (str, bytes)):
            raise TypeError("prices must be a numeric sequence")

        try:
            values = list(data.prices)
        except TypeError as exc:
            raise TypeError(
                "prices must be a numeric sequence"
            ) from exc

        if not values:
            raise ValueError("prices must not be empty")

        for price in values:
            if isinstance(price, bool):
                raise TypeError(
                    "prices must contain numeric values"
                )

            if not isinstance(price, (int, float)):
                raise TypeError(
                    "prices must contain numeric values"
                )

            if not math.isfinite(float(price)):
                raise ValueError(
                    "prices must contain finite numeric values"
                )

            if price <= 0:
                raise ValueError(
                    "prices must contain positive values"
                )
