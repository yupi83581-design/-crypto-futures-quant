"""
Production inference boundary.

Pipeline:

real market data
    -> RSI feature
    -> fitted probability model
    -> probability output

This module performs inference only.
It does not train models and does not place exchange orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class InferenceInput:
    """Market-data request for one inference cycle."""

    symbol: str
    timeframe: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class InferenceResult:
    """Immutable result of one inference cycle."""

    symbol: str
    timeframe: str
    observations: int
    usable_observations: int
    latest_rsi: float
    probability: float


class ProductionInferenceEngine:
    """
    Connect real market data to a fitted probability model.

    The model must already be fitted before this engine is created.
    Training is deliberately outside this production inference boundary.
    """

    def __init__(
        self,
        *,
        market_data_adapter: Any,
        model: Any,
        feature_builder: Callable[
            [list[dict[str, Any]]],
            Sequence[float | None],
        ],
    ) -> None:
        if not hasattr(market_data_adapter, "fetch_market_data"):
            raise TypeError(
                "market_data_adapter must provide fetch_market_data"
            )

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "model must provide predict_proba"
            )

        if not callable(feature_builder):
            raise TypeError(
                "feature_builder must be callable"
            )

        self._market_data_adapter = market_data_adapter
        self._model = model
        self._feature_builder = feature_builder

    def run(self, data: InferenceInput) -> InferenceResult:
        """Fetch real market data and produce one probability output."""
        self._validate_input(data)

        records = self._market_data_adapter.fetch_market_data(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start_time=data.start_time,
            end_time=data.end_time,
        )

        if not isinstance(records, list):
            raise TypeError(
                "market_data_adapter.fetch_market_data must return a list"
            )

        if not records:
            raise ValueError(
                "market_data_adapter returned no market data"
            )

        features = self._feature_builder(records)

        if not isinstance(features, Sequence):
            raise TypeError(
                "feature_builder must return a sequence"
            )

        usable_features = [
            float(value)
            for value in features
            if value is not None
        ]

        if not usable_features:
            raise ValueError(
                "feature_builder produced no usable features"
            )

        for value in usable_features:
            if not math.isfinite(value):
                raise ValueError(
                    "features must contain only finite values"
                )

        probabilities = self._model.predict_proba(
            usable_features
        )

        if not isinstance(probabilities, Sequence):
            raise TypeError(
                "model.predict_proba must return a sequence"
            )

        if len(probabilities) != len(usable_features):
            raise ValueError(
                "model probability count must match usable feature count"
            )

        validated_probabilities = [
            self._validate_probability(value)
            for value in probabilities
        ]

        latest_rsi = usable_features[-1]
        probability = validated_probabilities[-1]

        return InferenceResult(
            symbol=data.symbol,
            timeframe=data.timeframe,
            observations=len(records),
            usable_observations=len(usable_features),
            latest_rsi=latest_rsi,
            probability=probability,
        )

    @staticmethod
    def _validate_input(data: InferenceInput) -> None:
        if not isinstance(data, InferenceInput):
            raise TypeError(
                "data must be an InferenceInput"
            )

        if not isinstance(data.symbol, str):
            raise TypeError(
                "symbol must be a string"
            )

        if not data.symbol.strip():
            raise ValueError(
                "symbol must be a non-empty string"
            )

        if not isinstance(data.timeframe, str):
            raise TypeError(
                "timeframe must be a string"
            )

        if not data.timeframe.strip():
            raise ValueError(
                "timeframe must be a non-empty string"
            )

        if not isinstance(data.start_time, str):
            raise TypeError(
                "start_time must be a string"
            )

        if not isinstance(data.end_time, str):
            raise TypeError(
                "end_time must be a string"
            )

        if not data.start_time.strip():
            raise ValueError(
                "start_time must be a non-empty string"
            )

        if not data.end_time.strip():
            raise ValueError(
                "end_time must be a non-empty string"
            )

    @staticmethod
    def _validate_probability(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(
                "probability must be numeric"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                "probability must be numeric"
            )

        probability = float(value)

        if not math.isfinite(probability):
            raise ValueError(
                "probability must be finite"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        return probability
