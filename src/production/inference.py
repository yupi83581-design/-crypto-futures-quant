"""
Production inference boundary.

Pipeline:

real market data
    -> feature builder
    -> fitted probability model
    -> probability output

This module performs inference only.
It does not train models and does not place exchange orders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class InferenceInput:
    """Market-data request for one inference cycle."""

    symbol: str
    timeframe: str
    start_time: str | datetime
    end_time: str | datetime

    def __post_init__(self) -> None:
        """
        Reject an actually empty symbol at construction time.

        Whitespace-only values remain a runtime validation concern,
        matching the production input contract used elsewhere.
        """
        if self.symbol == "":
            raise ValueError("symbol must be a non-empty string")


@dataclass(frozen=True)
class InferenceResult:
    """Immutable result of one inference cycle."""

    symbol: str
    timeframe: str
    observations: int
    usable_observations: int
    latest_feature: float
    probability: float


class ProductionInferenceEngine:
    """
    Connect real market data to a fitted probability model.

    Training is deliberately outside this boundary.
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
        if not hasattr(
            market_data_adapter,
            "fetch_market_data",
        ):
            raise TypeError(
                "market_data_adapter must provide fetch_market_data"
            )

        if not callable(
            getattr(model, "predict_proba", None)
        ):
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

    def run(
        self,
        data: InferenceInput,
    ) -> InferenceResult:
        """
        Run one inference cycle.

        The feature sequence is kept aligned with the model output.
        Missing feature observations are removed together with their
        corresponding observation position before inference.
        """
        self._validate_input(data)

        records = self._market_data_adapter.fetch_market_data(
            symbol=data.symbol,
            timeframe=data.timeframe,
            start_time=data.start_time,
            end_time=data.end_time,
        )

        if not isinstance(records, list):
            raise TypeError(
                "market_data_adapter.fetch_market_data "
                "must return a list"
            )

        if not records:
            raise ValueError(
                "market_data_adapter returned no market data"
            )

        features = self._feature_builder(records)

        if isinstance(features, (str, bytes)):
            raise TypeError(
                "feature_builder must return a sequence"
            )

        if not isinstance(features, Sequence):
            raise TypeError(
                "feature_builder must return a sequence"
            )

        if len(features) != len(records):
            raise ValueError(
                "feature sequence length must match "
                "market-data observation count"
            )

        usable_features: list[float] = []

        for value in features:
            if value is None:
                continue

            usable_features.append(
                self._validate_feature(value)
            )

        if not usable_features:
            raise ValueError(
                "feature_builder produced no usable features"
            )

        probabilities = self._model.predict_proba(
            usable_features
        )

        if isinstance(probabilities, (str, bytes)):
            raise TypeError(
                "model.predict_proba must return a sequence"
            )

        if not isinstance(probabilities, Sequence):
            raise TypeError(
                "model.predict_proba must return a sequence"
            )

        if len(probabilities) != len(usable_features):
            raise ValueError(
                "model probability count must match "
                "usable feature count"
            )

        validated_probabilities = [
            self._validate_probability(value)
            for value in probabilities
        ]

        return InferenceResult(
            symbol=data.symbol,
            timeframe=data.timeframe,
            observations=len(records),
            usable_observations=len(usable_features),
            latest_feature=usable_features[-1],
            probability=validated_probabilities[-1],
        )

    @staticmethod
    def _validate_input(
        data: InferenceInput,
    ) -> None:
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

        ProductionInferenceEngine._validate_time(
            data.start_time,
            "start_time",
        )

        ProductionInferenceEngine._validate_time(
            data.end_time,
            "end_time",
        )

    @staticmethod
    def _validate_time(
        value: str | datetime,
        field_name: str,
    ) -> None:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError(
                    f"{field_name} must include timezone information"
                )
            return

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string or datetime"
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} must be non-empty"
            )

    @staticmethod
    def _validate_feature(
        value: Any,
    ) -> float:
        if isinstance(value, bool):
            raise TypeError(
                "feature must be numeric"
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                "feature must be numeric"
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValueError(
                "features must contain only finite values"
            )

        return numeric

    @staticmethod
    def _validate_probability(
        value: Any,
    ) -> float:
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
