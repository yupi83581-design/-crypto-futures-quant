"""Interface for loading historical market data, plus a concrete adapter
implementation over a HistoricalDataSource.
"""
from abc import ABC, abstractmethod
from typing import Any

from src.data.historical.source import HistoricalDataSource


class HistoricalDataLoader(ABC):
    """Abstract interface for historical market data sources."""

    @abstractmethod
    def load(
        self,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Load historical market data for the requested scope."""
        raise NotImplementedError


class DefaultHistoricalDataLoader(HistoricalDataLoader):
    """Adapts any HistoricalDataSource into the HistoricalDataLoader
    interface: fans out over every (symbol, timeframe) pair requested,
    calls the source once per pair, and merges results deterministically
    (sorted by event_time, then symbol, then timeframe -- never by
    insertion order or wall-clock time, so replay is reproducible).
    """

    def __init__(self, source: HistoricalDataSource):
        self._source = source

    def load(
        self,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for symbol in symbols:
            for timeframe in timeframes:
                records.extend(
                    self._source.fetch(symbol, timeframe, start_time, end_time)
                )
        records.sort(
            key=lambda r: (
                r.get("event_time", ""),
                r.get("symbol", ""),
                r.get("timeframe", ""),
            )
        )
        return records
