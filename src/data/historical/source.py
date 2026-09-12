"""Interface for historical market data sources, plus a concrete JSONL-file
implementation.
"""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class HistoricalDataSource(ABC):
    """Abstract interface for historical market data providers."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Fetch raw historical market data."""
        raise NotImplementedError


class JSONLHistoricalSource(HistoricalDataSource):
    """Reads historical market-data records from a single JSONL file."""

    def __init__(self, file_path: Union[str, Path]):
        self._file_path = Path(file_path)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with open(self._file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                if record.get("symbol") != symbol:
                    continue
                if record.get("timeframe") != timeframe:
                    continue
                event_time = record.get("event_time")
                if event_time is None:
                    continue
                if not (start_time <= event_time <= end_time):
                    continue
                results.append(record)
        return results
