"""Interface for historical market data sources."""

from abc import ABC, abstractmethod
from typing import Any


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
