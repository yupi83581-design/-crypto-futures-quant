"""Interface for loading historical market data."""

from abc import ABC, abstractmethod
from typing import Any


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
