"""Base interface for market data collectors."""

from abc import ABC, abstractmethod
from typing import Any


class MarketDataCollector(ABC):
    """Abstract interface for market data collectors."""

    @abstractmethod
    def collect(self) -> list[dict[str, Any]]:
        """Collect normalized market data records."""
        raise NotImplementedError
