"""Exchange adapter interface."""

from abc import ABC, abstractmethod
from typing import Any


class ExchangeAdapter(ABC):
    """Abstract interface for exchange market data sources."""

    @abstractmethod
    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
    ) -> list[dict[str, Any]]:
        """Fetch market data for a symbol and timeframe."""
        raise NotImplementedError
