"""Normalized market data record."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketRecord:
    """Standardized market data record."""

    symbol: str
    data_type: str
    event_time: datetime
    available_time: datetime
    ingestion_time: datetime
    payload: dict
