"""Configuration for market data collection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectorConfig:
    """Immutable configuration for a market data collector."""

    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
