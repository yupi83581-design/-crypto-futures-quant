"""Validation helpers for market data collector configuration."""

from .config import CollectorConfig


def validate_config(config: CollectorConfig) -> None:
    """Validate collector configuration."""
    if not config.symbols:
        raise ValueError("symbols must not be empty")

    if not config.timeframes:
        raise ValueError("timeframes must not be empty")
