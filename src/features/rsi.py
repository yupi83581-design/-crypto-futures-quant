"""
src/features/rsi.py

RSI(14) - Phase 2, Feature Engine.

Computes the Relative Strength Index using Wilder's original smoothing
(RMA). Operates on the flat-dict record contract verified in Phase 1
(DefaultHistoricalDataLoader.load() output) -- list[dict] with numeric
"close" and string "event_time" (ascending). No dependency on
MarketRecord: no adapter to it has been verified to exist yet.
"""
from __future__ import annotations

from typing import Any, Optional

DEFAULT_PERIOD = 14


def compute_rsi(
    records: list[dict[str, Any]],
    period: int = DEFAULT_PERIOD,
) -> list[Optional[float]]:
    """Computes Wilder's RSI for a chronologically ordered OHLCV series.

    Returns a list the same length as `records`. Entry i depends only on
    records[0..i] -- never future records. Entries before enough history
    has accumulated (i < period) are None.

    Degenerate-case convention (documented, not left implicit):
        - avg_gain == 0 and avg_loss == 0: RSI = 50.0 (neutral).
        - avg_loss == 0 and avg_gain > 0: RSI = 100.0.

    Raises ValueError for: invalid period, missing/non-numeric close,
    missing event_time, or records not sorted ascending by event_time.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    n = len(records)
    if n == 0:
        return []

    closes: list[float] = []
    event_times: list[str] = []
    for idx, record in enumerate(records):
        if "close" not in record:
            raise ValueError(f"record at index {idx} is missing required 'close' field")
        close = record["close"]
        if isinstance(close, bool) or not isinstance(close, (int, float)):
            raise ValueError(f"record at index {idx} has non-numeric 'close': {close!r}")
        closes.append(float(close))

        if "event_time" not in record or record["event_time"] is None:
            raise ValueError(f"record at index {idx} is missing required 'event_time' field")
        event_times.append(record["event_time"])

    for i in range(1, n):
        if event_times[i] < event_times[i - 1]:
            raise ValueError(
                "records must be sorted ascending by event_time: "
                f"index {i - 1} ({event_times[i - 1]!r}) comes after "
                f"index {i} ({event_times[i]!r})"
            )

    rsi_values: list[Optional[float]] = [None] * n
    if n <= period:
        return rsi_values

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains[i] = max(delta, 0.0)
        losses[i] = max(-delta, 0.0)

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    rsi_values[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi_values[i] = _rsi_from_averages(avg_gain, avg_loss)

    return rsi_values


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
