"""
tests/integration/test_binance_live.py

Opt-in integration test: proves BinancePublicMarketDataAdapter genuinely
fetches LIVE data from Binance USDS-M Futures public API -- not mock,
not fixture, not fake transport. Skipped by default; requires network
and must be explicitly enabled via RUN_REAL_DATA_INTEGRATION=1, so the
default `pytest -q` run never depends on internet access.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from src.data.collector.adapter import BinancePublicMarketDataAdapter

RUN_INTEGRATION = os.environ.get("RUN_REAL_DATA_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set RUN_REAL_DATA_INTEGRATION=1 to run this live Binance integration test.",
)


def test_binance_live_fetch_returns_closed_5m_btcusdt_candles():
    # No http_get injected -> uses the adapter's real default transport
    # (urllib against the actual Binance public endpoint).
    adapter = BinancePublicMarketDataAdapter()

    now = datetime.now(timezone.utc)
    # 10-minute buffer before "now" guarantees every candle in this window
    # is already closed; 1-hour span reliably yields multiple 5m candles.
    end_time = now - timedelta(minutes=10)
    start_time = end_time - timedelta(hours=1)

    records = adapter.fetch_market_data(
        "BTCUSDT",
        "5m",
        start_time=start_time,
        end_time=end_time,
    )

    assert len(records) >= 1, "expected at least one live 5m candle from Binance"

    event_times = [r["event_time"] for r in records]
    assert event_times == sorted(event_times), "records must be sorted ascending by event_time"
    assert len(event_times) == len(set(event_times)), "no duplicate event_time allowed"

    for record in records:
        assert record["exchange"] == "BINANCE"
        assert record["market_type"] == "FUTURES"
        assert record["symbol"] == "BTCUSDT"
        assert record["data_type"] == "OHLCV"
        assert record["timeframe"] == "5m"

        assert record["open"] > 0
        assert record["high"] > 0
        assert record["low"] > 0
        assert record["close"] > 0
        assert record["volume"] >= 0
        assert record["high"] >= max(record["open"], record["close"], record["low"])
        assert record["low"] <= min(record["open"], record["close"], record["high"])

        assert record["event_time"] <= record["available_time"] <= record["ingestion_time"]

        available_time = datetime.fromisoformat(
            record["available_time"].replace("Z", "+00:00")
        )
        assert available_time <= now, "candle must already be closed, not in-progress"
