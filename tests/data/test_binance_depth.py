"""Tests for the read-only Binance Futures depth adapter."""

from __future__ import annotations

import json

from src.data.collector.binance_depth import BinancePublicDepthAdapter


def test_depth_adapter_normalizes_public_book() -> None:
    payload = {
        "bids": [["100.0", "2.5"], ["99.0", "1.0"]],
        "asks": [["101.0", "3.0"], ["102.0", "1.5"]],
    }

    adapter = BinancePublicDepthAdapter(
        http_get=lambda _url, _timeout: json.dumps(payload).encode("utf-8")
    )

    book = adapter.fetch_depth("BTCUSDT")

    assert book.bids[0].price == 100.0
    assert book.bids[0].quantity == 2.5
    assert book.asks[0].price == 101.0
    assert book.asks[0].quantity == 3.0
