"""Real exchange integration for runtime -> snapshot -> HTTP.

This test is intentionally opt-in because public exchange access may be blocked
from CI networks. When enabled, it uses Binance public Futures klines only; no
fixture candles or synthetic market state are permitted.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

import pytest

from src.production.runtime import ProductionSnapshotRuntime, RuntimeConfig
from src.production.snapshot_http import create_snapshot_server


@pytest.mark.skipif(
    os.getenv("RUN_REAL_DATA_INTEGRATION") != "1",
    reason="real exchange integration disabled",
)
def test_real_binance_runtime_snapshot_http() -> None:
    config = RuntimeConfig(
        symbol=os.getenv("QUANT_SYMBOL", "BTCUSDT"),
        timeframe=os.getenv("QUANT_TIMEFRAME", "5m"),
        lookback_candles=int(os.getenv("QUANT_LOOKBACK_CANDLES", "240")),
        training_fraction=0.8,
        refresh_seconds=300.0,
        host="127.0.0.1",
        port=0,
    )
    runtime = ProductionSnapshotRuntime(config)

    inference = runtime.refresh_once()
    assert inference.observations > 0
    assert 0.0 <= inference.probability <= 1.0

    server = create_snapshot_server(
        "127.0.0.1",
        0,
        runtime.state.get,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(
            f"http://{host}:{port}/snapshot",
            timeout=5,
        ) as response:
            assert response.status == 200
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["status"] == "READY"
        assert payload["market_data"]["status"] == "LIVE"
        assert payload["market_data"]["provenance"] == "BINANCE_PUBLIC_FUTURES_KLINES"
        assert payload["probability"] == pytest.approx(inference.probability)
        assert payload["validation_status"] == "INSUFFICIENT_EVIDENCE"
        assert payload["risk"] == {"status": "UNAVAILABLE"}
        assert payload["decision"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
