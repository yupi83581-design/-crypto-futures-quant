"""Unit tests for runtime lifecycle behavior without market-data fixtures."""

from __future__ import annotations

import threading
import urllib.request
import urllib.error

from src.production.runtime import ProductionSnapshotRuntime, RuntimeConfig


def test_runtime_starts_with_explicit_offline_state() -> None:
    runtime = ProductionSnapshotRuntime(
        RuntimeConfig(
            lookback_candles=40,
            host="127.0.0.1",
            port=0,
        )
    )

    snapshot = runtime.state.get()

    assert snapshot.status == "ERROR"
    assert snapshot.market_data["status"] == "OFFLINE"
    assert snapshot.probability is None
    assert snapshot.risk == {"status": "UNAVAILABLE"}
    assert snapshot.decision is None
    assert snapshot.validation_status == "INSUFFICIENT_EVIDENCE"


def test_runtime_http_serves_current_state_without_fabrication() -> None:
    runtime = ProductionSnapshotRuntime(
        RuntimeConfig(
            lookback_candles=40,
            host="127.0.0.1",
            port=0,
        )
    )
    server = runtime._server = __import__(
        "src.production.snapshot_http",
        fromlist=["create_snapshot_server"],
    ).create_snapshot_server(
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
            body = response.read().decode("utf-8")

        assert '"status":"ERROR"' in body
        assert '"status":"OFFLINE"' in body
        assert '"probability":null' in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_runtime_freshness_gate_blocks_stale_market_data():
    from datetime import datetime, timezone

    from src.production.runtime import _validate_freshness

    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    records = [{
        "event_time": "2026-09-23T11:45:00.000Z",
        "available_time": "2026-09-23T11:45:00.001Z",
    }]

    with __import__("pytest").raises(RuntimeError, match="market data is stale"):
        _validate_freshness(
            records,
            now=now,
            interval_seconds=300,
            max_stale_intervals=2,
        )


def test_runtime_freshness_gate_accepts_recent_closed_data():
    from datetime import datetime, timezone

    from src.production.runtime import _validate_freshness

    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    records = [{
        "event_time": "2026-09-23T11:55:00.000Z",
        "available_time": "2026-09-23T11:55:00.001Z",
    }]

    _validate_freshness(
        records,
        now=now,
        interval_seconds=300,
        max_stale_intervals=2,
    )


def test_snapshot_endpoint_rejects_order_like_mutations():
    runtime = ProductionSnapshotRuntime(
        RuntimeConfig(
            lookback_candles=40,
            host="127.0.0.1",
            port=0,
        )
    )
    server = runtime._server = __import__(
        "src.production.snapshot_http",
        fromlist=["create_snapshot_server"],
    ).create_snapshot_server(
        "127.0.0.1",
        0,
        runtime.state.get,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = urllib.request.Request(
            f"http://{host}:{port}/snapshot",
            method="POST",
        )
        with __import__("pytest").raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
