"""Tests for the Binance public market-data adapter."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.data.collector.adapter import (
    BinancePublicMarketDataAdapter,
    MalformedMarketDataError,
    MarketDataNetworkError,
    MarketDataTimeoutError,
)


UTC = timezone.utc


def kline(
    open_ms: int,
    *,
    open_price: str = "100.0",
    high_price: str = "105.0",
    low_price: str = "95.0",
    close_price: str = "102.0",
    volume: str = "10.0",
):
    return [
        open_ms,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        open_ms + 299_999,
        "1000.0",
        10,
        "5.0",
        "500.0",
        "0",
    ]


def json_response(payload):
    """Return the bytes contract expected from the HTTP transport."""
    return json.dumps(payload).encode("utf-8")


def test_fetch_market_data_paginates_and_returns_canonical_records():
    pages = [
        [
            kline(0),
            kline(300_000, open_price="102.0"),
        ],
        [
            kline(600_000, open_price="103.0"),
        ],
    ]

    calls = []

    def fake_http_get(url, timeout):
        calls.append((url, timeout))
        return json_response(pages.pop(0))

    adapter = BinancePublicMarketDataAdapter(
        http_get=fake_http_get,
        page_limit=2,
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=datetime(1970, 1, 1, tzinfo=UTC),
        end_time=datetime(1970, 1, 1, 0, 15, tzinfo=UTC),
    )

    assert len(records) == 3
    assert [record["event_time"] for record in records] == [
        datetime(1970, 1, 1, 0, 0, tzinfo=UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        datetime(1970, 1, 1, 0, 5, tzinfo=UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        datetime(1970, 1, 1, 0, 10, tzinfo=UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
    ]
    assert len(calls) == 2


def test_identical_duplicate_is_deduplicated():
    payload = [
        kline(0),
        kline(0),
        kline(300_000),
    ]

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(payload),
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=datetime(1970, 1, 1, tzinfo=UTC),
        end_time=datetime(1970, 1, 1, 0, 10, tzinfo=UTC),
    )

    assert len(records) == 2


def test_conflicting_duplicate_is_rejected():
    payload = [
        kline(0, close_price="102.0"),
        kline(0, close_price="103.0"),
    ]

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(payload),
    )

    with pytest.raises(MalformedMarketDataError):
        adapter.fetch_market_data(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=datetime(1970, 1, 1, tzinfo=UTC),
            end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
        )


def test_incomplete_latest_candle_is_excluded():
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    open_time = now - timedelta(minutes=2)

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(
            [kline(int(open_time.timestamp() * 1000))]
        ),
        clock=lambda: now,
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=open_time - timedelta(minutes=5),
        end_time=now,
    )

    assert records == []


def test_closed_candle_is_accepted():
    ingestion_time = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response([kline(0)]),
        clock=lambda: ingestion_time,
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=datetime(1970, 1, 1, tzinfo=UTC),
        end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
    )

    assert len(records) == 1
    assert records[0]["event_time"] == "1970-01-01T00:00:00.000Z"


def test_invalid_ohlc_relationship_is_rejected():
    payload = [
        kline(
            0,
            open_price="100.0",
            high_price="99.0",
            low_price="95.0",
            close_price="98.0",
        )
    ]

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(payload),
    )

    with pytest.raises(MalformedMarketDataError):
        adapter.fetch_market_data(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=datetime(1970, 1, 1, tzinfo=UTC),
            end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
        )


def test_negative_volume_is_rejected():
    payload = [
        kline(0, volume="-1.0"),
    ]

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(payload),
    )

    with pytest.raises(MalformedMarketDataError):
        adapter.fetch_market_data(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=datetime(1970, 1, 1, tzinfo=UTC),
            end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
        )


def test_available_time_is_not_after_ingestion_time():
    ingestion_time = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response([kline(0)]),
        clock=lambda: ingestion_time,
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=datetime(1970, 1, 1, tzinfo=UTC),
        end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
    )

    record = records[0]

    assert record["event_time"] <= record["available_time"]
    assert record["available_time"] <= record["ingestion_time"]


def test_timeout_is_classified_separately():
    def fake_http_get(url, timeout):
        raise TimeoutError("timed out")

    adapter = BinancePublicMarketDataAdapter(
        http_get=fake_http_get,
        max_retries=0,
    )

    with pytest.raises(MarketDataTimeoutError):
        adapter.fetch_market_data(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=datetime(1970, 1, 1, tzinfo=UTC),
            end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
        )


def test_network_error_is_classified_separately():
    from urllib.error import URLError

    def fake_http_get(url, timeout):
        raise URLError("network unavailable")

    adapter = BinancePublicMarketDataAdapter(
        http_get=fake_http_get,
        max_retries=0,
    )

    with pytest.raises(MarketDataNetworkError):
        adapter.fetch_market_data(
            symbol="BTCUSDT",
            timeframe="5m",
            start_time=datetime(1970, 1, 1, tzinfo=UTC),
            end_time=datetime(1970, 1, 1, 0, 5, tzinfo=UTC),
        )


def test_records_are_sorted_by_event_time():
    payload = [
        kline(300_000),
        kline(0),
        kline(600_000),
    ]

    adapter = BinancePublicMarketDataAdapter(
        http_get=lambda url, timeout: json_response(payload),
    )

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start_time=datetime(1970, 1, 1, tzinfo=UTC),
        end_time=datetime(1970, 1, 1, 0, 15, tzinfo=UTC),
    )

    event_times = [record["event_time"] for record in records]

    assert event_times == sorted(event_times)
