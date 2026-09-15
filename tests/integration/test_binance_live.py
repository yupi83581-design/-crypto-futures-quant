from datetime import datetime, timedelta, timezone

from src.data.collector.adapter import BinancePublicMarketDataAdapter


def test_binance_futures_live_market_data():
    """
    Live integration test.

    This test intentionally uses the adapter's real HTTP transport.
    No mock, fixture, or fake response is supplied.
    """

    now = datetime.now(timezone.utc)

    # Align to the beginning of the current 5-minute candle.
    end = now.replace(
        minute=(now.minute // 5) * 5,
        second=0,
        microsecond=0,
    )

    # Request the previous fully closed 5-minute candle.
    start = end - timedelta(minutes=5)

    adapter = BinancePublicMarketDataAdapter()

    records = adapter.fetch_market_data(
        symbol="BTCUSDT",
        timeframe="5m",
        start=start,
        end=end,
    )

    assert records, "Binance returned no market-data records."

    record = records[0]

    assert record.symbol == "BTCUSDT"
    assert record.timeframe == "5m"
    assert record.open_time < record.close_time

    assert record.open > 0
    assert record.high > 0
    assert record.low > 0
    assert record.close > 0
    assert record.volume >= 0
