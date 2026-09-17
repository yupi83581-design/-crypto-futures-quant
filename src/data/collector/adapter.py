"""Exchange market-data adapters for read-only quantitative research."""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.data.historical.source import HistoricalDataSource


class MarketDataError(Exception):
    """Base exception for market-data acquisition failures."""


class MarketDataConfigurationError(MarketDataError):
    """Raised when adapter configuration or request parameters are invalid."""


class MarketDataTimeoutError(MarketDataError):
    """Raised when an HTTP request times out after all retries."""


class MarketDataNetworkError(MarketDataError):
    """Raised when a non-timeout network failure occurs."""


class MarketDataHTTPError(MarketDataError):
    """Raised when the exchange returns an HTTP error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class MalformedMarketDataError(MarketDataError):
    """Raised when the exchange response violates the expected data contract."""


class ExchangeAdapter(ABC):
    """Abstract interface for exchange market-data sources."""

    @abstractmethod
    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch normalized market data for a symbol and timeframe."""
        raise NotImplementedError


TIMEFRAME_MILLISECONDS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}


class BinancePublicMarketDataAdapter(
    HistoricalDataSource,
    ExchangeAdapter,
):
    """Read-only Binance USDⓈ-M Futures OHLCV adapter.

    This adapter uses Binance's public klines endpoint only. It never handles
    private credentials and never places orders.
    """

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    EXCHANGE = "BINANCE"
    MARKET_TYPE = "FUTURES"
    DATA_TYPE = "OHLCV"
    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.25,
        page_limit: int = 1000,
        http_get: Callable[[str, float], bytes] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise MarketDataConfigurationError(
                f"timeout_seconds must be finite and > 0, got {timeout_seconds!r}"
            )

        if max_retries < 0:
            raise MarketDataConfigurationError(
                f"max_retries must be >= 0, got {max_retries!r}"
            )

        if retry_delay_seconds < 0 or not math.isfinite(retry_delay_seconds):
            raise MarketDataConfigurationError(
                "retry_delay_seconds must be finite and >= 0, "
                f"got {retry_delay_seconds!r}"
            )

        if page_limit <= 0 or page_limit > 1500:
            raise MarketDataConfigurationError(
                f"page_limit must be between 1 and 1500, got {page_limit!r}"
            )

        self.base_url = base_url or self.BASE_URL
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.page_limit = int(page_limit)
        self._http_get = http_get or self._default_http_get
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Implement HistoricalDataSource for DefaultHistoricalDataLoader."""
        return self.fetch_market_data(
            symbol,
            timeframe,
            start_time=start_time,
            end_time=end_time,
        )

    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch fully closed Binance OHLCV candles in [start, end)."""

        normalized_symbol = self._validate_symbol(symbol)
        interval_ms = self._validate_timeframe(timeframe)

        start_dt = self._parse_datetime(start_time, "start_time")
        end_dt = self._parse_datetime(end_time, "end_time")

        if start_dt is None or end_dt is None:
            raise MarketDataConfigurationError(
                "start_time and end_time are required"
            )

        if start_dt >= end_dt:
            raise MarketDataConfigurationError(
                "start_time must be earlier than end_time"
            )

        ingestion_time = self._utc_now()

        if end_dt > ingestion_time:
            raise MarketDataConfigurationError(
                "end_time cannot be in the future"
            )

        start_ms = self._datetime_to_ms(start_dt)
        end_ms = self._datetime_to_ms(end_dt)

        records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            url = self._build_url(
                symbol=normalized_symbol,
                timeframe=timeframe,
                start_ms=cursor_ms,
                end_ms=end_ms,
            )

            payload = self._request_json(url)

            if not isinstance(payload, list):
                raise MalformedMarketDataError(
                    "Binance response must be a JSON array"
                )

            if not payload:
                break

            page_open_times: list[int] = []

            for raw_kline in payload:
                canonical = self._canonical_record(
                    raw_kline=raw_kline,
                    symbol=normalized_symbol,
                    timeframe=timeframe,
                    interval_ms=interval_ms,
                    ingestion_time=ingestion_time,
                )

                if canonical is None:
                    continue

                record, open_time_ms = canonical

                if open_time_ms < start_ms:
                    continue

                if open_time_ms >= end_ms:
                    continue

                key = (
                    normalized_symbol,
                    timeframe,
                    open_time_ms,
                )

                existing = records_by_key.get(key)

                if existing is None:
                    records_by_key[key] = record
                elif not self._records_equal(existing, record):
                    raise MalformedMarketDataError(
                        "conflicting duplicate candle for "
                        f"{normalized_symbol} {timeframe} "
                        f"event_time={record['event_time']}"
                    )

                page_open_times.append(open_time_ms)

            raw_open_times = [
                self._extract_open_time(item) for item in payload
            ]

            if not raw_open_times:
                raise MalformedMarketDataError(
                    "Binance response contained no valid candle timestamps"
                )

            page_last_open_ms = max(raw_open_times)

            if page_last_open_ms < cursor_ms:
                raise MalformedMarketDataError(
                    "Binance pagination cursor did not advance"
                )

            next_cursor_ms = page_last_open_ms + interval_ms

            if next_cursor_ms <= cursor_ms:
                raise MalformedMarketDataError(
                    "Binance pagination produced a non-advancing cursor"
                )

            cursor_ms = next_cursor_ms

            if len(payload) < self.page_limit:
                break

        records = sorted(
            records_by_key.values(),
            key=lambda record: (
                record["event_time"],
                record["symbol"],
                record["timeframe"],
            ),
        )

        return records

    def _canonical_record(
        self,
        *,
        raw_kline: Any,
        symbol: str,
        timeframe: str,
        interval_ms: int,
        ingestion_time: datetime,
    ) -> tuple[dict[str, Any], int] | None:
        if not isinstance(raw_kline, list):
            raise MalformedMarketDataError(
                "each Binance kline must be an array"
            )

        if len(raw_kline) < 6:
            raise MalformedMarketDataError(
                "Binance kline must contain at least 6 fields"
            )

        open_time_ms = self._parse_integer(
            raw_kline[0],
            "kline open time",
        )

        close_time_ms = self._parse_integer(
            raw_kline[6],
            "kline close time",
        ) if len(raw_kline) > 6 else open_time_ms + interval_ms - 1

        if close_time_ms < open_time_ms:
            raise MalformedMarketDataError(
                "kline close time must not precede open time"
            )

        expected_close_time_ms = open_time_ms + interval_ms - 1

        if close_time_ms != expected_close_time_ms:
            raise MalformedMarketDataError(
                "kline close time does not match requested timeframe"
            )

        open_price = self._parse_float(raw_kline[1], "open")
        high_price = self._parse_float(raw_kline[2], "high")
        low_price = self._parse_float(raw_kline[3], "low")
        close_price = self._parse_float(raw_kline[4], "close")
        volume = self._parse_float(raw_kline[5], "volume")

        if open_price <= 0:
            raise MalformedMarketDataError("open price must be > 0")

        if high_price <= 0 or low_price <= 0 or close_price <= 0:
            raise MalformedMarketDataError(
                "OHLC prices must all be > 0"
            )

        if high_price < max(open_price, close_price, low_price):
            raise MalformedMarketDataError(
                "high price is lower than an OHLC value"
            )

        if low_price > min(open_price, close_price, high_price):
            raise MalformedMarketDataError(
                "low price is higher than an OHLC value"
            )

        if volume < 0:
            raise MalformedMarketDataError("volume must be >= 0")

        close_time = self._ms_to_datetime(close_time_ms)
        available_time = close_time + timedelta(milliseconds=1)

        if available_time > ingestion_time:
            return None

        event_time = self._ms_to_datetime(open_time_ms)

        record = {
            "exchange": self.EXCHANGE,
            "market_type": self.MARKET_TYPE,
            "symbol": symbol,
            "data_type": self.DATA_TYPE,
            "event_time": self._format_datetime(event_time),
            "available_time": self._format_datetime(available_time),
            "ingestion_time": self._format_datetime(ingestion_time),
            "schema_version": self.SCHEMA_VERSION,
            "timeframe": timeframe,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }

        return record, open_time_ms

    def _request_json(self, url: str) -> Any:
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            try:
                raw_response = self._http_get(url, self.timeout_seconds)

                try:
                    return json.loads(raw_response.decode("utf-8"))
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    raise MalformedMarketDataError(
                        "Binance response is not valid JSON"
                    ) from exc

            except HTTPError as exc:
                raise MarketDataHTTPError(
                    exc.code,
                    exc.reason or "HTTP request failed",
                ) from exc

            except TimeoutError as exc:
                if attempt >= self.max_retries:
                    raise MarketDataTimeoutError(
                        "Binance request timed out after "
                        f"{attempts} attempts"
                    ) from exc

            except URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, TimeoutError):
                    if attempt >= self.max_retries:
                        raise MarketDataTimeoutError(
                            "Binance request timed out after "
                            f"{attempts} attempts"
                        ) from exc
                else:
                    raise MarketDataNetworkError(
                        f"Binance network request failed: {reason}"
                    ) from exc

            except OSError as exc:
                raise MarketDataNetworkError(
                    f"Binance network request failed: {exc}"
                ) from exc

            if attempt < self.max_retries:
                self._sleep(self.retry_delay_seconds)

        raise MarketDataError("Binance request failed unexpectedly")

    @staticmethod
    def _default_http_get(url: str, timeout: float) -> bytes:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "crypto-futures-quant/1.0",
            },
            method="GET",
        )

        with urlopen(request, timeout=timeout) as response:
            return response.read()

    def _build_url(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> str:
        return (
            f"{self.base_url}"
            f"?symbol={symbol}"
            f"&interval={timeframe}"
            f"&startTime={start_ms}"
            f"&endTime={end_ms - 1}"
            f"&limit={self.page_limit}"
        )

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise MarketDataConfigurationError(
                "symbol must be a string"
            )

        normalized = symbol.strip().upper()

        if not normalized:
            raise MarketDataConfigurationError(
                "symbol must not be empty"
            )

        if "/" in normalized or any(
            character.isspace() for character in normalized
        ):
            raise MarketDataConfigurationError(
                "symbol must be Binance-style without separators"
            )

        return normalized

    @staticmethod
    def _validate_timeframe(timeframe: str) -> int:
        if not isinstance(timeframe, str):
            raise MarketDataConfigurationError(
                "timeframe must be a string"
            )

        if timeframe == "1M":
            raise MarketDataConfigurationError(
                "1M is not supported because a calendar month is not "
                "a fixed millisecond interval"
            )

        interval_ms = TIMEFRAME_MILLISECONDS.get(timeframe)

        if interval_ms is None:
            raise MarketDataConfigurationError(
                f"unsupported timeframe: {timeframe!r}"
            )

        return interval_ms

    @staticmethod
    def _parse_datetime(
        value: str | datetime | None,
        field_name: str,
    ) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise MarketDataConfigurationError(
                    f"{field_name} must include timezone information"
                )
            return value.astimezone(timezone.utc)

        if not isinstance(value, str):
            raise MarketDataConfigurationError(
                f"{field_name} must be an ISO-8601 datetime string"
            )

        text = value.strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise MarketDataConfigurationError(
                f"{field_name} must be a valid ISO-8601 datetime"
            ) from exc

        if parsed.tzinfo is None:
            raise MarketDataConfigurationError(
                f"{field_name} must include timezone information"
            )

        return parsed.astimezone(timezone.utc)

    def _utc_now(self) -> datetime:
        now = self._clock()

        if now.tzinfo is None:
            raise MarketDataConfigurationError(
                "adapter clock must return a timezone-aware datetime"
            )

        return now.astimezone(timezone.utc)

    @staticmethod
    def _datetime_to_ms(value: datetime) -> int:
        return int(value.timestamp() * 1000)

    @staticmethod
    def _ms_to_datetime(value: int) -> datetime:
        return datetime.fromtimestamp(
            value / 1000,
            tz=timezone.utc,
        )

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")

    @staticmethod
    def _parse_integer(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise MalformedMarketDataError(
                f"{field_name} must be an integer"
            )

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise MalformedMarketDataError(
                f"{field_name} must be an integer"
            ) from exc

        if isinstance(value, float) and value != parsed:
            raise MalformedMarketDataError(
                f"{field_name} must be an integer"
            )

        return parsed

    @staticmethod
    def _parse_float(value: Any, field_name: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise MalformedMarketDataError(
                f"{field_name} must be numeric"
            ) from exc

        if not math.isfinite(parsed):
            raise MalformedMarketDataError(
                f"{field_name} must be finite"
            )

        return parsed

    @staticmethod
    def _extract_open_time(raw_kline: Any) -> int:
        if not isinstance(raw_kline, list) or not raw_kline:
            raise MalformedMarketDataError(
                "Binance response contains an invalid kline"
            )

        return BinancePublicMarketDataAdapter._parse_integer(
            raw_kline[0],
            "kline open time",
        )

    @staticmethod
    def _records_equal(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> bool:
        comparable_fields = (
            "exchange",
            "market_type",
            "symbol",
            "data_type",
            "event_time",
            "available_time",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        return all(
            first.get(field) == second.get(field)
            for field in comparable_fields
        )
