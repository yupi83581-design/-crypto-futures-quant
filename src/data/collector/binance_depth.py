"""Read-only Binance USDⓈ-M Futures order-book adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.market_integrity.detector import OrderBookLevel


class BinanceDepthError(Exception):
    """Raised when public Binance depth data cannot be acquired or validated."""


@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]


class BinancePublicDepthAdapter:
    """Fetch public Binance Futures depth; never creates or cancels orders."""

    BASE_URL = "https://fapi.binance.com/fapi/v1/depth"

    def __init__(self, *, base_url: str | None = None, timeout_seconds: float = 10.0,
                 limit: int = 20, http_get: Callable[[str, float], bytes] | None = None) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and > 0")
        if limit < 5 or limit > 1000:
            raise ValueError("limit must be between 5 and 1000")
        self.base_url = base_url or self.BASE_URL
        self.timeout_seconds = float(timeout_seconds)
        self.limit = int(limit)
        self._http_get = http_get or self._default_http_get

    def fetch_depth(self, symbol: str) -> OrderBookSnapshot:
        normalized = symbol.strip().upper()
        if not normalized or "/" in normalized or any(c.isspace() for c in normalized):
            raise ValueError("symbol must be Binance-style without separators")
        url = f"{self.base_url}?symbol={normalized}&limit={self.limit}"
        try:
            payload = json.loads(self._http_get(url, self.timeout_seconds).decode("utf-8"))
        except HTTPError as exc:
            raise BinanceDepthError(f"HTTP {exc.code}: {exc.reason or 'depth request failed'}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BinanceDepthError(f"depth network request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BinanceDepthError("depth response is not valid JSON") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("bids"), list) or not isinstance(payload.get("asks"), list):
            raise BinanceDepthError("depth response must contain bids and asks arrays")
        bids = self._levels(payload["bids"], "bids")
        asks = self._levels(payload["asks"], "asks")
        if not bids or not asks:
            raise BinanceDepthError("depth response contains an empty side")
        return OrderBookSnapshot(bids=tuple(bids), asks=tuple(asks))

    @staticmethod
    def _levels(raw_levels: list[Any], side: str) -> list[OrderBookLevel]:
        levels: list[OrderBookLevel] = []
        for raw in raw_levels:
            if not isinstance(raw, list) or len(raw) < 2:
                raise BinanceDepthError(f"invalid {side} level")
            try:
                price = float(raw[0])
                quantity = float(raw[1])
            except (TypeError, ValueError) as exc:
                raise BinanceDepthError(f"invalid {side} level values") from exc
            if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity <= 0:
                raise BinanceDepthError(f"invalid {side} level values")
            levels.append(OrderBookLevel(price=price, quantity=quantity))
        return levels

    @staticmethod
    def _default_http_get(url: str, timeout: float) -> bytes:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "crypto-futures-quant/1.0"}, method="GET")
        with urlopen(request, timeout=timeout) as response:
            return response.read()
