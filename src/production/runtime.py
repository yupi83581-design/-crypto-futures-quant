"""Persistent runtime composition for the read-only quant snapshot service.

The runtime wires the repository's existing Binance market-data adapter,
RSI feature, baseline logistic model, production inference engine, snapshot
bridge, and HTTP transport. It never fabricates market data or substitutes
missing numeric values.

The baseline model is fitted only from an earlier slice of the same real
exchange dataset fetched at startup. The current inference window is fetched
separately through ProductionInferenceEngine, preserving a temporal boundary.
This is a research/paper runtime only: it places no exchange orders.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from src.data.collector.adapter import BinancePublicMarketDataAdapter, MarketDataError
from src.data.collector.binance_depth import BinancePublicDepthAdapter, BinanceDepthError
from src.market_integrity.detector import assess_market_integrity
from src.features.rsi import compute_rsi
from src.models.baseline_pipeline import BaselineRSIPipeline
from src.production.command_center_contract import QuantCommandCenterSnapshot
from src.production.inference import InferenceInput, InferenceResult, ProductionInferenceEngine
from src.production.paper_cycle import ProductionPaperCycle
from src.monitoring.journal import Journal
from src.paper.engine import PaperTradingEngine
from src.production.snapshot_bridge import build_command_center_snapshot
from src.production.snapshot_http import create_snapshot_server

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    lookback_candles: int = 240
    training_fraction: float = 0.8
    refresh_seconds: float = 300.0
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_seconds: float = 10.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            symbol=os.getenv("QUANT_SYMBOL", cls.symbol).strip().upper(),
            timeframe=os.getenv("QUANT_TIMEFRAME", cls.timeframe).strip(),
            lookback_candles=int(os.getenv("QUANT_LOOKBACK_CANDLES", str(cls.lookback_candles))),
            training_fraction=float(os.getenv("QUANT_TRAINING_FRACTION", str(cls.training_fraction))),
            refresh_seconds=float(os.getenv("QUANT_REFRESH_SECONDS", str(cls.refresh_seconds))),
            host=os.getenv("QUANT_SNAPSHOT_HOST", cls.host),
            port=int(os.getenv("QUANT_SNAPSHOT_PORT", str(cls.port))),
            request_timeout_seconds=float(
                os.getenv("QUANT_HTTP_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))
            ),
            max_retries=int(os.getenv("QUANT_HTTP_MAX_RETRIES", str(cls.max_retries))),
        )

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("QUANT_SYMBOL must not be empty")
        if not self.timeframe:
            raise ValueError("QUANT_TIMEFRAME must not be empty")
        if self.lookback_candles < 40:
            raise ValueError("QUANT_LOOKBACK_CANDLES must be >= 40")
        if not 0.5 <= self.training_fraction < 1.0:
            raise ValueError("QUANT_TRAINING_FRACTION must be >= 0.5 and < 1")
        if self.refresh_seconds <= 0:
            raise ValueError("QUANT_REFRESH_SECONDS must be > 0")
        if not 0 <= self.port <= 65535:
            raise ValueError("QUANT_SNAPSHOT_PORT must be between 0 and 65535")


class RuntimeState:
    """Thread-safe last-known runtime snapshot state."""

    def __init__(self, symbol: str) -> None:
        self._lock = threading.Lock()
        self._snapshot = build_command_center_snapshot(
            symbol=symbol,
            status="ERROR",
            market_data={"status": "OFFLINE", "reason": "runtime not connected"},
            validation_status="INSUFFICIENT_EVIDENCE",
        )
        self._last_success: datetime | None = None
        self._last_error: str | None = None

    def get(self) -> QuantCommandCenterSnapshot:
        with self._lock:
            return self._snapshot

    def set_snapshot(self, snapshot: QuantCommandCenterSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._last_success = datetime.now(timezone.utc)
            self._last_error = None

    def set_offline(self, reason: str) -> None:
        with self._lock:
            symbol = self._snapshot.symbol
            self._snapshot = build_command_center_snapshot(
                symbol=symbol,
                status="ERROR",
                market_data={"status": "OFFLINE", "reason": reason},
                validation_status="INSUFFICIENT_EVIDENCE",
            )
            self._last_error = reason


class ProductionSnapshotRuntime:
    """Long-running composition of real data, inference, snapshot, and HTTP."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        market_data_adapter: BinancePublicMarketDataAdapter | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.adapter = market_data_adapter or BinancePublicMarketDataAdapter(
            timeout_seconds=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )
        self.depth_adapter = BinancePublicDepthAdapter(
            timeout_seconds=config.request_timeout_seconds,
        )
        self._previous_depth = None
        self.state = RuntimeState(config.symbol)
        self.paper_engine = PaperTradingEngine()
        self.journal = Journal()
        self.paper_cycle = ProductionPaperCycle(
            paper_engine=self.paper_engine,
            journal=self.journal,
        )
        self._stop = threading.Event()
        self._server = None

    def _window(self) -> tuple[str, str]:
        end = self._clock().astimezone(timezone.utc)
        interval_seconds = _timeframe_seconds(self.config.timeframe)
        start = end - timedelta(seconds=interval_seconds * self.config.lookback_candles)
        return start.isoformat(), end.isoformat()

    def _fit_model(self, records: Sequence[dict[str, Any]]) -> Any:
        split = int(len(records) * self.config.training_fraction)
        training = list(records[:split])
        if len(training) < 20:
            raise RuntimeError("insufficient real market data for model training")
        pipeline = BaselineRSIPipeline()
        pipeline.fit(training)
        return pipeline.model

    @staticmethod
    def _rsi_features(records: list[dict[str, Any]]) -> Sequence[float | None]:
        return compute_rsi(records, period=14)

    def refresh_once(self) -> InferenceResult:
        start_time, end_time = self._window()
        records = self.adapter.fetch_market_data(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        if len(records) < 40:
            raise RuntimeError("real exchange returned insufficient closed candles")

        _validate_freshness(
            records,
            now=self._clock().astimezone(timezone.utc),
            interval_seconds=_timeframe_seconds(self.config.timeframe),
        )

        split = int(len(records) * self.config.training_fraction)
        if split <= 0 or split >= len(records):
            raise RuntimeError("invalid temporal training/inference split")

        model = self._fit_model(records)
        inference_engine = ProductionInferenceEngine(
            market_data_adapter=self.adapter,
            model=model,
            feature_builder=self._rsi_features,
        )

        inference = inference_engine.run(
            InferenceInput(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                start_time=records[split]["event_time"],
                end_time=end_time,
            )
        )

        latest_close = float(records[-1]["close"])
        latest_low = float(records[-1]["low"])
        depth = self.depth_adapter.fetch_depth(self.config.symbol)
        baseline_volume = sum(float(r["volume"]) for r in records[-21:-1]) / max(1, len(records[-21:-1]))
        integrity = assess_market_integrity(
            bid_depth=depth.bids,
            ask_depth=depth.asks,
            prior_bid_depth=self._previous_depth.bids if self._previous_depth is not None else None,
            prior_ask_depth=self._previous_depth.asks if self._previous_depth is not None else None,
            recent_volume=float(records[-1]["volume"]),
            baseline_volume=baseline_volume,
        )
        self._previous_depth = depth
        paper_cycle = self.paper_cycle.run(
            symbol=inference.symbol,
            probability=inference.probability,
            entry_price=latest_close,
            stop_price=latest_low,
            market_integrity=integrity,
        )
        paper_status = {
            "status": "PAPER" if paper_cycle.paper_position is not None else "NO_TRADE",
            "reason": paper_cycle.decision.reason,
            "market_integrity": {
                "status": integrity.status,
                "spoofing_risk": integrity.spoofing_risk,
                "liquidity_withdrawal_risk": integrity.liquidity_withdrawal_risk,
                "volume_anomaly_risk": integrity.volume_anomaly_risk,
                "reasons": list(integrity.reasons),
            },
            "journal_entries": self.journal.snapshot().total_entries,
        }
        journal_snapshot = self.journal.snapshot()

        snapshot = build_command_center_snapshot(
            symbol=inference.symbol,
            status="READY",
            market_data={
                "status": "LIVE",
                "exchange": "BINANCE",
                "market_type": "FUTURES",
                "data_type": "OHLCV",
                "timeframe": inference.timeframe,
                "observations": inference.observations,
                "usable_observations": inference.usable_observations,
                "provenance": "BINANCE_PUBLIC_FUTURES_KLINES",
            },
            inference=inference,
            paper_trading=paper_status,
            journal_monitoring=journal_snapshot,
            validation_status="INSUFFICIENT_EVIDENCE",
        )
        self.state.set_snapshot(snapshot)
        return inference

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            try:
                inference = self.refresh_once()
                LOGGER.info(
                    "runtime refresh OK symbol=%s timeframe=%s probability=%s",
                    inference.symbol,
                    inference.timeframe,
                    inference.probability,
                )
            except (MarketDataError, BinanceDepthError, RuntimeError, ValueError, TypeError) as exc:
                LOGGER.error("runtime refresh unavailable: %s", exc)
                self.state.set_offline(str(exc))
            self._stop.wait(self.config.refresh_seconds)

    def start(self) -> None:
        """Start HTTP transport and background refresh loop."""
        self._server = create_snapshot_server(
            self.config.host,
            self.config.port,
            self.state.get,
        )
        worker = threading.Thread(
            target=self._refresh_loop,
            name="quant-runtime-refresh",
            daemon=True,
        )
        worker.start()
        actual_host, actual_port = self._server.server_address
        LOGGER.info(
            "snapshot runtime listening on http://%s:%s/snapshot",
            actual_host,
            actual_port,
        )
        try:
            self._server.serve_forever(poll_interval=0.5)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def _timeframe_seconds(timeframe: str) -> int:
    from src.data.collector.adapter import TIMEFRAME_MILLISECONDS

    milliseconds = TIMEFRAME_MILLISECONDS.get(timeframe)
    if milliseconds is None:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return milliseconds // 1000


def _validate_freshness(
    records: Sequence[dict[str, Any]],
    *,
    now: datetime,
    interval_seconds: int,
    max_stale_intervals: int = 2,
) -> None:
    """Fail closed when the latest closed market observation is too old."""
    if not records:
        raise RuntimeError("freshness check received no market records")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if max_stale_intervals < 1:
        raise ValueError("max_stale_intervals must be positive")

    latest = records[-1]
    event_time = datetime.fromisoformat(
        str(latest["event_time"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    available_time = datetime.fromisoformat(
        str(latest["available_time"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)

    if event_time > now:
        raise RuntimeError("latest market event is in the future")
    if available_time > now:
        raise RuntimeError("latest market observation is not yet available")

    age_seconds = (now - available_time).total_seconds()
    if age_seconds > interval_seconds * max_stale_intervals:
        raise RuntimeError(
            f"market data is stale: age={age_seconds:.1f}s "
            f"limit={interval_seconds * max_stale_intervals}s"
        )


def main() -> None:
    logging.basicConfig(
        level=os.getenv("QUANT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    runtime = ProductionSnapshotRuntime(RuntimeConfig.from_env())
    try:
        runtime.start()
    except KeyboardInterrupt:
        runtime.stop()


if __name__ == "__main__":
    main()

