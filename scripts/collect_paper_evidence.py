"""One real-market paper-trading evidence cycle.

Reads only public Binance USDⓈ-M Futures data. No fixtures, synthetic candles,
private credentials, or exchange orders. State is persisted by the CI workflow
so successive real observations form a continuous paper-trading ledger.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from src.data.collector.adapter import BinancePublicMarketDataAdapter
from src.market_integrity.detector import OrderBookLevel, assess_market_integrity
from src.models.baseline_pipeline import BaselineRSIPipeline
from src.models.expected_value import calculate_expected_value
from src.risk.engine import RiskConfig, assess_risk
from src.decision.engine import DecisionConfig, make_decision

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "evidence" / "paper_state.json"
LEDGER = ROOT / "evidence" / "paper_trading.jsonl"


def fetch_depth(symbol: str, limit: int = 20) -> tuple[list[OrderBookLevel], list[OrderBookLevel]]:
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "crypto-futures-quant/1.0"})
    with urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    bids = [OrderBookLevel(float(p), float(q)) for p, q in payload["bids"]]
    asks = [OrderBookLevel(float(p), float(q)) for p, q in payload["asks"]]
    if not bids or not asks:
        raise RuntimeError("Binance returned an empty order book")
    return bids, asks


def main() -> None:
    symbol = os.getenv("QUANT_SYMBOL", "BTCUSDT")
    timeframe = os.getenv("QUANT_TIMEFRAME", "5m")
    adapter = BinancePublicMarketDataAdapter(timeout_seconds=10, max_retries=2)

    end = datetime.now(timezone.utc)
    start = end.timestamp() - 300 * 240
    records = adapter.fetch_market_data(
        symbol,
        timeframe,
        datetime.fromtimestamp(start, timezone.utc),
        end,
    )
    if len(records) < 40:
        raise RuntimeError(f"real Binance returned only {len(records)} closed candles")

    split = int(len(records) * 0.8)
    pipeline = BaselineRSIPipeline()
    pipeline.fit(records[:split])

    features = pipeline._prepare_dataset(records[split:])
    if not features:
        raise RuntimeError("real inference window produced no usable observations")
    probability = float(pipeline.model.predict_proba([features[-1][0]])[0][1])

    bids, asks = fetch_depth(symbol)
    recent_volume = float(records[-1]["volume"])
    baseline_volume = sum(float(r["volume"]) for r in records[-21:-1]) / 20.0
    integrity = assess_market_integrity(
        bid_depth=bids,
        ask_depth=asks,
        recent_volume=recent_volume,
        baseline_volume=baseline_volume,
    )

    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "equity": 100000.0,
        "position": None,
        "completed_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "cycles": 0,
    }

    close = float(records[-1]["close"])
    if state["position"] is not None:
        pos = state["position"]
        pnl = (close - pos["entry_price"]) * pos["quantity"]
        state["equity"] += pnl
        state["completed_trades"] += 1
        state["total_pnl"] += pnl
        if pnl > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1
        state["position"] = None

    reward = close * 0.02
    loss = max(0.0, close - float(records[-1]["low"]))
    ev = calculate_expected_value(
        probability=probability,
        reward=reward,
        loss=loss,
        fee=0.0,
        slippage=0.0,
    )
    risk = assess_risk(
        equity=float(state["equity"]),
        entry_price=close,
        stop_price=float(records[-1]["low"]),
        net_expected_value=ev.net_expected_value,
        config=RiskConfig(),
    ) if integrity.status == "NORMAL" else None
    decision = make_decision(
        probability=probability,
        expected_value=ev.net_expected_value if integrity.status == "NORMAL" else 0.0,
        risk_approved=bool(risk and risk.approved),
        config=DecisionConfig(),
    )

    opened = False
    if decision.approved and risk is not None:
        state["position"] = {
            "entry_price": close,
            "quantity": risk.position_notional / close,
            "entry_time": records[-1]["event_time"],
        }
        opened = True

    state["cycles"] += 1
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "market_data": "BINANCE_PUBLIC_FUTURES_KLINES",
        "order_book": "BINANCE_PUBLIC_FUTURES_DEPTH",
        "candle_event_time": records[-1]["event_time"],
        "close": close,
        "probability_raw": probability,
        "integrity_status": integrity.status,
        "integrity_reasons": list(integrity.reasons),
        "expected_value_net": ev.net_expected_value,
        "risk_approved": bool(risk and risk.approved),
        "decision_approved": decision.approved,
        "decision_reason": decision.reason,
        "paper_position_opened": opened,
        "equity": state["equity"],
        "completed_trades": state["completed_trades"],
        "wins": state["wins"],
        "losses": state["losses"],
        "total_pnl": state["total_pnl"],
        "cycles": state["cycles"],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True))


if __name__ == "__main__":
    main()
