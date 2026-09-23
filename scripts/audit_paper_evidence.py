"""Audit the latest real paper-evidence window without modifying evidence.

The audit is deliberately forensic: it never deletes, rewrites, or fabricates
evidence. A validation window must contain consecutive closed 5m Binance
observations with no timestamp gaps beyond one candle interval.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "evidence" / "paper_trading.jsonl"
WINDOW = 12
EXPECTED_INTERVAL_SECONDS = 300


def _load_events() -> list[dict]:
    if not LEDGER.exists():
        raise RuntimeError("paper evidence ledger does not exist")
    events = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def audit_latest_window() -> dict:
    events = _load_events()
    if len(events) < WINDOW:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": f"need at least {WINDOW} events", "events": len(events)}

    latest = events[-WINDOW:]
    errors: list[str] = []

    for index, event in enumerate(latest):
        if event.get("symbol") != "BTCUSDT":
            errors.append(f"event {index + 1}: unexpected symbol")
        if event.get("timeframe") != "5m":
            errors.append(f"event {index + 1}: unexpected timeframe")
        if event.get("market_data") != "BINANCE_PUBLIC_FUTURES_KLINES":
            errors.append(f"event {index + 1}: missing Binance public klines provenance")
        if event.get("order_book") != "BINANCE_PUBLIC_FUTURES_DEPTH":
            errors.append(f"event {index + 1}: missing Binance public depth provenance")
        probability = event.get("probability_raw")
        if not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
            errors.append(f"event {index + 1}: invalid probability")
        if event.get("integrity_status") not in {"NORMAL", "SUSPICIOUS"}:
            errors.append(f"event {index + 1}: invalid integrity status")

    candle_times = [_parse_time(event["candle_event_time"]) for event in latest]
    for left, right in zip(candle_times, candle_times[1:]):
        delta = (right - left).total_seconds()
        if delta != EXPECTED_INTERVAL_SECONDS:
            errors.append(f"candle gap is {delta:.0f}s; expected {EXPECTED_INTERVAL_SECONDS}s")

    cycle_numbers = [event.get("cycles") for event in latest]
    if any(not isinstance(value, int) for value in cycle_numbers):
        errors.append("cycle numbers must be integers")
    elif cycle_numbers != list(range(cycle_numbers[0], cycle_numbers[0] + WINDOW)):
        errors.append("latest window cycle numbers are not consecutive")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "window_size": WINDOW,
        "total_events": len(events),
        "first_candle_event_time": latest[0]["candle_event_time"],
        "last_candle_event_time": latest[-1]["candle_event_time"],
        "cycle_start": cycle_numbers[0],
        "cycle_end": cycle_numbers[-1],
        "errors": errors,
    }


def main() -> int:
    report = audit_latest_window()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
