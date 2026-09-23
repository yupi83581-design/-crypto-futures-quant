import json
from pathlib import Path

from scripts.audit_paper_evidence import audit_latest_window


def test_latest_window_passes_for_consecutive_real_events(tmp_path, monkeypatch):
    import scripts.audit_paper_evidence as module

    events = []
    for i in range(12):
        minute = i * 5
        events.append({
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "market_data": "BINANCE_PUBLIC_FUTURES_KLINES",
            "order_book": "BINANCE_PUBLIC_FUTURES_DEPTH",
            "candle_event_time": f"2026-09-23T10:{minute:02d}:00.000Z",
            "probability_raw": 0.55,
            "integrity_status": "SUSPICIOUS",
            "cycles": i + 1,
        })
    path = tmp_path / "paper_trading.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in events) + "\n")
    monkeypatch.setattr(module, "LEDGER", path)
    report = audit_latest_window()
    assert report["status"] == "PASS"


def test_latest_window_rejects_timestamp_gap(tmp_path, monkeypatch):
    import scripts.audit_paper_evidence as module

    events = []
    for i in range(12):
        minute = i * 5
        if i == 6:
            minute += 5
        events.append({
            "symbol": "BTCUSDT",
            "timeframe": "5m",
            "market_data": "BINANCE_PUBLIC_FUTURES_KLINES",
            "order_book": "BINANCE_PUBLIC_FUTURES_DEPTH",
            "candle_event_time": f"2026-09-23T10:{minute:02d}:00.000Z",
            "probability_raw": 0.55,
            "integrity_status": "SUSPICIOUS",
            "cycles": i + 1,
        })
    path = tmp_path / "paper_trading.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in events) + "\n")
    monkeypatch.setattr(module, "LEDGER", path)
    report = audit_latest_window()
    assert report["status"] == "FAIL"
