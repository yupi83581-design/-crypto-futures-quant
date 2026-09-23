"""Deterministic Binance Vision USD-M Futures 5m validation.

Research-only. Uses official Binance Vision monthly archives, performs temporal
walk-forward/OOS validation, reserves an untouched final test window, accounts
for fees/slippage, and records reproducibility/data-integrity evidence.
No exchange orders are placed and the execution lock remains OFF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import statistics
import subprocess
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.models.baseline_pipeline import BaselineRSIPipeline
from src.research.dsr import deflated_sharpe_ratio
from src.research.pbo import pbo_cs_cv

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{symbol}-5m-{year:04d}-{month:02d}.zip"
LABEL_HORIZON = 3
RSI_PERIOD = 14
TRAIN_DAYS = 90
TEST_DAYS = 30
FINAL_TEST_DAYS = 30
THRESHOLDS = (0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60)
FEE = 0.0005
SLIPPAGE = 0.0002


@dataclass(frozen=True)
class Trade:
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    net_return: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2019-09-01T00:00:00Z")
    p.add_argument("--end", default=None)
    p.add_argument("--cache", default=".phase10_cache")
    p.add_argument("--output", default="evidence/phase10_backtest.json")
    return p.parse_args()


def month_range(start: datetime, end: datetime):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def download_month(symbol: str, year: int, month: int, cache: Path) -> tuple[list[dict], str]:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{symbol}-5m-{year:04d}-{month:02d}.zip"
    if not target.exists():
        url = ARCHIVE.format(symbol=symbol, year=year, month=month)
        req = urllib.request.Request(url, headers={"User-Agent": "crypto-futures-quant-validation/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    rows: list[dict] = []
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
        csv_name = next(n for n in names if n.endswith(".csv"))
        with zf.open(csv_name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                if not row or not row[0].isdigit():
                    continue
                open_time = datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc)
                close_time = datetime.fromtimestamp(int(row[6]) / 1000, timezone.utc)
                rows.append({
                    "symbol": symbol,
                    "timeframe": "5m",
                    "event_time": open_time.isoformat().replace("+00:00", "Z"),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": close_time,
                })
    return rows, digest


def load_data(symbol: str, start: datetime, end: datetime, cache: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    archive_hashes: dict[str, str] = {}
    for year, month in month_range(start, end):
        try:
            rows, digest = download_month(symbol, year, month, cache)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        records.extend(rows)
        archive_hashes[f"{year:04d}-{month:02d}"] = digest

    by_time: dict[str, dict] = {}
    for record in records:
        key = record["event_time"]
        previous = by_time.get(key)
        if previous is not None:
            comparable = ("open", "high", "low", "close", "volume")
            if any(previous[field] != record[field] for field in comparable):
                raise RuntimeError(f"conflicting Binance Vision rows for {key}")
            continue
        by_time[key] = record

    ordered = sorted(by_time.values(), key=lambda r: parse_time(r["event_time"]))
    selected = [r for r in ordered if start <= parse_time(r["event_time"]) <= end and r["close_time"] <= end]
    if not selected:
        raise RuntimeError("official Binance Vision returned no usable bars")

    integrity = validate_ohlcv_integrity(selected)
    canonical = json.dumps(
        [{k: r[k] for k in ("event_time", "open", "high", "low", "close", "volume")} for r in selected],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    integrity["dataset_sha256"] = hashlib.sha256(canonical).hexdigest()
    integrity["archive_sha256"] = archive_hashes
    return selected, integrity


def validate_ohlcv_integrity(records: list[dict]) -> dict:
    failures: list[str] = []
    duplicate_times = len(records) - len({r["event_time"] for r in records})
    if duplicate_times:
        failures.append("duplicate event_time")
    expected_step = timedelta(minutes=5)
    gaps = 0
    for i, r in enumerate(records):
        o, h, l, c, v = (float(r[k]) for k in ("open", "high", "low", "close", "volume"))
        if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c, v) < 0:
            failures.append(f"invalid numeric OHLCV at {i}")
        if not (l <= min(o, c) <= max(o, c) <= h):
            failures.append(f"OHLC ordering violation at {i}")
        if v < 0:
            failures.append(f"negative volume at {i}")
        if i:
            prev = parse_time(records[i - 1]["event_time"])
            cur = parse_time(r["event_time"])
            if cur - prev != expected_step:
                gaps += 1
    if gaps:
        failures.append(f"{gaps} timestamp gaps")
    return {
        "passed": not failures,
        "bars": len(records),
        "duplicate_timestamps": duplicate_times,
        "timestamp_gaps": gaps,
        "failures": failures,
    }


def predict_fold(train: list[dict], test: list[dict]) -> tuple[list[float], list[dict]]:
    pipeline = BaselineRSIPipeline(rsi_period=RSI_PERIOD, label_horizon=LABEL_HORIZON)
    pipeline.fit(train)
    warmup = train[-RSI_PERIOD:]
    evaluation = pipeline.evaluate(warmup + test)
    # The warmup prefix contributes RSI state but is not part of the test ledger.
    usable = min(len(test) - LABEL_HORIZON, len(evaluation.probabilities) - LABEL_HORIZON)
    if usable <= 0:
        return [], []
    return list(evaluation.probabilities[-(usable + LABEL_HORIZON):-LABEL_HORIZON]), test[:usable]


def trades_from_predictions(probabilities: list[float], observations: list[dict], threshold: float) -> list[Trade]:
    trades: list[Trade] = []
    for idx, p in enumerate(probabilities):
        if p < threshold or idx + LABEL_HORIZON >= len(observations):
            continue
        entry = observations[idx]
        exit_rec = observations[idx + LABEL_HORIZON]
        net = (float(exit_rec["close"]) - float(entry["close"])) / float(entry["close"]) - 2.0 * (FEE + SLIPPAGE)
        trades.append(Trade(entry["event_time"], float(entry["close"]), exit_rec["event_time"], float(exit_rec["close"]), net))
    return trades


def equity_curve(returns: list[float], initial: float = 100_000.0) -> list[float]:
    equity = initial
    curve = [equity]
    for r in returns:
        equity *= 1.0 + r
        curve.append(equity)
    return curve


def max_drawdown(curve: list[float]) -> float:
    peak = curve[0]
    dd = 0.0
    for value in curve:
        peak = max(peak, value)
        dd = max(dd, (peak - value) / peak)
    return dd


def performance(trades: list[Trade]) -> dict:
    returns = [t.net_return for t in trades]
    curve = equity_curve(returns)
    wins = sum(r > 0 for r in returns)
    losses = sum(r <= 0 for r in returns)
    expectancy = statistics.mean(returns) if returns else 0.0
    return {
        "trade_count": len(returns),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": wins / len(returns) if returns else 0.0,
        "expectancy_per_trade": expectancy,
        "net_return": (curve[-1] / curve[0]) - 1.0,
        "max_drawdown": max_drawdown(curve),
        "initial_equity": curve[0],
        "final_equity": curve[-1],
        "equity_curve": curve,
        "fees_per_round_trip": 2.0 * FEE,
        "slippage_per_round_trip": 2.0 * SLIPPAGE,
        "funding_cost": 0.0,
        "funding_relevance": "NOT_APPLICABLE: 15-minute maximum holding horizon is below normal funding interval; no funding leg is modeled.",
        "trade_accounting_hash": hashlib.sha256(
            json.dumps([t.__dict__ for t in trades], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def strategy_returns_for_wfo(records: list[dict], development_end: datetime) -> tuple[dict[float, list[float]], int]:
    development = [r for r in records if parse_time(r["event_time"]) < development_end]
    step = TEST_DAYS * 24 * 12
    train_size = TRAIN_DAYS * 24 * 12
    all_trades = {threshold: [] for threshold in THRESHOLDS}
    folds = 0
    cursor = train_size
    while cursor + step <= len(development):
        train = development[cursor - train_size:cursor]
        test = development[cursor:cursor + step]
        probabilities, observations = predict_fold(train, test)
        for threshold in THRESHOLDS:
            all_trades[threshold].extend(trades_from_predictions(probabilities, observations, threshold))
        folds += 1
        cursor += step
    return {t: [trade.net_return for trade in trades] for t, trades in all_trades.items()}, folds


def main() -> None:
    args = parse_args()
    start = parse_time(args.start)
    end = parse_time(args.end) if args.end else datetime.now(timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")

    records, data_integrity = load_data(args.symbol, start, end, Path(args.cache))
    if len(records) < 10_000:
        raise SystemExit(f"Insufficient official Binance Vision data: {len(records)} bars")

    final_cut = end - timedelta(days=FINAL_TEST_DAYS)
    if final_cut <= start:
        raise SystemExit("date range is too short for untouched final test set")

    wfo_returns, fold_count = strategy_returns_for_wfo(records, final_cut)
    baseline_wfo_trades = [
        Trade("", 0.0, "", 0.0, r) for r in wfo_returns[0.50]
    ]
    if len(baseline_wfo_trades) < 30:
        raise SystemExit("insufficient WFO trades for performance evidence")

    pbo = pbo_cs_cv([wfo_returns[t] for t in THRESHOLDS], block_count=8)
    dsr = deflated_sharpe_ratio(wfo_returns[0.50], trial_count=len(THRESHOLDS))

    # Untouched final test: fixed threshold 0.50, selected before final evaluation.
    development = [r for r in records if parse_time(r["event_time"]) < final_cut]
    final_test = [r for r in records if parse_time(r["event_time"]) >= final_cut]
    pipeline = BaselineRSIPipeline(rsi_period=RSI_PERIOD, label_horizon=LABEL_HORIZON)
    pipeline.fit(development)
    probabilities = pipeline.evaluate(development[-RSI_PERIOD:] + final_test).probabilities
    usable = max(0, len(final_test) - LABEL_HORIZON)
    final_probs = list(probabilities[-(usable + LABEL_HORIZON):-LABEL_HORIZON]) if usable else []
    final_trades = trades_from_predictions(final_probs, final_test[:usable], 0.50)

    result = {
        "status": "COMPLETE",
        "source": "Binance Vision official USD-M Futures monthly klines",
        "symbol": args.symbol,
        "timeframe": "5m",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bars": len(records),
        "data_integrity": data_integrity,
        "walk_forward": {
            "folds": fold_count,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "label_horizon": LABEL_HORIZON,
            "lookahead_protection": True,
            "selection_threshold": 0.50,
        },
        "trading_performance_wfo": performance(baseline_wfo_trades),
        "untouched_final_test": {
            "start": final_cut.isoformat(),
            "end": end.isoformat(),
            "bars": len(final_test),
            "threshold": 0.50,
            "performance": performance(final_trades),
            "untouched": True,
        },
        "dsr": {
            **dsr.__dict__,
            "trial_count_basis": "complete explicit threshold strategy universe",
            "trial_universe": list(THRESHOLDS),
        },
        "pbo_cscv": pbo.__dict__,
        "method": {
            "fee_assumption": FEE,
            "slippage_assumption": SLIPPAGE,
            "funding": "not applicable to 15-minute fixed holding horizon",
            "reproducible": True,
            "strategy_universe": list(THRESHOLDS),
        },
        "reproducibility": {
            "git_commit": _git_commit(),
            "python": _python_version(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "validation_status": "INSUFFICIENT_EVIDENCE",
        "execution_lock": "READ_ONLY",
        "real_money_execution": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _python_version() -> str:
    import sys
    return sys.version.split()[0]


if __name__ == "__main__":
    main()
