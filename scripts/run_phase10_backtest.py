"""Phase 10 research backtest on official Binance Vision USD-M Futures data.

Research-only. No exchange orders, no validation-gate changes, no execution unlock.
Uses the repository's existing BaselineRSIPipeline and direction-label horizon.
The archive source is data.binance.vision.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import math
import statistics
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.models.baseline_pipeline import BaselineRSIPipeline

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{symbol}-5m-{year:04d}-{month:02d}.zip"
LABEL_HORIZON = 3
RSI_WARMUP = 14
TRAIN_DAYS = 90
TEST_DAYS = 30
THRESHOLDS = (0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60)
FEE = 0.0005
SLIPPAGE = 0.0002


@dataclass(frozen=True)
class Trade:
    entry_time: str
    entry_price: float
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


def download_month(symbol: str, year: int, month: int, cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{symbol}-5m-{year:04d}-{month:02d}.zip"
    if not target.exists():
        url = ARCHIVE.format(symbol=symbol, year=year, month=month)
        req = urllib.request.Request(url, headers={"User-Agent": "crypto-futures-quant-phase10/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            target.write_bytes(response.read())
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
    return rows


def load_data(symbol: str, start: datetime, end: datetime, cache: Path) -> list[dict]:
    records: list[dict] = []
    for year, month in month_range(start, end):
        try:
            records.extend(download_month(symbol, year, month, cache))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
    records.sort(key=lambda r: r["event_time"])
    return [r for r in records if start <= parse_time(r["event_time"]) <= end and r["close_time"] <= end]


def predict_fold(train: list[dict], test: list[dict]) -> tuple[list[float], list[int], list[dict]]:
    warmup = train[-RSI_WARMUP:]
    future = test[-LABEL_HORIZON:]
    evaluation = warmup + test + future
    pipeline = BaselineRSIPipeline(rsi_period=14, label_horizon=LABEL_HORIZON)
    pipeline.fit(train)
    result = pipeline.evaluate(evaluation)
    # First 14 evaluation records are RSI warmup. Last 3 are label warmup.
    usable = max(0, len(test) - LABEL_HORIZON)
    probabilities = list(result.probabilities[:usable])
    labels = list(result.actual_labels[:usable])
    observations = test[:usable]
    if len(probabilities) != len(observations):
        raise RuntimeError("walk-forward alignment failed")
    return probabilities, labels, observations


def trades_from_predictions(probabilities, observations, threshold) -> list[Trade]:
    trades: list[Trade] = []
    for p, rec in zip(probabilities, observations):
        if p < threshold:
            continue
        idx = observations.index(rec)
        if idx + LABEL_HORIZON >= len(observations):
            continue
        entry = float(rec["close"])
        exit_price = float(observations[idx + LABEL_HORIZON]["close"])
        net = (exit_price - entry) / entry - 2.0 * (FEE + SLIPPAGE)
        trades.append(Trade(rec["event_time"], entry, exit_price, net))
    return trades


def max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    dd = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak)
    return dd


def sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    sd = statistics.stdev(returns)
    return 0.0 if sd == 0 else mean / sd * math.sqrt(len(returns))


def pbo_cs_cv(strategy_returns: list[list[float]]) -> float | None:
    if len(strategy_returns) < 4:
        return None
    n = min(len(x) for x in strategy_returns)
    if n < 16:
        return None
    # Equal contiguous blocks; CSCV requires an even number of blocks.
    blocks = 8 if n >= 80 else 4
    block_size = n // blocks
    trimmed = [x[:block_size * blocks] for x in strategy_returns]
    combos = list(itertools.combinations(range(blocks), blocks // 2))
    if not combos:
        return None
    worse = 0
    total = 0
    for train_blocks in combos:
        test_blocks = tuple(i for i in range(blocks) if i not in train_blocks)
        train_scores = []
        for series in trimmed:
            train = [v for b in train_blocks for v in series[b*block_size:(b+1)*block_size]]
            train_scores.append(statistics.mean(train))
        selected = max(range(len(train_scores)), key=train_scores.__getitem__)
        test = [v for b in test_blocks for v in trimmed[selected][b*block_size:(b+1)*block_size]]
        selected_test = statistics.mean(test) if test else 0.0
        cross = [statistics.mean([v for b in test_blocks for v in s[b*block_size:(b+1)*block_size]]) for s in trimmed]
        if selected_test < statistics.median(cross):
            worse += 1
        total += 1
    return worse / total if total else None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    return statistics.NormalDist().inv_cdf(p)


def main() -> None:
    args=parse_args()
    start=parse_time(args.start)
    end=parse_time(args.end) if args.end else datetime.now(timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")
    records=load_data(args.symbol,start,end,Path(args.cache))
    if len(records) < 10000:
        raise SystemExit(f"Insufficient official Binance Vision data: {len(records)} bars")
    step=30*24*12
    train_size=90*24*12
    all_trades={t:[] for t in THRESHOLDS}
    fold_count=0
    cursor=train_size
    while cursor + step <= len(records):
        train=records[cursor-train_size:cursor]
        test=records[cursor:cursor+step]
        probs, labels, observations=predict_fold(train,test)
        for threshold in THRESHOLDS:
            all_trades[threshold].extend(trades_from_predictions(probs,observations,threshold))
        fold_count += 1
        cursor += step
    baseline=all_trades[0.50]
    rets=[t.net_return for t in baseline]
    wins=sum(r>0 for r in rets)
    losses=sum(r<=0 for r in rets)
    expectancy=statistics.mean(rets) if rets else 0.0
    result={
        "status":"COMPLETE",
        "source":"Binance Vision official USD-M Futures monthly klines",
        "symbol":args.symbol,"timeframe":"5m",
        "start":start.isoformat(),"end":end.isoformat(),
        "bars":len(records),"walk_forward_folds":fold_count,
        "method":{"train_days":TRAIN_DAYS,"test_days":TEST_DAYS,"label_horizon":LABEL_HORIZON,"rsi_period":14,"fee_assumption":FEE,"slippage_assumption":SLIPPAGE},
        "baseline":{"threshold":0.50,"trade_count":len(rets),"win_rate":wins/len(rets) if rets else 0.0,"expectancy_per_trade":expectancy,"max_drawdown":max_drawdown(rets),"sharpe":sharpe(rets),"total_return":math.prod(1+r for r in rets)-1 if rets else 0.0},
        "pbo":pbo_cs_cv([[t.net_return for t in all_trades[x]] for x in THRESHOLDS]),
        "deflated_sharpe_ratio":deflated_sharpe(rets,len(THRESHOLDS)),
        "strategy_variants":{"thresholds":list(THRESHOLDS),"trials":len(THRESHOLDS)},
        "validation_status":"INSUFFICIENT_EVIDENCE",
        "execution_lock":"READ_ONLY",
        "real_money_execution":False,
    }
    import json
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
