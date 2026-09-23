"""Forensic Phase-10 accounting audit.

Preserves evidence/phase10_backtest.json and creates a separate audit artifact.
The audit reconstructs the same WFO/final-test trade stream from the official
Binance Vision dataset, then compares the original overlapping accounting with
single-position non-overlapping accounting. It never changes the historical
artifact or the dataset.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.models.baseline_pipeline import BaselineRSIPipeline
from src.models.calibration import evaluate_calibration
from scripts.run_phase10_backtest import (
    LABEL_HORIZON,
    FINAL_TEST_DAYS,
    THRESHOLDS,
    load_data,
    parse_time,
    performance,
    strategy_returns_for_wfo,
    trades_from_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "evidence" / "phase10_backtest.json"
OUT = ROOT / "evidence" / "phase10_accounting_audit.json"


def select_non_overlapping(trades):
    ordered = sorted(trades, key=lambda t: (parse_time(t.entry_time), parse_time(t.exit_time)))
    selected = []
    next_available = None
    for trade in ordered:
        entry = parse_time(trade.entry_time)
        exit_time = parse_time(trade.exit_time)
        if next_available is None or entry >= next_available:
            selected.append(trade)
            next_available = exit_time
    return selected


def overlap_diagnostics(trades):
    ordered = sorted(trades, key=lambda t: (parse_time(t.entry_time), parse_time(t.exit_time)))
    max_concurrency = 0
    for i, trade in enumerate(ordered):
        entry = parse_time(trade.entry_time)
        concurrent = sum(
            1
            for other in ordered
            if parse_time(other.entry_time) < parse_time(trade.exit_time)
            and parse_time(other.exit_time) > entry
        )
        max_concurrency = max(max_concurrency, concurrent)
    selected = select_non_overlapping(ordered)
    return {
        "raw_trade_count": len(ordered),
        "non_overlapping_trade_count": len(selected),
        "overlap_trade_count": len(ordered) - len(selected),
        "max_concurrent_positions": max_concurrency,
        "holding_horizon_minutes": LABEL_HORIZON * 5,
        "accounting_interpretation": (
            "ORIGINAL compounds each completed trade as if full equity were "
            "available for every 5m signal; AUDITED_NON_OVERLAPPING permits one "
            "long position at a time and therefore removes concurrent entries."
        ),
    }


def final_test_trades(records, end):
    final_cut = end - timedelta(days=FINAL_TEST_DAYS)
    development = [r for r in records if parse_time(r["event_time"]) < final_cut]
    final_test = [r for r in records if parse_time(r["event_time"]) >= final_cut]
    purge_bars = LABEL_HORIZON
    embargo_bars = LABEL_HORIZON
    final_fit = development[:-purge_bars]
    final_eval_test = final_test[embargo_bars:]
    pipeline = BaselineRSIPipeline(rsi_period=14, label_horizon=LABEL_HORIZON)
    pipeline.fit(final_fit)
    evaluation = pipeline.evaluate(final_fit[-14:] + final_eval_test)
    usable = max(0, len(final_eval_test) - LABEL_HORIZON)
    probabilities = (
        list(evaluation.probabilities[-(usable + LABEL_HORIZON):-LABEL_HORIZON])
        if usable
        else []
    )
    observations = final_eval_test[:usable]
    trades = trades_from_predictions(probabilities, observations, 0.50)
    return final_cut, final_test, trades


def main():
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    start = parse_time(historical["start"])
    end = parse_time(historical["end"])

    records, integrity = load_data(
        historical["symbol"],
        start,
        end,
        ROOT / ".phase10_cache",
    )

    development_end = end - timedelta(days=FINAL_TEST_DAYS)
    wfo_trades, _, folds = strategy_returns_for_wfo(records, development_end)

    original_wfo = historical["trading_performance_wfo"]
    raw_wfo = wfo_trades[0.50]
    audited_wfo = select_non_overlapping(raw_wfo)

    final_cut, final_test, raw_final = final_test_trades(records, end)
    audited_final = select_non_overlapping(raw_final)

    report = {
        "status": "COMPLETE",
        "audit_type": "PHASE10_OVERLAPPING_TRADE_ACCOUNTING",
        "historical_artifact": "evidence/phase10_backtest.json",
        "historical_artifact_preserved": True,
        "source": historical["source"],
        "symbol": historical["symbol"],
        "timeframe": historical["timeframe"],
        "historical_range": {"start": historical["start"], "end": historical["end"]},
        "methodology": {
            "label_horizon_bars": LABEL_HORIZON,
            "timeframe_minutes": 5,
            "holding_horizon_minutes": LABEL_HORIZON * 5,
            "original_accounting": "full-equity compounding on every qualifying 5m signal, without position-state/exposure constraint",
            "audited_accounting": "single-position, non-overlapping trade accounting",
            "selection_rule": "chronological trades; accept a trade only when its entry_time is at or after the prior accepted trade's exit_time",
            "lookahead": "unchanged; signals and labels are reconstructed by the existing Phase-10 pipeline",
            "fees_slippage": {
                "fee_fraction": historical["method"]["fee_assumption"],
                "slippage_fraction": historical["method"]["slippage_assumption"],
            },
            "final_test": "same untouched final-test date boundary; no threshold re-selection",
        },
        "data_integrity_recomputed": integrity,
        "walk_forward_folds": folds,
        "wfo": {
            "original_artifact": original_wfo,
            "raw_reconstructed": performance(raw_wfo),
            "audited_non_overlapping": performance(audited_wfo),
            "overlap_diagnostics": overlap_diagnostics(raw_wfo),
        },
        "untouched_final_test": {
            "start": final_cut.isoformat(),
            "end": end.isoformat(),
            "bars": len(final_test),
            "threshold": 0.50,
            "raw_reconstructed": performance(raw_final),
            "audited_non_overlapping": performance(audited_final),
            "overlap_diagnostics": overlap_diagnostics(raw_final),
            "untouched": True,
        },
        "conclusion": {
            "accounting_methodology_issue_present": len(audited_wfo) < len(raw_wfo),
            "strategy_edge_verdict": "NOT_DETERMINED_BY_THIS_AUDIT",
            "reason": "This audit isolates portfolio-accounting distortion; it does not declare an edge or no-edge verdict.",
        },
        "real_money_execution": False,
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
