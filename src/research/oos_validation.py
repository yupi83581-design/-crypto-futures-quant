"""Unified out-of-sample evidence evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from src.models.calibration import CalibrationResult, evaluate_calibration

@dataclass(frozen=True)
class OOSValidationResult:
    observations: int
    brier_score: float
    calibration: CalibrationResult
    baseline_brier_score: float | None
    gross_return: float
    fees: float
    slippage: float
    net_return: float
    max_drawdown: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float


def validate_oos(*, actual: Sequence[int], probabilities: Sequence[float], baseline_probabilities: Sequence[float] | None = None, prices: Sequence[float] | None = None, signals: Sequence[int] | None = None, fee_fraction: float = 0.0, slippage_fraction: float = 0.0) -> OOSValidationResult:
    if len(actual) != len(probabilities): raise ValueError("actual and probabilities must have the same length")
    if not actual: raise ValueError("OOS observations must not be empty")
    if baseline_probabilities is not None and len(baseline_probabilities) != len(actual): raise ValueError("baseline probabilities must match actual length")
    if prices is not None and len(prices) != len(actual): raise ValueError("prices must match actual length")
    if signals is not None and len(signals) != len(actual): raise ValueError("signals must match actual length")
    if not 0 <= fee_fraction <= 1 or not 0 <= slippage_fraction <= 1: raise ValueError("cost fractions must be between 0 and 1")
    calibration = evaluate_calibration(actual, probabilities)
    baseline_brier = evaluate_calibration(actual, baseline_probabilities).brier_score if baseline_probabilities is not None else None
    gross = fees = slippage = 0.0
    equity = peak = 1.0
    max_dd = 0.0
    trades = wins = losses = 0
    if prices is not None and signals is not None:
        for i in range(len(prices) - 1):
            if signals[i] not in (0, 1): raise ValueError("signals must contain only 0 or 1")
            if signals[i] == 0: continue
            p0, p1 = float(prices[i]), float(prices[i + 1])
            if p0 <= 0 or p1 <= 0: raise ValueError("prices must be positive")
            r = (p1 - p0) / p0
            cost_fee = 2 * fee_fraction; cost_slip = 2 * slippage_fraction
            net = r - cost_fee - cost_slip
            gross += r; fees += cost_fee; slippage += cost_slip; equity *= 1 + net; peak = max(peak, equity); max_dd = max(max_dd, (peak-equity)/peak)
            trades += 1; wins += int(net > 0); losses += int(net <= 0)
    return OOSValidationResult(len(actual), calibration.brier_score, calibration, baseline_brier, gross, fees, slippage, equity - 1.0, max_dd, trades, wins, losses, wins / trades if trades else 0.0)
