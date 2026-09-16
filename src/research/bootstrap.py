"""Bootstrap uncertainty estimates for realized trade returns."""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Sequence

@dataclass(frozen=True)
class BootstrapResult:
    samples: int
    confidence_level: float
    mean_return: float
    lower_bound: float
    upper_bound: float
    probability_adverse: float
    max_drawdown_mean: float
    max_drawdown_upper_bound: float


def bootstrap_trade_returns(*, trade_returns: Sequence[float], samples: int = 2000, confidence_level: float = 0.95, seed: int = 0) -> BootstrapResult:
    if not trade_returns: raise ValueError("trade_returns must not be empty")
    if samples < 1: raise ValueError("samples must be positive")
    if not 0 < confidence_level < 1: raise ValueError("confidence_level must be between 0 and 1")
    values = [float(x) for x in trade_returns]
    if any(x != x or x in (float("inf"), float("-inf")) for x in values): raise ValueError("trade returns must be finite")
    rng = random.Random(seed); means=[]; drawdowns=[]
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        equity=peak=1.0; dd=0.0
        for r in draw:
            equity *= 1+r; peak=max(peak,equity); dd=max(dd,(peak-equity)/peak)
        means.append(sum(draw)/len(draw)); drawdowns.append(dd)
    means.sort(); drawdowns.sort(); alpha=(1-confidence_level)/2
    lo=means[int(alpha*(samples-1))]; hi=means[int((1-alpha)*(samples-1))]
    dd_hi=drawdowns[int((1-alpha)*(samples-1))]
    return BootstrapResult(samples, confidence_level, sum(values)/len(values), lo, hi, sum(x < 0 for x in means)/samples, sum(drawdowns)/samples, dd_hi)
