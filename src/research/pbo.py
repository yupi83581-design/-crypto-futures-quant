"""Combinatorially Symmetric Cross-Validation (CSCV) PBO."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class PBOResult:
    strategy_count: int
    observations: int
    block_count: int
    path_count: int
    overfit_paths: int
    pbo: float
    omega_values: tuple[float, ...]
    logit_values: tuple[float, ...]
    selected_strategy_indices: tuple[int, ...]


def pbo_cs_cv(
    strategy_returns: Sequence[Sequence[float]],
    *,
    block_count: int = 8,
) -> PBOResult:
    """Run formal CSCV and return the probability of backtest overfitting.

    Each strategy must contain the same time-ordered observations. The
    observations are split into S equal contiguous blocks (S must be even).
    Every S/2 block combination is used as IS; its complement is OOS.
    The IS winner is evaluated against all strategies on OOS. OOS ranks use
    average ranks for ties, with rank 1 being worst. Relative rank is
    omega = rank / (N + 1), keeping omega strictly inside (0, 1),
    and the logit is log(omega / (1 - omega)). PBO is the fraction of paths
    with logit < 0, i.e. the IS winner lands below the OOS median.
    """
    series = _validate_inputs(strategy_returns, block_count)
    n_strategies = len(series)
    n = len(series[0])
    if n < block_count:
        raise ValueError("observations must be at least block_count")
    if n % block_count:
        raise ValueError("observations must divide evenly into block_count")
    block_size = n // block_count
    if block_size < 2:
        raise ValueError("each CSCV block must contain at least 2 observations")

    combinations = list(itertools.combinations(range(block_count), block_count // 2))
    omega_values = []
    logit_values = []
    selected = []
    overfit = 0

    for is_blocks in combinations:
        oos_blocks = tuple(i for i in range(block_count) if i not in is_blocks)
        is_scores = [
            statistics.mean(_concat_blocks(s, is_blocks, block_size))
            for s in series
        ]
        winner = max(range(n_strategies), key=lambda i: (is_scores[i], -i))
        oos_scores = [
            statistics.mean(_concat_blocks(s, oos_blocks, block_size))
            for s in series
        ]
        winner_score = oos_scores[winner]
        worse = sum(score < winner_score for score in oos_scores)
        equal = sum(score == winner_score for score in oos_scores)
        # Rank 1 is worst and rank N is best, matching CSCV's
        # overfitting orientation: low OOS rank => negative logit.
        rank = 1.0 + worse + (equal - 1.0) / 2.0
        # Bailey/Lopez de Prado CSCV uses relative rank r/(N+1),
        # keeping omega strictly inside (0, 1) before the logit.
        omega = rank / (n_strategies + 1.0)
        logit = math.log(omega / (1.0 - omega))
        omega_values.append(omega)
        logit_values.append(logit)
        selected.append(winner)
        overfit += int(logit < 0.0)

    return PBOResult(
        strategy_count=n_strategies,
        observations=n,
        block_count=block_count,
        path_count=len(combinations),
        overfit_paths=overfit,
        pbo=overfit / len(combinations),
        omega_values=tuple(omega_values),
        logit_values=tuple(logit_values),
        selected_strategy_indices=tuple(selected),
    )


def _concat_blocks(series: Sequence[float], blocks: Sequence[int], size: int) -> list[float]:
    values = []
    for block in blocks:
        values.extend(series[block * size:(block + 1) * size])
    return values


def _validate_inputs(strategy_returns: Sequence[Sequence[float]], block_count: int) -> list[list[float]]:
    if len(strategy_returns) < 2:
        raise ValueError("at least 2 strategies are required")
    if not isinstance(block_count, int) or isinstance(block_count, bool) or block_count < 2 or block_count % 2:
        raise ValueError("block_count must be a positive even integer")
    lengths = {len(s) for s in strategy_returns}
    if len(lengths) != 1:
        raise ValueError("all strategies must have the same observation count")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("strategy returns must not be empty")
    result = []
    for si, series in enumerate(strategy_returns):
        values = []
        for i, value in enumerate(series):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"strategy {si} return {i} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"strategy {si} return {i} must be finite")
            values.append(value)
        result.append(values)
    return result
