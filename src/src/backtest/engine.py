"""Deterministic long-only backtest engine for quantitative research."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a deterministic research backtest."""

    initial_equity: float = 100_000.0
    fee_fraction: float = 0.0
    slippage_fraction: float = 0.0


@dataclass(frozen=True)
class BacktestTrade:
    """One completed long-only research trade."""

    entry_price: float
    exit_price: float
    gross_return: float
    total_cost: float
    net_return: float
    equity_before: float
    equity_after: float


@dataclass(frozen=True)
class BacktestResult:
    """Complete deterministic backtest result."""

    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    equity_curve: tuple[float, ...]
    trades: tuple[BacktestTrade, ...]


def run_backtest(
    *,
    prices: Sequence[float],
    signals: Sequence[int],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a one-period-hold, long-only backtest.

    A signal of 1 at index i means:
        enter at prices[i]
        exit at prices[i + 1]

    A signal of 0 means no trade.

    Only information available at index i is used to determine the
    position for the following period. No future price is used to
    generate the signal.

    The engine is research-only. It does not place orders and does not
    use leverage.
    """
    if config is None:
        config = BacktestConfig()

    _validate_config(config)
    validated_prices = _validate_prices(prices)
    validated_signals = _validate_signals(signals)

    if len(validated_prices) != len(validated_signals):
        raise ValueError(
            "prices and signals must contain the same number "
            "of observations"
        )

    if len(validated_prices) < 2:
        raise ValueError(
            "prices and signals must contain at least 2 observations"
        )

    equity = config.initial_equity
    equity_curve = [equity]
    trades: list[BacktestTrade] = []

    for index in range(len(validated_prices) - 1):
        signal = validated_signals[index]

        if signal == 0:
            equity_curve.append(equity)
            continue

        entry_price = validated_prices[index]
        exit_price = validated_prices[index + 1]

        gross_return = (
            exit_price - entry_price
        ) / entry_price

        total_cost = (
            2.0
            * (config.fee_fraction + config.slippage_fraction)
        )

        net_return = gross_return - total_cost

        equity_before = equity
        equity = equity * (1.0 + net_return)

        trades.append(
            BacktestTrade(
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=gross_return,
                total_cost=total_cost,
                net_return=net_return,
                equity_before=equity_before,
                equity_after=equity,
            )
        )

        equity_curve.append(equity)

    total_return = (
        equity / config.initial_equity
    ) - 1.0

    max_drawdown = _calculate_max_drawdown(
        equity_curve
    )

    winning_trades = sum(
        1
        for trade in trades
        if trade.net_return > 0.0
    )

    losing_trades = sum(
        1
        for trade in trades
        if trade.net_return <= 0.0
    )

    win_rate = (
        winning_trades / len(trades)
        if trades
        else 0.0
    )

    return BacktestResult(
        initial_equity=config.initial_equity,
        final_equity=equity,
        total_return=total_return,
        max_drawdown=max_drawdown,
        trade_count=len(trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        equity_curve=tuple(equity_curve),
        trades=tuple(trades),
    )


def _validate_config(config: BacktestConfig) -> None:
    _validate_positive(
        config.initial_equity,
        "initial_equity",
    )
    _validate_fraction(
        config.fee_fraction,
        "fee_fraction",
    )
    _validate_fraction(
        config.slippage_fraction,
        "slippage_fraction",
    )


def _validate_prices(
    prices: Sequence[float],
) -> list[float]:
    validated: list[float] = []

    for index, price in enumerate(prices):
        if isinstance(price, bool) or not isinstance(
            price,
            (int, float),
        ):
            raise ValueError(
                f"price at index {index} must be numeric"
            )

        value = float(price)

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"price at index {index} must be finite and positive"
            )

        validated.append(value)

    return validated


def _validate_signals(
    signals: Sequence[int],
) -> list[int]:
    validated: list[int] = []

    for index, signal in enumerate(signals):
        if isinstance(signal, bool) or not isinstance(
            signal,
            int,
        ):
            raise ValueError(
                f"signal at index {index} must be an integer"
            )

        if signal not in (0, 1):
            raise ValueError(
                f"signal at index {index} must be 0 or 1"
            )

        validated.append(signal)

    return validated


def _validate_positive(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    numeric = float(value)

    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive"
        )


def _validate_fraction(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{name} must be numeric"
        )

    numeric = float(value)

    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"{name} must be finite and between 0 and 1"
        )


def _calculate_max_drawdown(
    equity_curve: Sequence[float],
) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        drawdown = (
            peak - equity
        ) / peak

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown
