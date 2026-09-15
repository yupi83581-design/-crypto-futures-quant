import math

import pytest

from src.backtest.engine import (
    BacktestConfig,
    run_backtest,
)


def test_profitable_long_trade_increases_equity():
    result = run_backtest(
        prices=[100.0, 110.0],
        signals=[1, 0],
    )

    assert result.trade_count == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.final_equity == pytest.approx(110_000.0)
    assert result.total_return == pytest.approx(0.10)
    assert result.win_rate == pytest.approx(1.0)


def test_losing_long_trade_decreases_equity():
    result = run_backtest(
        prices=[100.0, 90.0],
        signals=[1, 0],
    )

    assert result.trade_count == 1
    assert result.winning_trades == 0
    assert result.losing_trades == 1
    assert result.final_equity == pytest.approx(90_000.0)
    assert result.total_return == pytest.approx(-0.10)
    assert result.win_rate == pytest.approx(0.0)


def test_zero_signal_produces_no_trade():
    result = run_backtest(
        prices=[100.0, 110.0, 120.0],
        signals=[0, 0, 0],
    )

    assert result.trade_count == 0
    assert result.final_equity == pytest.approx(100_000.0)
    assert result.total_return == pytest.approx(0.0)
    assert result.win_rate == pytest.approx(0.0)


def test_only_signal_before_final_price_can_create_trade():
    result = run_backtest(
        prices=[100.0, 110.0, 120.0],
        signals=[1, 1, 1],
    )

    assert result.trade_count == 2
    assert result.final_equity == pytest.approx(132_000.0)


def test_fee_and_slippage_reduce_return():
    result = run_backtest(
        prices=[100.0, 110.0],
        signals=[1, 0],
        config=BacktestConfig(
            fee_fraction=0.001,
            slippage_fraction=0.001,
        ),
    )

    assert result.trade_count == 1
    assert result.trades[0].total_cost == pytest.approx(0.004)
    assert result.trades[0].net_return == pytest.approx(0.096)
    assert result.final_equity == pytest.approx(109_600.0)


def test_equity_curve_is_recorded():
    result = run_backtest(
        prices=[100.0, 110.0, 99.0],
        signals=[1, 1, 0],
    )

    assert len(result.equity_curve) == 3
    assert result.equity_curve[0] == pytest.approx(100_000.0)
    assert result.equity_curve[1] == pytest.approx(110_000.0)
    assert result.equity_curve[2] == pytest.approx(99_000.0)


def test_max_drawdown_is_calculated():
    result = run_backtest(
        prices=[100.0, 110.0, 99.0],
        signals=[1, 1, 0],
    )

    assert result.max_drawdown == pytest.approx(
        0.10,
    )


def test_trade_contains_full_accounting():
    result = run_backtest(
        prices=[100.0, 105.0],
        signals=[1, 0],
    )

    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(105.0)
    assert trade.gross_return == pytest.approx(0.05)
    assert trade.total_cost == pytest.approx(0.0)
    assert trade.net_return == pytest.approx(0.05)
    assert trade.equity_before == pytest.approx(100_000.0)
    assert trade.equity_after == pytest.approx(105_000.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number"):
        run_backtest(
            prices=[100.0, 110.0],
            signals=[1],
        )


def test_too_few_observations_are_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        run_backtest(
            prices=[100.0],
            signals=[1],
        )


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, 0.0],
        [100.0, -1.0],
        [100.0, float("nan")],
        [100.0, float("inf")],
    ],
)
def test_invalid_prices_are_rejected(prices):
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        run_backtest(
            prices=prices,
            signals=[1, 0],
        )


@pytest.mark.parametrize(
    "signals",
    [
        [1, 2],
        [1, -1],
        [1.0, 0],
        [True, 0],
    ],
)
def test_invalid_signals_are_rejected(signals):
    with pytest.raises(ValueError):
        run_backtest(
            prices=[100.0, 110.0],
            signals=signals,
        )


@pytest.mark.parametrize(
    "config",
    [
        BacktestConfig(initial_equity=0.0),
        BacktestConfig(initial_equity=-1.0),
        BacktestConfig(fee_fraction=-0.1),
        BacktestConfig(fee_fraction=1.1),
        BacktestConfig(slippage_fraction=-0.1),
        BacktestConfig(slippage_fraction=1.1),
    ],
)
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        run_backtest(
            prices=[100.0, 110.0],
            signals=[1, 0],
            config=config,
        )


def test_all_results_are_finite():
    result = run_backtest(
        prices=[100.0, 105.0, 100.0],
        signals=[1, 1, 0],
    )

    assert math.isfinite(result.final_equity)
    assert math.isfinite(result.total_return)
    assert math.isfinite(result.max_drawdown)
    assert math.isfinite(result.win_rate)
