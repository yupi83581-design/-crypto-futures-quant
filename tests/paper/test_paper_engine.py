import math

import pytest

from src.paper.engine import (
    PaperConfig,
    PaperTradingEngine,
)


def test_initial_account_has_configured_equity():
    engine = PaperTradingEngine(
        PaperConfig(initial_equity=200_000.0)
    )

    assert engine.equity == pytest.approx(200_000.0)
    assert engine.position is None
    assert engine.account.completed_trades == ()


def test_open_long_creates_virtual_position():
    engine = PaperTradingEngine()

    position = engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    assert position.entry_price == pytest.approx(100.0)
    assert position.quantity == pytest.approx(10.0)
    assert position.entry_equity == pytest.approx(100_000.0)
    assert engine.position == position


def test_profitable_long_trade_increases_equity():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    trade = engine.close_position(
        exit_price=110.0,
    )

    assert trade.gross_pnl == pytest.approx(100.0)
    assert trade.equity_before == pytest.approx(100_000.0)
    assert trade.equity_after == pytest.approx(100_100.0)
    assert engine.equity == pytest.approx(100_100.0)
    assert engine.position is None


def test_losing_long_trade_decreases_equity():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    trade = engine.close_position(
        exit_price=90.0,
    )

    assert trade.gross_pnl == pytest.approx(-100.0)
    assert trade.equity_before == pytest.approx(100_000.0)
    assert trade.equity_after == pytest.approx(99_900.0)
    assert engine.equity == pytest.approx(99_900.0)


def test_unrealized_profit_is_calculated():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    assert engine.unrealized_pnl(
        mark_price=105.0
    ) == pytest.approx(50.0)


def test_unrealized_loss_is_calculated():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    assert engine.unrealized_pnl(
        mark_price=95.0
    ) == pytest.approx(-50.0)


def test_unrealized_pnl_is_zero_without_position():
    engine = PaperTradingEngine()

    assert engine.unrealized_pnl(
        mark_price=100.0
    ) == pytest.approx(0.0)


def test_cannot_open_second_position():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    with pytest.raises(
        ValueError,
        match="already open",
    ):
        engine.open_long(
            entry_price=105.0,
            quantity=5.0,
        )


def test_cannot_close_without_position():
    engine = PaperTradingEngine()

    with pytest.raises(
        ValueError,
        match="no paper position",
    ):
        engine.close_position(
            exit_price=110.0
        )


def test_completed_trade_is_recorded():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    trade = engine.close_position(
        exit_price=110.0,
    )

    assert engine.account.completed_trades == (trade,)
    assert len(engine.account.completed_trades) == 1


def test_multiple_completed_trades_compound_equity():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )
    first_trade = engine.close_position(
        exit_price=110.0,
    )

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )
    second_trade = engine.close_position(
        exit_price=120.0,
    )

    assert first_trade.gross_pnl == pytest.approx(100.0)
    assert second_trade.gross_pnl == pytest.approx(200.0)
    assert engine.equity == pytest.approx(100_300.0)
    assert len(engine.account.completed_trades) == 2


def test_reset_restores_initial_state():
    engine = PaperTradingEngine(
        PaperConfig(initial_equity=200_000.0)
    )

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )
    engine.close_position(
        exit_price=110.0,
    )

    engine.reset()

    assert engine.equity == pytest.approx(200_000.0)
    assert engine.position is None
    assert engine.account.completed_trades == ()


@pytest.mark.parametrize(
    "entry_price, quantity",
    [
        (0.0, 10.0),
        (-1.0, 10.0),
        (float("nan"), 10.0),
        (float("inf"), 10.0),
        (100.0, 0.0),
        (100.0, -1.0),
        (100.0, float("nan")),
        (100.0, float("inf")),
    ],
)
def test_invalid_open_long_values_are_rejected(
    entry_price,
    quantity,
):
    engine = PaperTradingEngine()

    with pytest.raises(ValueError):
        engine.open_long(
            entry_price=entry_price,
            quantity=quantity,
        )


@pytest.mark.parametrize(
    "exit_price",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_exit_price_is_rejected(exit_price):
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    with pytest.raises(ValueError):
        engine.close_position(
            exit_price=exit_price
        )


@pytest.mark.parametrize(
    "mark_price",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_mark_price_is_rejected(mark_price):
    engine = PaperTradingEngine()

    with pytest.raises(ValueError):
        engine.unrealized_pnl(
            mark_price=mark_price
        )


@pytest.mark.parametrize(
    "config",
    [
        PaperConfig(initial_equity=0.0),
        PaperConfig(initial_equity=-1.0),
        PaperConfig(initial_equity=float("nan")),
        PaperConfig(initial_equity=float("inf")),
    ],
)
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        PaperTradingEngine(config)


def test_trade_accounting_is_complete():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=250.0,
        quantity=4.0,
    )

    trade = engine.close_position(
        exit_price=275.0,
    )

    assert trade.entry_price == pytest.approx(250.0)
    assert trade.exit_price == pytest.approx(275.0)
    assert trade.quantity == pytest.approx(4.0)
    assert trade.gross_pnl == pytest.approx(100.0)
    assert trade.equity_before == pytest.approx(100_000.0)
    assert trade.equity_after == pytest.approx(100_100.0)


def test_results_remain_finite():
    engine = PaperTradingEngine()

    engine.open_long(
        entry_price=100.0,
        quantity=10.0,
    )

    trade = engine.close_position(
        exit_price=110.0,
    )

    assert math.isfinite(trade.gross_pnl)
    assert math.isfinite(trade.equity_before)
    assert math.isfinite(trade.equity_after)
    assert math.isfinite(engine.equity)
