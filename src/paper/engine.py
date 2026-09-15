"""Deterministic paper-trading engine for quantitative research."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PaperConfig:
    """Configuration for a paper-trading account."""

    initial_equity: float = 100_000.0


@dataclass(frozen=True)
class PaperPosition:
    """Currently open virtual long position."""

    entry_price: float
    quantity: float
    entry_equity: float


@dataclass(frozen=True)
class PaperTrade:
    """Completed virtual long trade."""

    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    equity_before: float
    equity_after: float


@dataclass(frozen=True)
class PaperAccount:
    """Current immutable snapshot of the paper account."""

    equity: float
    position: PaperPosition | None
    completed_trades: tuple[PaperTrade, ...]


class PaperTradingEngine:
    """Stateful paper-trading engine with virtual execution only.

    The engine supports:
    - opening one virtual long position;
    - closing the virtual position;
    - calculating virtual PnL;
    - tracking virtual equity;
    - recording completed trades.

    It never places real orders and never connects to an exchange.
    """

    def __init__(
        self,
        config: PaperConfig | None = None,
    ) -> None:
        if config is None:
            config = PaperConfig()

        _validate_config(config)

        self._initial_equity = float(config.initial_equity)
        self._equity = float(config.initial_equity)
        self._position: PaperPosition | None = None
        self._completed_trades: list[PaperTrade] = []

    @property
    def account(self) -> PaperAccount:
        """Return the current immutable account snapshot."""
        return PaperAccount(
            equity=self._equity,
            position=self._position,
            completed_trades=tuple(
                self._completed_trades
            ),
        )

    @property
    def equity(self) -> float:
        """Return current virtual equity."""
        return self._equity

    @property
    def position(self) -> PaperPosition | None:
        """Return the current virtual position."""
        return self._position

    def open_long(
        self,
        *,
        entry_price: float,
        quantity: float,
    ) -> PaperPosition:
        """Open one virtual long position."""
        _validate_positive(
            entry_price,
            "entry_price",
        )
        _validate_positive(
            quantity,
            "quantity",
        )

        if self._position is not None:
            raise ValueError(
                "a paper position is already open"
            )

        position = PaperPosition(
            entry_price=float(entry_price),
            quantity=float(quantity),
            entry_equity=self._equity,
        )

        self._position = position
        return position

    def close_position(
        self,
        *,
        exit_price: float,
    ) -> PaperTrade:
        """Close the current virtual position."""
        _validate_positive(
            exit_price,
            "exit_price",
        )

        if self._position is None:
            raise ValueError(
                "no paper position is open"
            )

        position = self._position
        equity_before = self._equity

        gross_pnl = (
            exit_price - position.entry_price
        ) * position.quantity

        self._equity = (
            self._equity + gross_pnl
        )

        trade = PaperTrade(
            entry_price=position.entry_price,
            exit_price=float(exit_price),
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            equity_before=equity_before,
            equity_after=self._equity,
        )

        self._completed_trades.append(trade)
        self._position = None

        return trade

    def unrealized_pnl(
        self,
        *,
        mark_price: float,
    ) -> float:
        """Return unrealized PnL for the open position."""
        _validate_positive(
            mark_price,
            "mark_price",
        )

        if self._position is None:
            return 0.0

        return (
            mark_price - self._position.entry_price
        ) * self._position.quantity

    def reset(self) -> None:
        """Reset the paper account to its initial state."""
        self._equity = self._initial_equity
        self._position = None
        self._completed_trades.clear()


def _validate_config(config: PaperConfig) -> None:
    _validate_positive(
        config.initial_equity,
        "initial_equity",
    )


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
