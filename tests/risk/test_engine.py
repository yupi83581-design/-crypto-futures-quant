from __future__ import annotations

import pytest

from src.risk.engine import (
    RiskConfig,
    RiskResult,
    assess_risk,
)


def test_positive_ev_and_valid_stop_are_approved():
    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=98,
        net_expected_value=0.02,
    )

    assert isinstance(result, RiskResult)
    assert result.approved is True
    assert result.reason == "risk limits passed"
    assert result.risk_fraction == pytest.approx(0.01)
    assert result.risk_amount == pytest.approx(1_000)
    assert result.position_notional == pytest.approx(50_000)
    assert result.position_fraction == pytest.approx(0.5)


def test_zero_expected_value_is_rejected():
    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=98,
        net_expected_value=0,
    )

    assert result.approved is False
    assert result.reason == "net_expected_value must be positive"
    assert result.position_notional == 0
    assert result.risk_amount == 0


def test_negative_expected_value_is_rejected():
    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=98,
        net_expected_value=-0.01,
    )

    assert result.approved is False
    assert result.reason == "net_expected_value must be positive"


def test_stop_at_entry_is_rejected():
    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=100,
        net_expected_value=0.02,
    )

    assert result.approved is False
    assert result.reason == "stop_price must be below entry_price"


def test_stop_above_entry_is_rejected():
    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=101,
        net_expected_value=0.02,
    )

    assert result.approved is False
    assert result.reason == "stop_price must be below entry_price"


def test_position_is_capped_by_max_position_fraction():
    config = RiskConfig(
        max_risk_fraction=0.01,
        max_position_fraction=0.25,
    )

    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=99,
        net_expected_value=0.02,
        config=config,
    )

    assert result.approved is True
    assert result.position_notional == pytest.approx(25_000)
    assert result.position_fraction == pytest.approx(0.25)
    assert result.risk_amount == pytest.approx(1_000)


def test_custom_risk_fraction_changes_risk_budget():
    config = RiskConfig(
        max_risk_fraction=0.02,
        max_position_fraction=1.0,
    )

    result = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=98,
        net_expected_value=0.02,
        config=config,
    )

    assert result.approved is True
    assert result.risk_fraction == pytest.approx(0.02)
    assert result.risk_amount == pytest.approx(2_000)
    assert result.position_notional == pytest.approx(100_000)


@pytest.mark.parametrize(
    ("equity", "entry_price", "stop_price"),
    [
        (0, 100, 98),
        (-1, 100, 98),
        (100_000, 0, -1),
        (100_000, 100, 0),
    ],
)
def test_non_positive_inputs_are_rejected_with_value_error(
    equity,
    entry_price,
    stop_price,
):
    with pytest.raises(ValueError):
        assess_risk(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            net_expected_value=0.02,
        )


@pytest.mark.parametrize(
    "net_expected_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_expected_value_is_rejected(net_expected_value):
    with pytest.raises(ValueError):
        assess_risk(
            equity=100_000,
            entry_price=100,
            stop_price=98,
            net_expected_value=net_expected_value,
        )


@pytest.mark.parametrize(
    "config",
    [
        RiskConfig(max_risk_fraction=0),
        RiskConfig(max_risk_fraction=-0.01),
        RiskConfig(max_risk_fraction=1.01),
        RiskConfig(max_position_fraction=0),
        RiskConfig(max_position_fraction=-0.1),
        RiskConfig(max_position_fraction=1.01),
        RiskConfig(max_drawdown_fraction=0),
        RiskConfig(max_drawdown_fraction=-0.1),
        RiskConfig(max_drawdown_fraction=1.01),
    ],
)
def test_invalid_risk_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        assess_risk(
            equity=100_000,
            entry_price=100,
            stop_price=98,
            net_expected_value=0.02,
            config=config,
        )


def test_boolean_numeric_input_is_rejected():
    with pytest.raises(ValueError):
        assess_risk(
            equity=True,
            entry_price=100,
            stop_price=98,
            net_expected_value=0.02,
        )


def test_position_sizing_uses_stop_distance():
    tight_stop = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=99,
        net_expected_value=0.02,
    )

    wide_stop = assess_risk(
        equity=100_000,
        entry_price=100,
        stop_price=95,
        net_expected_value=0.02,
    )

    assert tight_stop.approved is True
    assert wide_stop.approved is True

    assert tight_stop.position_notional == pytest.approx(100_000)
    assert wide_stop.position_notional == pytest.approx(20_000)

    assert tight_stop.position_notional > wide_stop.position_notional
