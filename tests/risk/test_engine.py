from __future__ import annotations
import pytest
from src.risk.engine import RiskConfig, RiskResult, assess_risk

def base(**kwargs):
    return assess_risk(equity=100_000, entry_price=100, stop_price=98, net_expected_value=.02, **kwargs)

def test_positive_ev_and_valid_stop_are_approved():
    r=base(); assert isinstance(r,RiskResult); assert r.approved; assert r.reason=="risk limits passed"; assert r.risk_amount==pytest.approx(1000); assert r.position_notional==pytest.approx(50000)

def test_zero_and_negative_ev_are_rejected():
    assert base(net_expected_value=0).approved is False
    assert base(net_expected_value=-.01).approved is False

def test_invalid_stop_is_rejected():
    assert assess_risk(equity=100_000,entry_price=100,stop_price=100,net_expected_value=.02).approved is False
    assert assess_risk(equity=100_000,entry_price=100,stop_price=101,net_expected_value=.02).approved is False

def test_position_is_capped():
    r=base(config=RiskConfig(max_position_fraction=.25)); assert r.position_notional==pytest.approx(25000)

def test_custom_risk_fraction_changes_budget():
    r=base(config=RiskConfig(max_risk_fraction=.02)); assert r.risk_amount==pytest.approx(2000)

@pytest.mark.parametrize(("equity","entry_price","stop_price"),[(0,100,98),(-1,100,98),(100000,0,-1),(100000,100,0)])
def test_non_positive_inputs_rejected(equity,entry_price,stop_price):
    with pytest.raises(ValueError): assess_risk(equity=equity,entry_price=entry_price,stop_price=stop_price,net_expected_value=.02)

@pytest.mark.parametrize("value",[float("nan"),float("inf"),float("-inf")])
def test_nonfinite_ev_rejected(value):
    with pytest.raises(ValueError): base(net_expected_value=value)

@pytest.mark.parametrize("config",[RiskConfig(max_risk_fraction=0),RiskConfig(max_position_fraction=0),RiskConfig(max_drawdown_fraction=0)])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError): base(config=config)

def test_drawdown_limit_is_enforced():
    config=RiskConfig(max_drawdown_fraction=.20)
    r=base(config=config,current_drawdown_fraction=.20)
    assert r.approved is False; assert r.reason=="max drawdown limit reached"

def test_drawdown_below_limit_is_allowed():
    r=base(config=RiskConfig(max_drawdown_fraction=.20),current_drawdown_fraction=.199)
    assert r.approved is True

def test_invalid_current_drawdown_is_rejected():
    with pytest.raises(ValueError): base(current_drawdown_fraction=1.01)
