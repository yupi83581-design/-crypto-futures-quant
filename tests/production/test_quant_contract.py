import pytest
from src.production.quant_contract import ProductionQuantCycle

def test_probability_to_ev_risk_decision_contract():
    r=ProductionQuantCycle().evaluate(raw_probability=.8,reward=.03,loss=.02,fee=.001,slippage=.001,equity=100000,entry_price=100,stop_price=98)
    assert r.probability==pytest.approx(.8); assert r.expected_value.net_expected_value>0; assert r.risk.approved; assert r.decision.approved

def test_drawdown_blocks_cycle():
    r=ProductionQuantCycle().evaluate(raw_probability=.8,reward=.03,loss=.02,fee=.001,slippage=.001,equity=100000,entry_price=100,stop_price=98,current_drawdown_fraction=.2)
    assert not r.risk.approved; assert not r.decision.approved

def test_calibrator_is_applied_before_ev():
    class Cal:
        def predict(self,p): return [.6]
    r=ProductionQuantCycle(calibrator=Cal()).evaluate(raw_probability=.9,reward=.03,loss=.02,fee=.001,slippage=.001,equity=100000,entry_price=100,stop_price=98)
    assert r.probability==.6

def test_invalid_calibrator_rejected():
    with pytest.raises(TypeError): ProductionQuantCycle(calibrator=object())
