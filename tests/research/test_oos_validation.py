import pytest
from src.research.oos_validation import validate_oos

def test_oos_reports_probability_and_cost_metrics():
    r=validate_oos(actual=[0,1,1,0], probabilities=[.2,.8,.7,.3], baseline_probabilities=[.5]*4, prices=[100,101,100,102], signals=[1,0,1,0], fee_fraction=.001, slippage_fraction=.001)
    assert r.observations==4; assert 0<=r.brier_score<=1; assert r.baseline_brier_score is not None; assert r.trade_count==2; assert r.fees>0; assert r.slippage>0

def test_oos_requires_aligned_inputs():
    with pytest.raises(ValueError): validate_oos(actual=[0], probabilities=[.5,.5])

def test_oos_rejects_empty():
    with pytest.raises(ValueError): validate_oos(actual=[], probabilities=[])

def test_oos_rejects_bad_signal():
    with pytest.raises(ValueError, match="signals"): validate_oos(actual=[0,1], probabilities=[.5,.5], prices=[1,2], signals=[2,0])
