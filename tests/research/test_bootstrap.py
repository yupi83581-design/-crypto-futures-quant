import pytest
from src.research.bootstrap import bootstrap_trade_returns

def test_bootstrap_is_deterministic():
    r=bootstrap_trade_returns(trade_returns=[.02,-.01,.03], samples=100, seed=7)
    assert r.samples==100; assert r.lower_bound<=r.mean_return<=r.upper_bound
    assert r==bootstrap_trade_returns(trade_returns=[.02,-.01,.03], samples=100, seed=7)

def test_bootstrap_rejects_invalid_inputs():
    with pytest.raises(ValueError): bootstrap_trade_returns(trade_returns=[])
    with pytest.raises(ValueError): bootstrap_trade_returns(trade_returns=[.1], samples=0)
    with pytest.raises(ValueError): bootstrap_trade_returns(trade_returns=[.1], confidence_level=1)
