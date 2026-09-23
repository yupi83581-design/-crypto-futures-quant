from __future__ import annotations

import pytest
from src.risk.engine import assess_risk, RiskConfig
from src.market_integrity.detector import OrderBookLevel, assess_market_integrity
from src.paper.engine import PaperTradingEngine

def test_ev_zero_blocks():
    assert not assess_risk(equity=100000, entry_price=100, stop_price=98, net_expected_value=0).approved

def test_stop_at_or_above_entry_blocks():
    for stop in (100, 101):
        assert not assess_risk(equity=100000, entry_price=100, stop_price=stop, net_expected_value=0.1).approved

def test_drawdown_limit_blocks():
    r = assess_risk(equity=100000, entry_price=100, stop_price=98, net_expected_value=0.1, current_drawdown_fraction=0.20)
    assert not r.approved and r.reason == "drawdown limit exceeded"

def test_position_cap_is_enforced():
    r = assess_risk(equity=100000, entry_price=100, stop_price=99, net_expected_value=0.1, config=RiskConfig(max_position_fraction=0.25))
    assert r.approved and r.position_fraction <= 0.25

def test_invalid_equity_and_price_fail_closed():
    with pytest.raises(ValueError):
        assess_risk(equity=0, entry_price=100, stop_price=98, net_expected_value=0.1)
    with pytest.raises(ValueError):
        assess_risk(equity=100000, entry_price=float("nan"), stop_price=98, net_expected_value=0.1)

def test_suspicious_integrity_is_detectable():
    levels = [OrderBookLevel(100.0, 100.0), OrderBookLevel(99.9, 1.0), OrderBookLevel(99.8, 1.0)]
    assert assess_market_integrity(bid_depth=levels, ask_depth=levels).status == "SUSPICIOUS"

def test_duplicate_paper_position_blocks():
    engine = PaperTradingEngine()
    engine.open_long(entry_price=100, quantity=1)
    with pytest.raises(ValueError, match="already open"):
        engine.open_long(entry_price=101, quantity=1)

def test_position_mismatch_blocks_close():
    with pytest.raises(ValueError, match="no paper position"):
        PaperTradingEngine().close_position(exit_price=100)

def test_invalid_mark_price_fails_closed():
    with pytest.raises(ValueError):
        PaperTradingEngine().unrealized_pnl(mark_price=0)
