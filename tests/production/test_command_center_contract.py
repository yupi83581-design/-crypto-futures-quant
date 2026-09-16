import pytest
from src.production.command_center_contract import build_snapshot

def test_snapshot_is_json_ready():
    r=build_snapshot(symbol="BTCUSDT",status="READY",probability=.7,risk_approved=True,observations=10)
    d=r.to_dict(); assert d["symbol"]=="BTCUSDT"; assert d["probability"]==.7

def test_snapshot_rejects_invalid_values():
    with pytest.raises(ValueError): build_snapshot(symbol="",status="READY")
    with pytest.raises(ValueError): build_snapshot(symbol="BTCUSDT",status="BAD")
    with pytest.raises(ValueError): build_snapshot(symbol="BTCUSDT",status="READY",probability=2)
    with pytest.raises(ValueError): build_snapshot(symbol="BTCUSDT",status="READY",observations=-1)
