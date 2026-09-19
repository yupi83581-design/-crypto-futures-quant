import pytest
from src.production.command_center_contract import build_snapshot


def test_snapshot_is_json_ready_and_covers_all_dashboard_domains():
    result = build_snapshot(
        symbol="BTCUSDT",
        status="READY",
        market_data={"status": "LIVE", "observations": 10},
        probability=0.7,
        calibration={"status": "CALIBRATED", "brier": 0.12},
        expected_value_cost={"expected_value": 0.03, "fees": 0.001, "slippage": 0.001},
        risk={"status": "APPROVED", "drawdown": 0.02},
        decision="LONG",
        backtest_oos={"net_return": 0.08},
        walk_forward={"folds": 4},
        robustness={"stability": 0.9},
        paper_trading={"state": "RUNNING", "equity": 101.0},
        journal_monitoring={"events": 12, "health": "OK"},
        final_validation={"status": "INSUFFICIENT_EVIDENCE"},
        validation_status="INSUFFICIENT_EVIDENCE",
    )
    data = result.to_dict()
    assert data["symbol"] == "BTCUSDT"
    assert data["probability"] == 0.7
    assert data["market_data"]["status"] == "LIVE"
    assert data["calibration"]["brier"] == 0.12
    assert data["expected_value_cost"]["fees"] == 0.001
    assert data["risk"]["status"] == "APPROVED"
    assert data["backtest_oos"]["net_return"] == 0.08
    assert data["walk_forward"]["folds"] == 4
    assert data["robustness"]["stability"] == 0.9
    assert data["paper_trading"]["state"] == "RUNNING"
    assert data["journal_monitoring"]["health"] == "OK"
    assert data["final_validation"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_legacy_compatibility_fields_are_mapped():
    result = build_snapshot(
        symbol="BTCUSDT",
        status="READY",
        probability=0.7,
        paper_equity=101.0,
        observations=10,
    )
    data = result.to_dict()
    assert data["paper_trading"]["equity"] == 101.0
    assert data["market_data"]["observations"] == 10


def test_snapshot_rejects_invalid_values():
    with pytest.raises(ValueError):
        build_snapshot(symbol="", status="READY")
    with pytest.raises(ValueError):
        build_snapshot(symbol="BTCUSDT", status="BAD")
    with pytest.raises(ValueError):
        build_snapshot(symbol="BTCUSDT", status="READY", probability=2)
    with pytest.raises(ValueError):
        build_snapshot(symbol="BTCUSDT", status="READY", observations=-1)
