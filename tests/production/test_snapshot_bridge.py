from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from src.decision.engine import DecisionResult
from src.production.command_center_contract import QuantCommandCenterSnapshot
from src.production.orchestrator import OrchestratorInput, ProductionOrchestrator
from src.production.snapshot_bridge import (
    build_command_center_snapshot,
    build_from_orchestrator_result,
)
from src.production.snapshot_http import create_snapshot_server
from src.risk.engine import RiskResult


def test_snapshot_aggregates_existing_results_without_recalculation():
    snapshot = build_command_center_snapshot(
        symbol="BTCUSDT",
        status="READY",
        market_data={"observations": 20, "timeframe": "5m"},
        inference={"probability": 0.73, "observations": 20},
        calibration={"brier_score": 0.12},
        expected_value_cost={"net_expected_value": 0.04},
        risk={"approved": True, "risk_fraction": 0.01},
        decision={"approved": True},
        paper_trading={"equity": 100.0},
        validation_status="INSUFFICIENT_EVIDENCE",
    )

    assert isinstance(snapshot, QuantCommandCenterSnapshot)
    assert snapshot.probability == 0.73
    assert snapshot.market_data["observations"] == 20
    assert snapshot.calibration["brier_score"] == 0.12
    assert snapshot.expected_value_cost["net_expected_value"] == 0.04
    assert snapshot.risk["approved"] is True
    assert snapshot.decision == "APPROVED"
    assert snapshot.paper_trading["equity"] == 100.0


def test_missing_domains_are_unavailable_and_not_zero():
    snapshot = build_command_center_snapshot(symbol="BTCUSDT", status="NO_TRADE")

    assert snapshot.probability is None
    assert snapshot.calibration == {"status": "UNAVAILABLE"}
    assert snapshot.expected_value_cost == {"status": "UNAVAILABLE"}
    assert snapshot.risk == {"status": "UNAVAILABLE"}
    assert snapshot.paper_trading == {"status": "UNAVAILABLE"}
    assert snapshot.walk_forward == {"status": "UNAVAILABLE"}
    assert snapshot.final_validation == {"status": "UNAVAILABLE"}
    assert 0 not in snapshot.calibration.values()


def test_inference_observations_can_feed_market_payload_without_calculation():
    snapshot = build_command_center_snapshot(
        symbol="BTCUSDT",
        status="READY",
        inference={
            "probability": 0.65,
            "observations": 20,
            "usable_observations": 3,
            "timeframe": "5m",
        },
    )

    assert snapshot.market_data == {
        "observations": 20,
        "usable_observations": 3,
        "timeframe": "5m",
    }


def test_incomplete_orchestrator_result_is_rejected():
    with pytest.raises(ValueError, match="incomplete orchestrator result"):
        build_from_orchestrator_result(object(), status="NO_TRADE")


def test_orchestrator_result_fields_are_mapped_directly():
    class Result:
        symbol = "BTCUSDT"
        model_output = {"probability": 0.81}
        decision = {"approved": False, "reason": "test"}
        risk = {"approved": False, "reason": "risk"}
        paper_result = {"equity": 99.0}
        monitoring_result = {"status": "OK"}

    snapshot = build_from_orchestrator_result(Result(), status="NO_TRADE")

    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.probability == 0.81
    assert snapshot.decision == "REJECTED"
    assert snapshot.risk["reason"] == "risk"
    assert snapshot.paper_trading["equity"] == 99.0
    assert snapshot.journal_monitoring["status"] == "OK"


def test_real_orchestrator_result_flows_into_snapshot_bridge():
    decision = DecisionResult(
        approved=True,
        reason="decision criteria passed",
        probability=0.72,
        expected_value=0.03,
        risk_approved=True,
    )
    risk = RiskResult(
        approved=True,
        reason="risk limits passed",
        risk_fraction=0.01,
        position_fraction=0.10,
        risk_amount=1.0,
        position_notional=10.0,
    )

    orchestrator = ProductionOrchestrator(
        feature_builder=lambda data: {"feature": data.prices[-1]},
        model=lambda features: {"probability": 0.72},
        decision_engine=lambda output: decision,
        risk_engine=lambda result: risk,
    )
    result = orchestrator.run(
        OrchestratorInput(
            symbol="BTCUSDT",
            prices=[100.0, 101.0, 102.0],
        )
    )

    snapshot = build_from_orchestrator_result(result, status="READY")

    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.probability == 0.72
    assert snapshot.decision == "APPROVED"
    assert snapshot.risk["approved"] is True
    assert snapshot.paper_trading == {"status": "UNAVAILABLE"}


def test_serialization_is_json_safe_and_read_only():
    snapshot = build_command_center_snapshot(
        symbol="BTCUSDT",
        status="READY",
        inference={"probability": 0.71},
    )
    encoded = json.dumps(snapshot.to_dict())
    decoded = json.loads(encoded)
    assert decoded["symbol"] == "BTCUSDT"
    assert decoded["probability"] == 0.71


def test_snapshot_provider_must_return_contract():
    server = create_snapshot_server("127.0.0.1", 0, lambda: {"fake": 1})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/snapshot"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url, timeout=2)
        assert error.value.code == 500
        payload = json.loads(error.value.read())
        assert payload["error"] == "snapshot unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_get_snapshot_returns_contract_json():
    snapshot = build_command_center_snapshot(
        symbol="BTCUSDT",
        status="READY",
        market_data={"observations": 20},
        inference={"probability": 0.77},
    )
    server = create_snapshot_server("127.0.0.1", 0, lambda: snapshot)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/snapshot"
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/json"
            payload = json.load(response)
        assert payload["symbol"] == "BTCUSDT"
        assert payload["market_data"]["observations"] == 20
        assert payload["probability"] == 0.77
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_get_snapshot_is_read_only():
    snapshot = build_command_center_snapshot(symbol="BTCUSDT", status="READY")
    server = create_snapshot_server("127.0.0.1", 0, lambda: snapshot)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/snapshot",
                method=method,
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=2)
            assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_unknown_path_returns_404():
    snapshot = build_command_center_snapshot(symbol="BTCUSDT", status="READY")
    server = create_snapshot_server("127.0.0.1", 0, lambda: snapshot)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/other",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
