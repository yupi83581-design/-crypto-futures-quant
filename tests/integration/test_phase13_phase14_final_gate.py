from __future__ import annotations

import json
import threading
import urllib.request

from src.decision.engine import DecisionResult
from src.production.command_center_contract import QuantCommandCenterSnapshot
from src.production.orchestrator import OrchestratorInput, ProductionOrchestrator
from src.production.snapshot_bridge import build_from_orchestrator_result
from src.production.snapshot_http import create_snapshot_server
from src.risk.engine import RiskResult
from src.research.final_validation import (
    GateStatus,
    REQUIRED_STAGES,
    evaluate_final_gate,
)


def test_phase13_to_phase14_to_final_gate_read_only_flow():
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

    evidence = {stage: GateStatus.PASS for stage in REQUIRED_STAGES}
    evidence["paper_trading"] = GateStatus.INSUFFICIENT_EVIDENCE
    final_gate = evaluate_final_gate(evidence)

    snapshot = build_from_orchestrator_result(
        result,
        status="READY",
        final_validation=final_gate,
        validation_status=final_gate.status.value,
    )

    assert isinstance(snapshot, QuantCommandCenterSnapshot)
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.probability == 0.72
    assert snapshot.decision == "APPROVED"
    assert snapshot.risk["approved"] is True
    assert snapshot.validation_status == "INSUFFICIENT_EVIDENCE"
    assert snapshot.final_validation["status"] == "INSUFFICIENT_EVIDENCE"

    server = create_snapshot_server("127.0.0.1", 0, lambda: snapshot)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/snapshot",
            timeout=2,
        ) as response:
            payload = json.load(response)

        assert payload["symbol"] == "BTCUSDT"
        assert payload["validation_status"] == "INSUFFICIENT_EVIDENCE"
        assert payload["final_validation"]["status"] == "INSUFFICIENT_EVIDENCE"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
