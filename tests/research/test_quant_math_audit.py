from __future__ import annotations
import json
from pathlib import Path

def test_quant_math_audit_is_evidence_driven():
    path = Path("evidence/quant_math_audit.json")
    assert path.exists()
    report = json.loads(path.read_text())
    assert report["real_money_execution"] is False
    assert report["pbo_cscv"]["selected_strategy_indices_constant"] is True
    assert report["pbo_cscv"]["omega_unique_count"] == 1
    assert report["pbo_cscv"]["logit_unique_count"] == 1
