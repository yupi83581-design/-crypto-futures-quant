import pytest
from src.research.final_validation import GateStatus, REQUIRED_STAGES, evaluate_final_gate

def all_status(status): return {s: status for s in REQUIRED_STAGES}

def test_final_gate_passes_only_when_every_stage_passes():
    r=evaluate_final_gate(all_status(GateStatus.PASS)); assert r.status is GateStatus.PASS; assert r.passed

def test_final_gate_fails_on_failure():
    e=all_status(GateStatus.PASS); e['risk']=GateStatus.FAIL; r=evaluate_final_gate(e); assert r.status is GateStatus.FAIL; assert 'risk' in r.reasons[0]

def test_final_gate_is_insufficient_when_missing_or_unverified():
    e=all_status(GateStatus.PASS); del e['oos']; assert evaluate_final_gate(e).status is GateStatus.INSUFFICIENT_EVIDENCE
    e=all_status(GateStatus.PASS); e['oos']=GateStatus.INSUFFICIENT_EVIDENCE; assert evaluate_final_gate(e).status is GateStatus.INSUFFICIENT_EVIDENCE

def test_final_gate_rejects_unknown_status():
    e=all_status(GateStatus.PASS); e['risk']='UNKNOWN'
    with pytest.raises(ValueError): evaluate_final_gate(e)
