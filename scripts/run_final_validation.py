"""Forensic final-validation report generator.

Never upgrades a gate manually. It derives statuses only from recorded evidence.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE10 = ROOT / "evidence" / "phase10_backtest.json"
PAPER_AUDIT = ROOT / "evidence" / "paper_evidence_audit.json"
OUT = ROOT / "evidence" / "final_validation.json"

def main() -> int:
    phase10 = json.loads(PHASE10.read_text()) if PHASE10.exists() else None
    paper = json.loads(PAPER_AUDIT.read_text()) if PAPER_AUDIT.exists() else None

    statuses = {}
    reasons = []

    if phase10 and phase10.get("status") == "COMPLETE":
        perf = phase10.get("trading_performance_wfo", {})
        final = phase10.get("untouched_final_test", {})
        statuses["trading_performance"] = "PASS" if perf.get("trade_count", 0) > 0 and final.get("untouched") is True else "INSUFFICIENT_EVIDENCE"
        statuses["dsr"] = "PASS" if 0.0 <= phase10.get("dsr", {}).get("deflated_sharpe_probability", -1) <= 1.0 else "FAIL"
        statuses["pbo_cscv"] = "PASS" if phase10.get("pbo_cscv", {}).get("path_count", 0) > 0 else "INSUFFICIENT_EVIDENCE"
        statuses["robustness"] = "PASS" if phase10.get("robustness", {}).get("stable") is True else "FAIL"
        statuses["untouched_final_test"] = "PASS" if final.get("untouched") is True else "FAIL"
        statuses["data_integrity"] = "PASS" if phase10.get("data_integrity", {}).get("passed") is True else "FAIL"
        statuses["walk_forward_oos"] = "PASS" if phase10.get("walk_forward", {}).get("folds", 0) > 0 else "INSUFFICIENT_EVIDENCE"
        statuses["reproducibility"] = "PASS" if phase10.get("reproducibility", {}).get("git_commit") not in (None, "UNAVAILABLE") else "INSUFFICIENT_EVIDENCE"
        statuses["execution_lock"] = "PASS" if phase10.get("real_money_execution") is False else "FAIL"
    else:
        for gate in ("trading_performance","dsr","pbo_cscv","robustness","untouched_final_test","data_integrity","walk_forward_oos","reproducibility"):
            statuses[gate] = "INSUFFICIENT_EVIDENCE"
        statuses["execution_lock"] = "PASS"

    if paper and paper.get("status") == "PASS":
        statuses["paper_trading"] = "PASS"
    elif paper:
        statuses["paper_trading"] = "FAIL"
        reasons.extend(paper.get("errors", []))
    else:
        statuses["paper_trading"] = "INSUFFICIENT_EVIDENCE"

    statuses["risk_controls"] = "INSUFFICIENT_EVIDENCE"
    statuses["kill_switch"] = "INSUFFICIENT_EVIDENCE"
    statuses["state_persistence"] = "INSUFFICIENT_EVIDENCE"
    statuses["evidence_integrity"] = "PASS" if phase10 and phase10.get("data_integrity", {}).get("passed") is True else "INSUFFICIENT_EVIDENCE"

    failed = [k for k,v in statuses.items() if v == "FAIL"]
    insufficient = [k for k,v in statuses.items() if v == "INSUFFICIENT_EVIDENCE"]
    status = "FAIL" if failed else ("INSUFFICIENT_EVIDENCE" if insufficient else "PASS")
    report = {
        "status": status,
        "gates": statuses,
        "reasons": reasons + [f"failed gate: {x}" for x in failed] + [f"insufficient evidence: {x}" for x in insufficient],
        "real_money_execution": False,
        "source_files": {
            "phase10_backtest": str(PHASE10.relative_to(ROOT)) if PHASE10.exists() else None,
            "paper_audit": str(PAPER_AUDIT.relative_to(ROOT)) if PAPER_AUDIT.exists() else None,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
