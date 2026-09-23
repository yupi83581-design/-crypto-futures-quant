"""Generate the CTO forensic authorization artifact from current evidence only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "evidence" / "final_validation.json"
PAPER = ROOT / "evidence" / "paper_evidence_audit.json"
PHASE10 = ROOT / "evidence" / "phase10_validation_refresh.json"
SECURITY = ROOT / "evidence" / "security_execution_audit.json"
ACCOUNTING = ROOT / "evidence" / "phase10_accounting_audit.json"
MATH = ROOT / "evidence" / "quant_math_audit.json"
OUT = ROOT / "evidence" / "cto_forensic_authorization.json"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    final = load(FINAL)
    paper = load(PAPER)
    phase10 = load(PHASE10) or load(ROOT / "evidence" / "phase10_backtest.json")
    security = load(SECURITY)
    accounting = load(ACCOUNTING)
    math_audit = load(MATH)

    final_status = final.get("status") if final else "INSUFFICIENT_EVIDENCE"
    security_status = security.get("status") if security else "INSUFFICIENT_EVIDENCE"
    paper_status = paper.get("status") if paper else "INSUFFICIENT_EVIDENCE"

    gates = {
        "accounting_forensic": accounting.get("status") if accounting else "INSUFFICIENT_EVIDENCE",
        "paper_evidence_forensic": paper_status,
        "dsr_mathematical_audit": math_audit.get("dsr", {}).get("status", "INSUFFICIENT_EVIDENCE") if math_audit else "INSUFFICIENT_EVIDENCE",
        "pbo_cscv_mathematical_audit": math_audit.get("pbo_cscv", {}).get("status", "INSUFFICIENT_EVIDENCE") if math_audit else "INSUFFICIENT_EVIDENCE",
        "quantitative_validation": final_status,
        "risk_failure_tests": "INSUFFICIENT_EVIDENCE",
        "runtime_snapshot_audit": "INSUFFICIENT_EVIDENCE",
        "security_execution_isolation": security_status,
        "final_authorization": "PASS" if final_status == "PASS" and security_status == "PASS" and paper_status == "PASS" else "FAIL",
    }

    report = {
        "status": "AUTHORIZED" if gates["final_authorization"] == "PASS" else "BLOCKED",
        "generated_by": "CTO forensic authorization generator",
        "execution_verified": False,
        "real_money_execution": False,
        "authorization": "AUTHORIZE_REAL_MONEY" if gates["final_authorization"] == "PASS" else "DO_NOT_AUTHORIZE_REAL_MONEY",
        "gates": gates,
        "source_files": {
            "final_validation": str(FINAL.relative_to(ROOT)) if FINAL.exists() else None,
            "paper_audit": str(PAPER.relative_to(ROOT)) if PAPER.exists() else None,
            "phase10": str((PHASE10 if PHASE10.exists() else ROOT / "evidence" / "phase10_backtest.json").relative_to(ROOT)) if phase10 else None,
            "security_audit": str(SECURITY.relative_to(ROOT)) if SECURITY.exists() else None,
        },
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "AUTHORIZED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
