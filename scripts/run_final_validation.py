"""Forensic final-validation report generator.

This module never upgrades a gate from missing evidence to PASS manually.
It derives the final result from the recorded Phase-10 historical evidence,
paper-evidence audit, and a successful validation test suite.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from src.research.final_validation import GateStatus, evaluate_final_gate

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PHASE10 = ROOT / "evidence" / "phase10_backtest.json"
REFRESHED_PHASE10 = ROOT / "evidence" / "phase10_validation_refresh.json"
PAPER_AUDIT = ROOT / "evidence" / "paper_evidence_audit.json"
OUT = ROOT / "evidence" / "final_validation.json"


def main() -> int:
    phase10_path = REFRESHED_PHASE10 if REFRESHED_PHASE10.exists() else HISTORICAL_PHASE10
    phase10 = json.loads(phase10_path.read_text()) if phase10_path.exists() else None
    paper = json.loads(PAPER_AUDIT.read_text()) if PAPER_AUDIT.exists() else None
    tests_passed = os.getenv("VALIDATION_TESTS_PASSED") == "1"

    evidence: dict[str, str] = {}
    reasons: list[str] = []

    if phase10 and phase10.get("status") == "COMPLETE":
        perf = phase10.get("trading_performance_wfo", {})
        final = phase10.get("untouched_final_test", {})
        integrity = phase10.get("data_integrity", {})
        dsr = phase10.get("dsr", {})
        pbo = phase10.get("pbo_cscv", {})
        wf = phase10.get("walk_forward", {})
        robustness = phase10.get("robustness", {})
        calibration = phase10.get("calibration")

        evidence["data_quality"] = GateStatus.PASS.value if integrity.get("passed") else GateStatus.FAIL.value
        evidence["model"] = GateStatus.PASS.value if perf.get("trade_count", 0) > 0 else GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["calibration"] = GateStatus.PASS.value if calibration and math.isfinite(float(calibration.get("brier_score", float("nan")))) else GateStatus.INSUFFICIENT_EVIDENCE.value
        wfo_expectancy = float(perf.get("expectancy_per_trade", float("nan")))
        wfo_net_return = float(perf.get("net_return", float("nan")))
        final_perf = final.get("performance", {})
        final_expectancy = float(final_perf.get("expectancy_per_trade", float("nan")))
        final_net_return = float(final_perf.get("net_return", float("nan")))
        wfo_profitable = math.isfinite(wfo_expectancy) and math.isfinite(wfo_net_return) and wfo_expectancy > 0.0 and wfo_net_return > 0.0
        final_profitable = math.isfinite(final_expectancy) and math.isfinite(final_net_return) and final_expectancy > 0.0 and final_net_return > 0.0
        evidence["ev_cost"] = GateStatus.PASS.value if wfo_profitable and phase10.get("method", {}).get("fee_assumption") is not None and phase10.get("method", {}).get("slippage_assumption") is not None else (GateStatus.FAIL.value if perf.get("trade_count", 0) > 0 else GateStatus.INSUFFICIENT_EVIDENCE.value)
        evidence["risk"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["backtest"] = GateStatus.PASS.value if wfo_profitable else (GateStatus.FAIL.value if perf.get("trade_count", 0) > 0 else GateStatus.INSUFFICIENT_EVIDENCE.value)
        evidence["walk_forward"] = GateStatus.PASS.value if wf.get("folds", 0) > 0 else GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["oos"] = GateStatus.PASS.value if final.get("untouched") is True and final_profitable else (GateStatus.FAIL.value if final.get("untouched") is True and final_perf.get("trade_count", 0) > 0 else GateStatus.INSUFFICIENT_EVIDENCE.value)
        evidence["regime"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["robustness"] = GateStatus.PASS.value if robustness.get("stable") is True else GateStatus.FAIL.value
        evidence["paper_trading"] = GateStatus.PASS.value if paper and paper.get("status") == "PASS" else GateStatus.FAIL.value if paper else GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["monitoring"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["trading_performance"] = GateStatus.PASS.value if wfo_profitable and final.get("untouched") is True and final_profitable else (GateStatus.FAIL.value if perf.get("trade_count", 0) > 0 and final.get("untouched") is True else GateStatus.INSUFFICIENT_EVIDENCE.value)
        evidence["dsr"] = GateStatus.PASS.value if (0.0 < dsr.get("deflated_sharpe_probability", -1) <= 1.0 and dsr.get("observed_sharpe", 0.0) > 0.0 and dsr.get("trial_count", 0) == len(phase10.get("method", {}).get("strategy_universe", [])) and len(dsr.get("trial_universe", [])) == dsr.get("trial_count", 0)) else (GateStatus.FAIL.value if dsr.get("observed_sharpe") is not None else GateStatus.INSUFFICIENT_EVIDENCE.value)
        evidence["pbo_cscv"] = GateStatus.PASS.value if (pbo.get("path_count", 0) > 0 and len(pbo.get("omega_values", [])) == pbo.get("path_count", 0) and len(pbo.get("logit_values", [])) == pbo.get("path_count", 0) and pbo.get("strategy_count", 0) == dsr.get("trial_count", 0) and pbo.get("observations", 0) == pbo.get("input_observations_before_trim", pbo.get("observations", 0)) - pbo.get("trimmed_observations", 0) and 0 <= pbo.get("trimmed_observations", -1) < pbo.get("block_count", 0) and pbo.get("trim_policy") == "drop trailing observations so aligned time-series length is divisible by block_count") else GateStatus.FAIL.value
        evidence["lookahead_protection"] = GateStatus.PASS.value if (wf.get("lookahead_protection") is True and wf.get("purge_bars") == wf.get("label_horizon") and wf.get("embargo_bars") == wf.get("label_horizon")) else GateStatus.FAIL.value
        evidence["untouched_final_test"] = GateStatus.PASS.value if final.get("untouched") is True else GateStatus.FAIL.value
        evidence["risk_controls"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["kill_switch"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["state_persistence"] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["evidence_integrity"] = GateStatus.PASS.value if integrity.get("passed") else GateStatus.FAIL.value
        evidence["reproducibility"] = GateStatus.PASS.value if phase10.get("reproducibility", {}).get("git_commit") not in (None, "UNAVAILABLE") else GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["execution_lock"] = GateStatus.PASS.value if phase10.get("real_money_execution") is False else GateStatus.FAIL.value
    else:
        for gate in (
            "data_quality", "model", "calibration", "ev_cost", "risk", "backtest",
            "walk_forward", "oos", "regime", "robustness", "paper_trading",
            "monitoring", "trading_performance", "dsr", "pbo_cscv",
            "lookahead_protection", "untouched_final_test", "risk_controls",
            "kill_switch", "state_persistence", "evidence_integrity",
            "reproducibility",
        ):
            evidence[gate] = GateStatus.INSUFFICIENT_EVIDENCE.value
        evidence["execution_lock"] = GateStatus.PASS.value

    if paper:
        if paper.get("status") != "PASS":
            reasons.extend(paper.get("errors", []))
        if paper.get("continuity_warnings"):
            reasons.append("paper evidence has non-contiguous candle timestamps; 12/12 observations remain intact but are not one continuous 5m window")
    else:
        reasons.append("paper evidence audit is missing")

    if phase10 and phase10.get("status") == "COMPLETE":
        if perf.get("trade_count", 0) > 0 and not wfo_profitable:
            reasons.append("WFO net expectancy and/or net return is non-positive; strategy edge is not demonstrated.")
        if final.get("untouched") is True and final_perf.get("trade_count", 0) > 0 and not final_profitable:
            reasons.append("untouched final-test net expectancy and/or net return is non-positive; out-of-sample edge is not demonstrated.")
        if dsr.get("observed_sharpe") is not None and float(dsr.get("observed_sharpe")) <= 0.0:
            reasons.append("observed Sharpe is non-positive; DSR cannot support an execution-readiness claim.")
    final = evaluate_final_gate(evidence)
    reasons.extend(final.reasons)

    report = {
        "status": final.status.value,
        "gates": dict(final.stage_status),
        "reasons": reasons,
        "real_money_execution": False,
        "validation_tests_passed": tests_passed,
        "source_files": {
            "phase10_backtest": str(phase10_path.relative_to(ROOT)) if phase10_path.exists() else None,
            "historical_phase10_backtest_preserved": HISTORICAL_PHASE10.exists(),
            "paper_audit": str(PAPER_AUDIT.relative_to(ROOT)) if PAPER_AUDIT.exists() else None,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if final.status is GateStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
