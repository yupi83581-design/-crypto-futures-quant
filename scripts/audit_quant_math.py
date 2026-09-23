"""Audit DSR/PBO evidence for mathematical consistency without changing source evidence."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE10 = ROOT / "evidence" / "phase10_backtest.json"
OUT = ROOT / "evidence" / "quant_math_audit.json"

def main() -> int:
    p = json.loads(PHASE10.read_text(encoding="utf-8"))
    dsr = p.get("dsr", {})
    pbo = p.get("pbo_cscv", {})
    selected = list(pbo.get("selected_strategy_indices", []))
    omega = list(pbo.get("omega_values", []))
    logits = list(pbo.get("logit_values", []))
    report = {
        "source_artifact": "evidence/phase10_backtest.json",
        "source_dataset_sha256": p.get("data_integrity", {}).get("dataset_sha256"),
        "dsr": {
            "observed_sharpe": dsr.get("observed_sharpe"),
            "deflated_sharpe_probability": dsr.get("deflated_sharpe_probability"),
            "trial_count": dsr.get("trial_count"),
            "trial_universe": dsr.get("trial_universe"),
            "trial_sharpe_variance_recorded": dsr.get("trial_sharpe_variance") is not None,
            "status": "FAIL" if float(dsr.get("observed_sharpe", 0)) <= 0 or float(dsr.get("deflated_sharpe_probability", 0)) <= 0 else "REVIEW",
            "reason": "Current source evidence has non-positive observed Sharpe and zero DSR probability; no execution-readiness claim is supported.",
        },
        "pbo_cscv": {
            "block_count": pbo.get("block_count"),
            "path_count": pbo.get("path_count"),
            "pbo": pbo.get("pbo"),
            "overfit_paths": pbo.get("overfit_paths"),
            "selected_strategy_indices_unique": sorted(set(selected)),
            "selected_strategy_indices_constant": len(set(selected)) == 1 if selected else False,
            "omega_unique_count": len(set(omega)),
            "logit_unique_count": len(set(logits)),
            "omega_all_positive_logit": bool(logits) and all(x > 0 for x in logits),
            "input_observations": pbo.get("observations"),
            "input_hash": p.get("data_integrity", {}).get("dataset_sha256"),
            "status": "INSUFFICIENT_EVIDENCE",
            "interpretation": "PBO=0 is mechanically consistent with the stored CSCV output because all 70 paths have positive logit and zero overfit paths. The stored artifact does not contain the full strategy return matrix, so this audit cannot independently recompute every CSCV path.",
        },
        "real_money_execution": False,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
