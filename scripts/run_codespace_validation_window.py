"""Collect a fresh, uninterrupted real Binance Futures paper-evidence window.

Existing evidence is preserved. The runner appends real observations until the
latest 12 newly collected events form a continuous 5-minute window. Failed
attempts remain as evidence; nothing is deleted or rewritten.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WINDOW = 12
WAIT_SECONDS = 300
BRANCH = "agent/phase13-audit"
ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "evidence" / "paper_trading.jsonl"
MAX_ATTEMPTS = 72


def run(*args: str, check: bool = True) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ROOT.as_posix() + (os.pathsep + existing if existing else "")
    subprocess.run(args, check=check, env=env)


def ledger_count() -> int:
    if not LEDGER.exists():
        return 0
    return sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())


def audit_latest_window() -> bool:
    result = subprocess.run(
        [sys.executable, "scripts/audit_paper_evidence.py"],
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT.as_posix()},
    )
    return result.returncode == 0


def main() -> int:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()
    if branch != BRANCH:
        print(f"STOP: expected branch {BRANCH!r}, got {branch!r}")
        return 2

    baseline_events = ledger_count()
    attempts = 0

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        print(
            f"=== REAL VALIDATION WINDOW ATTEMPT {attempts}/{MAX_ATTEMPTS} "
            f"(target={WINDOW} consecutive 5m events) ===",
            flush=True,
        )

        try:
            run(sys.executable, "scripts/collect_paper_evidence.py")
        except subprocess.CalledProcessError:
            print("Collection attempt failed; preserving prior evidence and retrying.", flush=True)
            if attempts < MAX_ATTEMPTS:
                time.sleep(WAIT_SECONDS)
            continue

        run("git", "config", "user.name", "codespace-paper-validation")
        run(
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        )
        run("git", "add", "evidence/paper_state.json", "evidence/paper_trading.jsonl")
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            check=False,
        ).returncode
        if staged != 0:
            run(
                "git",
                "commit",
                "-m",
                f"Record validation-window paper evidence attempt {attempts}",
            )
            run("git", "push", "origin", f"HEAD:{BRANCH}")

        new_events = ledger_count() - baseline_events
        if new_events >= WINDOW and audit_latest_window():
            print("=== REAL VALIDATION WINDOW PASS: 12 consecutive 5m events ===", flush=True)
            return 0

        if attempts < MAX_ATTEMPTS:
            print("Continuity gate not yet PASS; waiting 300 seconds.", flush=True)
            time.sleep(WAIT_SECONDS)

    print("=== REAL VALIDATION WINDOW BLOCKED: 12 consecutive 5m events not proven ===", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
