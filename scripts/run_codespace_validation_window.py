"""Collect an additional uninterrupted real Binance Futures paper-evidence window.

Existing evidence is preserved. This runner appends 12 new observations after
the current state and never resets the ledger or paper state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

WINDOW = 12
WAIT_SECONDS = 300
BRANCH = "agent/phase13-audit"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(*args: str, check: bool = True) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ROOT + (os.pathsep + existing if existing else "")
    subprocess.run(args, check=check, env=env)


def main() -> int:
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != BRANCH:
        print(f"STOP: expected branch {BRANCH!r}, got {branch!r}")
        return 2

    for cycle in range(1, WINDOW + 1):
        print(f"=== REAL VALIDATION WINDOW CYCLE {cycle}/{WINDOW} ===", flush=True)
        run(sys.executable, "scripts/collect_paper_evidence.py")
        run("git", "config", "user.name", "codespace-paper-validation")
        run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        run("git", "add", "evidence/paper_state.json", "evidence/paper_trading.jsonl")
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode
        if staged != 0:
            run("git", "commit", "-m", f"Record validation-window paper evidence cycle {cycle}")
            run("git", "push", "origin", f"HEAD:{BRANCH}")
        if cycle < WINDOW:
            print("Waiting 5 minutes for the next closed-market observation...", flush=True)
            time.sleep(WAIT_SECONDS)

    print("=== REAL VALIDATION WINDOW COLLECTION COMPLETE: 12/12 ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
