"""Run the real Binance Futures paper-evidence collector from a Codespace.

This is intentionally separate from GitHub-hosted Actions because Binance can
return HTTP 451 to hosted-runner egress. It uses the existing collector without
changing its market-data semantics. No exchange orders are placed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

CYCLES = 12
WAIT_SECONDS = 300
BRANCH = "agent/phase13-audit"


def run(*args: str, check: bool = True) -> None:
    subprocess.run(args, check=check)


def main() -> int:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()
    if branch != BRANCH:
        print(f"STOP: expected branch {BRANCH!r}, got {branch!r}")
        return 2

    run("git", "config", "user.name", "codespace-paper-evidence")
    run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
    )

    for cycle in range(1, CYCLES + 1):
        print(f"=== REAL CODESPACE PAPER CYCLE {cycle}/{CYCLES} ===", flush=True)

        run(sys.executable, "scripts/collect_paper_evidence.py")

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
                f"Record real paper-trading evidence cycle {cycle}",
            )
            run("git", "push", "origin", f"HEAD:{BRANCH}")
        else:
            print("No evidence-file changes to commit.", flush=True)

        if cycle < CYCLES:
            print("Waiting 5 minutes for the next closed-market observation...", flush=True)
            time.sleep(WAIT_SECONDS)

    print("=== REAL CODESPACE PAPER EVIDENCE COMPLETE: 12/12 ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
