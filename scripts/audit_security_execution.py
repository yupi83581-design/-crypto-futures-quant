"""Static security/execution-isolation audit for the research runtime.

This audit is conservative: any private-key material or exchange-order API
surface found in the paper/runtime source causes FAIL. Absence of findings is
reported as PASS for static isolation only; it does not prove runtime behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src", ROOT / "scripts")
OUT = ROOT / "evidence" / "security_execution_audit.json"

PRIVATE_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"api[_-]?secret", re.IGNORECASE),
    re.compile(r"secret[_-]?key", re.IGNORECASE),
)

ORDER_PATTERNS = (
    re.compile(r"futures?_create_order", re.IGNORECASE),
    re.compile(r"create_order\s*\(", re.IGNORECASE),
    re.compile(r"new_order\s*\(", re.IGNORECASE),
    re.compile(r"/order(?:\?|\s|$)", re.IGNORECASE),
    re.compile(r"ORDER\s*\(?!BOOK)", re.IGNORECASE),
)


def main() -> int:
    findings = []
    scanned_files = 0

    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative = str(path.relative_to(ROOT))
            if relative == "scripts/audit_security_execution.py" or relative.startswith("tests/"):
                continue
            scanned_files += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    findings.append({"type": "private_credential_pattern", "file": relative, "pattern": pattern.pattern})
            for pattern in ORDER_PATTERNS:
                if pattern.search(text):
                    findings.append({"type": "exchange_order_pattern", "file": relative, "pattern": pattern.pattern})

    report = {
        "status": "PASS" if not findings else "FAIL",
        "scope": "static execution-isolation audit",
        "scanned_files": scanned_files,
        "findings": findings,
        "real_money_execution": False,
        "limitation": "Static PASS does not prove runtime behavior; behavioral tests and runtime audit remain separate gates.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
