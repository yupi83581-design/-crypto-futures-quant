#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Minimal Android/Termux setup for the existing read-only quant runtime.
# This script does not change runtime architecture and does not install
# exchange SDKs, API keys, or trading/execution components.

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "WARNING: expected Android ARM64 (aarch64); detected $(uname -m)."
fi

echo "[1/4] Updating Termux packages..."
pkg update -y

echo "[2/4] Installing base tools..."
pkg install -y python python-pip git curl

# jsonschema is the repository's only declared third-party dependency.
# Current jsonschema releases depend on rpds-py, a Rust-backed native module.
# Prefer Termux's native package so pip does not try to compile Rust code on
# the phone. If the installed Termux repository does not provide it, stop
# explicitly instead of silently changing the dependency or architecture.
if apt-cache show python-rpds-py >/dev/null 2>&1; then
  echo "[3/4] Installing Termux native rpds-py dependency..."
  pkg install -y python-rpds-py
else
  echo "ERROR: this Termux repository does not provide python-rpds-py."
  echo "Refresh/switch the official Termux repository, then rerun this script."
  echo "No Rust toolchain or alternate dependency is installed automatically."
  exit 1
fi

echo "[4/4] Installing repository Python dependencies..."
python -m pip install -r requirements.txt

python - <<'PY'
import jsonschema
import rpds

print("Termux dependency check: OK")
print(f"Python: {__import__('sys').version.split()[0]}")
print(f"jsonschema: {jsonschema.__version__}")
print(f"rpds: {getattr(rpds, '__version__', 'installed')}")
PY

echo
echo "Setup complete. Runtime defaults are 0.0.0.0:8080."
echo "Start with: python -m src.production.runtime"
echo "Verify with: curl -sS http://127.0.0.1:8080/snapshot"
