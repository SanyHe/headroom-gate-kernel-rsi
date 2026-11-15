#!/usr/bin/env bash
# Local bootstrap + verification. No GPU required.
#   ./init.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY=python3
if [[ ! -x .venv/bin/python ]]; then
  echo "[init] creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[init] installing dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "[init] replaying the archived gate verdicts"
python -m gate.cli --all

echo "[init] running the test suite"
python -m pytest tests -q

echo
echo "[init] OK — gate replays and tests pass. No GPU was used."
