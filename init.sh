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
# Pry the index away from a stale global pip config: a dead or mirror-only
# index-url makes the bootstrap fail even when PyPI is reachable. The official
# index gets a short deadline; if it cannot answer, fall back to a public
# mirror. Set PIP_INDEX_URL to pin a specific index and skip the fallback.
install_deps() {
  python -m pip install --quiet --index-url "$1" --timeout "$2" --retries 2 -r requirements.txt
}
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  install_deps "$PIP_INDEX_URL" 120
elif ! install_deps "https://pypi.org/simple" 20; then
  echo "[init] pypi.org is too slow here; retrying via a public mirror" >&2
  install_deps "https://pypi.tuna.tsinghua.edu.cn/simple" 120
fi

echo "[init] replaying the archived gate verdicts"
python -m gate.cli --all

echo "[init] running the test suite"
python -m pytest tests -q

echo
echo "[init] OK — gate replays and tests pass. No GPU was used."
