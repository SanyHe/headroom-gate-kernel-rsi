#!/usr/bin/env bash
# Start a detached job that survives terminal close / SSH disconnect.
#   bash scripts/detach.sh <job_name> <command...>
# Prints the run directory. Success is decided by status.json, never by the terminal.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/detach.sh <job_name> <command...>" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_NAME=$1
shift

RUNS_DIR="$REPO_ROOT/evidence/runs"
mkdir -p "$RUNS_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUFFIX=$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')
RUN_ID="${STAMP}_${JOB_NAME//[^A-Za-z0-9_.-]/_}_${SUFFIX}"
RUN_DIR="$RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/artifacts"

PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=$(command -v python3)

"$PY" "$REPO_ROOT/scripts/lib/runmeta.py" init "$RUN_DIR" "$@"
"$PY" "$REPO_ROOT/scripts/lib/runmeta.py" status "$RUN_DIR" QUEUED "message=created"

# setsid: new session + no controlling terminal. stdin closed so the child can
# never block on the terminal. SIGHUP is ignored by the supervisor.
# set -m: without job control bash starts background jobs with SIGINT ignored,
# which would make the supervisor unable to trap a graceful stop.
# macOS has no setsid; there the job is detached with nohup only, which is
# enough for local smoke tests but is not a replacement for the Linux contract.
set -m
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash "$REPO_ROOT/scripts/job_supervisor.sh" "$RUN_DIR" "$@" \
    </dev/null >>"$RUN_DIR/launcher.log" 2>&1 &
else
  nohup bash "$REPO_ROOT/scripts/job_supervisor.sh" "$RUN_DIR" "$@" \
    </dev/null >>"$RUN_DIR/launcher.log" 2>&1 &
fi
SUP_PID=$!
disown "$SUP_PID" 2>/dev/null || true

# A detached job must prove it is alive before we return the run dir.
for _ in $(seq 1 30); do
  [[ -s "$RUN_DIR/supervisor.pid" ]] && break
  sleep 0.1
done
if [[ ! -s "$RUN_DIR/supervisor.pid" ]] || ! kill -0 "$(cat "$RUN_DIR/supervisor.pid")" 2>/dev/null; then
  "$PY" "$REPO_ROOT/scripts/lib/runmeta.py" status "$RUN_DIR" FAILED \
    "exit_code=1" "message=supervisor did not start" >/dev/null 2>&1 || true
  echo "ERROR: supervisor failed to start; see $RUN_DIR/launcher.log" >&2
  echo "$RUN_DIR"
  exit 1
fi

echo "$RUN_DIR"
