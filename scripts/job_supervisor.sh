#!/usr/bin/env bash
# Supervisor for a detached run. Started by scripts/detach.sh via setsid.
# Contract: QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELLED
# SIGINT/SIGTERM are forwarded to the child as SIGINT so training/engine code can
# shut down gracefully. We never kill -9: a stuck run is reported, not destroyed.
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: job_supervisor.sh <run_dir> <command...>" >&2
  exit 2
fi

RUN_DIR=$1
shift
export RUN_DIR
export RESUME_COUNT=${RESUME_COUNT:-0}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT" >&2; exit 1; }
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=$(command -v python3)

mkdir -p "$RUN_DIR/artifacts"
echo $$ >"$RUN_DIR/supervisor.pid"
trap '' HUP

# `sid` is Linux; BSD/macOS ps only knows `sess` (an opaque address) and prints
# its keyword list on stdout when asked for an unknown field, so validate.
SID=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')
[[ "${SID:-}" =~ ^[0-9]+$ ]] || SID=$(ps -o sess= -p $$ 2>/dev/null | tr -d ' ')
[[ "${SID:-}" =~ ^[0-9]+$ ]] || SID=0
SUP_PPID=$(ps -o ppid= -p $$ | tr -d ' ')
SIGNAL=""
CHILD=""
RC=0

set_status() {
  "$PY" "$REPO_ROOT/scripts/lib/runmeta.py" status "$RUN_DIR" "$@" >/dev/null 2>&1 || true
}

forward_int() {
  SIGNAL=${SIGNAL:-INT}
  if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -INT "$CHILD" 2>/dev/null || true
  fi
}
# Stage two: the child already had 30 s of SIGINT. Escalate, then let `wait` reap.
forward_term() {
  SIGNAL=TERM
  if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD" 2>/dev/null || true
  fi
}
trap forward_int INT
trap forward_term TERM

set_status RUNNING "supervisor_pid=$$" "supervisor_sid=$SID" \
  "supervisor_ppid=$SUP_PPID" "resume_count=${RESUME_COUNT}"

# Heartbeat is independent of the child so a hung child still shows a live
# supervisor and a stale phase.
(
  while sleep 5; do
    "$PY" "$REPO_ROOT/scripts/lib/runmeta.py" heartbeat "$RUN_DIR" "$$" >/dev/null 2>&1 || true
  done
) &
HEARTBEAT=$!

START_EPOCH=$(date +%s)
# Job control: a plain `cmd &` would start the child with SIGINT ignored, so a
# graceful stop could never reach it.
set -m
"$@" >>"$RUN_DIR/stdout.log" 2>>"$RUN_DIR/stderr.log" &
CHILD=$!

while true; do
  wait "$CHILD"
  RC=$?
  kill -0 "$CHILD" 2>/dev/null || break
done

kill "$HEARTBEAT" 2>/dev/null || true
wait "$HEARTBEAT" 2>/dev/null || true

END_EPOCH=$(date +%s)
FINAL_PPID=$(ps -o ppid= -p $$ | tr -d ' ')
echo "$RC" >"$RUN_DIR/exit_code"

if [[ -n "$SIGNAL" ]]; then
  STATE=CANCELLED
  [[ "$RC" -eq 0 ]] && RC=130
elif [[ "$RC" -eq 0 ]]; then
  STATE=SUCCEEDED
else
  STATE=FAILED
fi

set_status "$STATE" "exit_code=$RC" "signal=${SIGNAL:-none}" \
  "duration_s=$((END_EPOCH - START_EPOCH))" "supervisor_ppid_final=${FINAL_PPID:-0}" \
  "message=finished with exit code $RC"

# One last heartbeat snapshot so check_run.sh sees the final progress.
"$PY" "$REPO_ROOT/scripts/lib/runmeta.py" heartbeat "$RUN_DIR" "$$" >/dev/null 2>&1 || true
exit "$RC"
