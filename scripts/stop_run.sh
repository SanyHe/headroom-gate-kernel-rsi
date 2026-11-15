#!/usr/bin/env bash
# Stop a run gracefully: SIGINT, wait 30 s, then SIGTERM, wait 10 s. Never SIGKILL.
# A stuck run is reported, not destroyed.
#   bash scripts/stop_run.sh <run_dir>
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/stop_run.sh <run_dir>" >&2
  exit 2
fi

RUN_DIR=$1
PID_FILE="$RUN_DIR/supervisor.pid"
[[ -s "$PID_FILE" ]] || { echo "no supervisor.pid in $RUN_DIR" >&2; exit 2; }
PID=$(cat "$PID_FILE")
kill -0 "$PID" 2>/dev/null || { echo "supervisor $PID is not running"; exit 0; }

echo "SIGINT -> $PID"
kill -INT "$PID" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || { echo "stopped cleanly"; exit 0; }
  sleep 1
done

echo "still alive after 30 s; SIGTERM -> $PID"
kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 10); do
  kill -0 "$PID" 2>/dev/null || { echo "stopped after SIGTERM"; exit 0; }
  sleep 1
done

echo "supervisor $PID did not exit; investigate (status.json + stderr.log); not killing" >&2
exit 1
