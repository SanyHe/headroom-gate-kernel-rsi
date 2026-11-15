#!/usr/bin/env bash
# Report one run: status.json, heartbeat age, last lines of stdout/stderr.
# Exit codes: 0 = SUCCEEDED, 1 = FAILED/CANCELLED, 2 = RUNNING/QUEUED.
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/check_run.sh <run_dir>" >&2
  exit 2
fi

RUN_DIR=$1
[[ -d "$RUN_DIR" ]] || { echo "no such run dir: $RUN_DIR" >&2; exit 2; }

PY=$(command -v python3)
STATE=$("$PY" - "$RUN_DIR" <<'PY'
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
print(json.dumps({k: status.get(k) for k in
                  ("run_id", "state", "started_at", "finished_at", "exit_code",
                   "signal", "duration_s", "resume_count", "message")}, indent=2))
hb = run / "heartbeat.json"
if hb.exists():
    h = json.loads(hb.read_text())
    age = time.time() - float(h.get("epoch", 0))
    print(f"heartbeat: {age:.1f}s ago · beats={h.get('beats')} · progress={h.get('progress')}")
    if age > 30:
        print("heartbeat is stale (>30s)")
else:
    print("heartbeat: missing")
print("state:", status.get("state", "UNKNOWN"))
PY
)
echo "$STATE"

STATE_ONLY=$(echo "$STATE" | awk '/^state: /{print $2}')
case "${STATE_ONLY:-UNKNOWN}" in
  SUCCEEDED) RC=0 ;;
  FAILED|CANCELLED) RC=1 ;;
  *) RC=2 ;;
esac

echo
echo "--- stdout.log (tail 5) ---"
tail -n 5 "$RUN_DIR/stdout.log" 2>/dev/null || true
echo "--- stderr.log (tail 5) ---"
tail -n 5 "$RUN_DIR/stderr.log" 2>/dev/null || true

exit "$RC"
