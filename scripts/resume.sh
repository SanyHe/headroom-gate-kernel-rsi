#!/usr/bin/env bash
# Resume a finished/failed run from its recorded argv. Never touch a live run.
#   bash scripts/resume.sh <run_dir>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/resume.sh <run_dir>" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR=$(cd "$1" && pwd)
PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=$(command -v python3)

# Refuses RUNNING/QUEUED: resuming a live run would double-book the GPU.
read -r STATE COUNT < <("$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
state = status.get("state", "UNKNOWN")
if state in ("RUNNING", "QUEUED"):
    raise SystemExit(f"refusing to resume: state is {state}; inspect status.json first")
print(state, int(status.get("resume_count", 0)) + 1)
PY
)
echo "previous state: $STATE · resume_count -> $COUNT"

CMD_JSON=$("$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
argv = json.loads((Path(sys.argv[1]) / "command.json").read_text())["argv"]
print(" ".join(json.dumps(a) for a in argv))
PY
)

cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/artifacts"
set -m   # keep SIGINT trappable in the backgrounded supervisor
if command -v setsid >/dev/null 2>&1; then
  eval "RESUME_COUNT=$COUNT setsid nohup bash scripts/job_supervisor.sh \"$RUN_DIR\" $CMD_JSON </dev/null >>\"$RUN_DIR/launcher.log\" 2>&1 &"
else
  eval "RESUME_COUNT=$COUNT nohup bash scripts/job_supervisor.sh \"$RUN_DIR\" $CMD_JSON </dev/null >>\"$RUN_DIR/launcher.log\" 2>&1 &"
fi
SUP_PID=$!
disown "$SUP_PID" 2>/dev/null || true

for _ in $(seq 1 30); do
  [[ -s "$RUN_DIR/supervisor.pid" ]] && break
  sleep 0.1
done
echo "resumed $RUN_DIR (resume_count=$COUNT)"
