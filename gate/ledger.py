"""Append-only experiment ledger.

Events: pre_registered | gate | measured | realised | sealed_eval | stop |
measurement_protocol | e2e_result | e2e_note | dispatch_result |
dispatch_note | p3_selection | p3_result
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO / "ledger.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append(entry, path=None):
    """Append one JSON object atomically under an exclusive file lock."""
    path = Path(path or LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": now(), **entry}
    line = json.dumps(record, sort_keys=True, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return record


def read_all(path=None):
    path = Path(path or LEDGER_PATH)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: corrupt ledger line: {exc}") from None
    return entries


def state_by_case(entries):
    """Rebuild the latest known state per case from the ledger alone."""
    state = {}
    for entry in entries:
        case_id = entry.get("case_id")
        if not case_id:
            continue
        rec = state.setdefault(case_id, {})
        rec["last_event"] = entry.get("event")
        rec["last_ts"] = entry.get("ts")
        if entry.get("event") == "gate":
            rec["verdict"] = entry.get("verdict")
            rec["predicted_upper_max"] = (entry.get("prediction") or {}).get(
                "speedup_upper_max")
        if "realised_speedup" in entry:
            rec["realised_speedup"] = entry["realised_speedup"]
    return state


def verify_ratchet(entries):
    """Falsification check for claim P1: realised <= latest predicted upper."""
    latest_prediction, violations = {}, []
    for entry in entries:
        case_id = entry.get("case_id")
        if not case_id:
            continue
        if entry.get("event") == "gate":
            upper = (entry.get("prediction") or {}).get("speedup_upper_max")
            if upper is not None:
                latest_prediction[case_id] = float(upper)
        elif entry.get("event") == "realised":
            realised = entry.get("realised_speedup")
            predicted = latest_prediction.get(case_id)
            if realised is None:
                violations.append({"case_id": case_id,
                                   "reason": "realised without realised_speedup"})
            elif predicted is None:
                violations.append({"case_id": case_id,
                                   "reason": "realised without a prior gate prediction"})
            elif float(realised) > predicted + 1e-9:
                violations.append({
                    "case_id": case_id,
                    "reason": "P1 falsified: realised exceeds predicted upper bound",
                    "realised_speedup": float(realised),
                    "predicted_upper_max": predicted,
                })
    return violations
