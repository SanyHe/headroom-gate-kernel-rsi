#!/usr/bin/env python3
"""Run metadata helpers for detached jobs.

  init RUN_DIR CMD...          write command.txt/json, environment.json, input_manifest.json
  status RUN_DIR STATE [k=v]   merge status.json
  heartbeat RUN_DIR [PID]      update heartbeat.json, copy progress from checkpoint.json
  checkpoint RUN_DIR PHASE PCT write checkpoint.json

All JSON writes are atomic (tmp file + rename) so a crash never leaves half a file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path, default=None) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_init(run_dir: Path, cmd: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    atomic_json(run_dir / "command.json", {"argv": cmd, "cwd": os.getcwd()})
    atomic_json(run_dir / "environment.json", {
        "run_id": run_dir.name, "started_at": now(),
        "hostname": socket.gethostname(), "user": os.environ.get("USER", ""),
        "cwd": os.getcwd(), "repo_root": str(REPO), "python": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip() or None,
        "launcher": {"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(os.getpid())},
    })
    files = []
    for pat in ("cases", "gate", "stats", "protocol", "scripts"):
        for path in sorted((REPO / pat).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append({"path": str(path.relative_to(REPO)),
                              "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json(run_dir / "input_manifest.json",
                {"created_at": now(), "file_count": len(files), "files": files})


def cmd_status(run_dir: Path, state: str, opts: dict) -> int:
    path = run_dir / "status.json"
    status = read_json(path)
    status.setdefault("run_id", run_dir.name)
    status.setdefault("created_at", now())
    status["state"] = state
    status["updated_at"] = now()
    if state == "RUNNING" and "started_at" not in status:
        status["started_at"] = status["updated_at"]
    ints = ("exit_code", "supervisor_pid", "supervisor_sid",
            "supervisor_ppid", "supervisor_ppid_final", "resume_count")
    for key in ints + ("signal", "message"):
        if key in opts:
            status[key] = int(opts[key]) if key in ints else opts[key]
    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        status["finished_at"] = status["updated_at"]
        if status.get("started_at"):
            try:
                start = datetime.fromisoformat(status["started_at"])
                end = datetime.fromisoformat(status["finished_at"])
                status["duration_s"] = round((end - start).total_seconds(), 3)
            except ValueError:
                pass
    atomic_json(path, status)
    return 0


def cmd_heartbeat(run_dir: Path, pid: int | None) -> int:
    checkpoint = read_json(run_dir / "checkpoint.json")
    beats = read_json(run_dir / "heartbeat.json").get("beats", 0) + 1
    atomic_json(run_dir / "heartbeat.json", {
        "updated_at": now(), "epoch": round(time.time(), 3), "beats": beats,
        "supervisor_pid": pid or os.getpid(), "progress": checkpoint})
    return 0


def cmd_checkpoint(run_dir: Path, phase: str, percent: int) -> int:
    atomic_json(run_dir / "checkpoint.json", {
        "updated_at": now(), "phase": phase, "percent": max(0, min(100, percent))})
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, run_dir = argv[1], Path(argv[2])
    if cmd == "init":
        if len(argv) < 4:
            print("init needs a command", file=sys.stderr)
            return 2
        cmd_init(run_dir, argv[3:])
        return 0
    if cmd == "status":
        opts = {}
        for item in argv[4:]:
            if "=" in item:
                key, value = item.split("=", 1)
                opts[key.lstrip("-")] = value
        return cmd_status(run_dir, argv[3], opts)
    if cmd == "heartbeat":
        return cmd_heartbeat(run_dir, int(argv[3]) if len(argv) > 3 else None)
    if cmd == "checkpoint":
        return cmd_checkpoint(run_dir, argv[3], int(argv[4]))
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
