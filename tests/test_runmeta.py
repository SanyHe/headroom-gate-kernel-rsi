import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lib"))
import runmeta  # noqa: E402


def test_init_writes_the_full_record(tmp_path):
    run = tmp_path / "run1"
    runmeta.cmd_init(run, ["python3", "-m", "gate.cli", "--all"])
    assert (run / "artifacts").is_dir()
    assert (run / "command.txt").read_text().endswith("--all\n")
    assert json.loads((run / "command.json").read_text())["argv"][-1] == "--all"
    env = json.loads((run / "environment.json").read_text())
    for key in ("run_id", "started_at", "hostname", "python", "cwd", "git_commit"):
        assert key in env
    manifest = json.loads((run / "input_manifest.json").read_text())
    assert manifest["file_count"] == len(manifest["files"]) > 0
    assert any(f["path"] == "gate/decide.py" for f in manifest["files"])


def test_status_transitions_set_and_preserve_fields(tmp_path):
    run = tmp_path / "run2"
    runmeta.cmd_init(run, ["true"])
    runmeta.cmd_status(run, "QUEUED", {"message": "created"})
    runmeta.cmd_status(run, "RUNNING", {"supervisor_pid": "10", "resume_count": "0"})
    running = json.loads((run / "status.json").read_text())
    assert running["state"] == "RUNNING"
    assert running["started_at"] and running["supervisor_pid"] == 10

    runmeta.cmd_status(run, "FAILED", {"exit_code": "3", "signal": "none"})
    done = json.loads((run / "status.json").read_text())
    assert done["state"] == "FAILED"
    assert done["exit_code"] == 3
    assert done["finished_at"] and done["duration_s"] >= 0
    assert done["supervisor_pid"] == 10          # history survives the transition
    assert done["message"] == "created"


def test_heartbeat_carries_checkpoint_progress(tmp_path):
    run = tmp_path / "run3"
    runmeta.cmd_init(run, ["true"])
    runmeta.cmd_checkpoint(run, "measure", 42)
    runmeta.cmd_heartbeat(run, 999)
    beat = json.loads((run / "heartbeat.json").read_text())
    assert beat["supervisor_pid"] == 999
    assert beat["beats"] == 1
    assert beat["progress"]["phase"] == "measure"
    assert beat["progress"]["percent"] == 42
    runmeta.cmd_heartbeat(run, 999)
    assert json.loads((run / "heartbeat.json").read_text())["beats"] == 2


def test_checkpoint_clamps_percent(tmp_path):
    run = tmp_path / "run4"
    runmeta.cmd_checkpoint(run, "warmup", 140)
    assert json.loads((run / "checkpoint.json").read_text())["percent"] == 100
    runmeta.cmd_checkpoint(run, "warmup", -5)
    assert json.loads((run / "checkpoint.json").read_text())["percent"] == 0


def test_atomic_json_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "x.json"
    runmeta.atomic_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_read_json_tolerates_missing_and_corrupt(tmp_path):
    assert runmeta.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert runmeta.read_json(bad) == {}


def test_main_rejects_bad_usage(tmp_path):
    assert runmeta.main(["runmeta.py"]) == 2
    assert runmeta.main(["runmeta.py", "init", str(tmp_path / "r")]) == 2
    assert runmeta.main(["runmeta.py", "nope", str(tmp_path)]) == 2
