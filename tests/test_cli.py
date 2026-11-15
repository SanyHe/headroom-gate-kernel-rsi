import json
from pathlib import Path

import pytest

from gate import cli, ledger

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "cases-rtx" / "swiglu-fusion-b4096-h7168-rtx6000d"
CASE_ID = "swiglu-fusion-b4096-h7168-rtx6000d"


@pytest.fixture()
def isolated_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    return path


def test_single_case_writes_verdict_file(isolated_ledger, capsys):
    rc = cli.main([str(CASE)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{CASE_ID}: GO" in out
    doc = json.loads((CASE / "gate_verdict.json").read_text())
    assert doc["kind"] == "hgk_gate_verdict"
    assert doc["prediction"]["speedup_upper_max"] > 2.0
    assert doc["case_sha256"]


def test_all_cases_record_and_verify_ledger(isolated_ledger, capsys):
    assert cli.main(["--all", "--record"]) == 0
    entries = ledger.read_all(isolated_ledger)
    assert len(entries) == 1
    assert entries[0]["case_id"] == CASE_ID
    assert entries[0]["verdict"] == "GO"
    assert cli.main(["--verify-ledger"]) == 0
    assert "P1 ratchet: no violations" in capsys.readouterr().out


def test_verify_ledger_flags_a_falsified_ratchet(isolated_ledger, capsys):
    assert cli.main(["--all", "--record"]) == 0
    ledger.append({"event": "realised", "case_id": CASE_ID,
                   "realised_speedup": 9.0}, path=isolated_ledger)
    assert cli.main(["--verify-ledger"]) == 4
    assert "P1 VIOLATIONS" in capsys.readouterr().out


def test_json_flag_prints_the_full_verdict(isolated_ledger, capsys):
    assert cli.main([str(CASE), "--json"]) == 0
    payload = capsys.readouterr().out
    start = payload.index("{")
    doc = json.loads(payload[start:])
    assert "decision_trace" in doc
    assert doc["checks"]["empirical_floor_present"] is True


def test_invalid_case_exits_3(isolated_ledger, tmp_path, capsys):
    case_dir = tmp_path / "bad-case"
    case_dir.mkdir()
    (case_dir / "case.yaml").write_text(
        "id: bad\nhypothesis: x\n"
        "workload: {kind: gemm, shapes: {}, dtype: bf16, bytes: {weights: 1}, "
        "measured_ms: [1.0], measurement_source: none}\n"
        "mechanism: {name: n, class: tiling, removable_bytes: 0, "
        "addressable_share_claim: 1.0, justification: j}\n"
        "out_of_scope_checks: {changes_precision: true}\n", encoding="utf-8")
    assert cli.main([str(case_dir)]) == 3
    assert json.loads((case_dir / "gate_verdict.json").read_text())["verdict"] == "INVALID"


def test_missing_case_yaml_is_rejected(isolated_ledger, tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path)])


def test_no_case_and_no_all_is_a_usage_error(isolated_ledger):
    with pytest.raises(SystemExit):
        cli.main([])
