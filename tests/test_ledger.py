import json

import pytest

from gate import ledger


def _entries(path):
    return ledger.read_all(path)


def test_append_is_append_only_and_readable(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "gate", "case_id": "a"}, path=path)
    ledger.append({"event": "realised", "case_id": "a", "realised_speedup": 1.5}, path=path)
    entries = _entries(path)
    assert len(entries) == 2
    assert entries[0]["event"] == "gate"
    assert all("ts" in e for e in entries)


def test_read_all_missing_file_is_empty(tmp_path):
    assert ledger.read_all(tmp_path / "nope.jsonl") == []


def test_corrupt_line_raises_with_lineno(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"event": "gate"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        ledger.read_all(path)


def test_state_by_case_rebuilds_from_events(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "gate", "case_id": "a", "verdict": "GO",
                   "prediction": {"speedup_upper_max": 5.0}}, path=path)
    ledger.append({"event": "measured", "case_id": "a"}, path=path)
    ledger.append({"event": "realised", "case_id": "a", "realised_speedup": 4.2}, path=path)
    state = ledger.state_by_case(_entries(path))
    assert state["a"]["verdict"] == "GO"
    assert state["a"]["predicted_upper_max"] == 5.0
    assert state["a"]["realised_speedup"] == 4.2
    assert state["a"]["last_event"] == "realised"


def test_ratchet_passes_when_realised_is_below_prediction(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "gate", "case_id": "a",
                   "prediction": {"speedup_upper_max": 5.004346}}, path=path)
    ledger.append({"event": "realised", "case_id": "a", "realised_speedup": 4.196}, path=path)
    assert ledger.verify_ratchet(_entries(path)) == []


def test_ratchet_catches_p1_falsification(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "gate", "case_id": "a",
                   "prediction": {"speedup_upper_max": 1.05}}, path=path)
    ledger.append({"event": "realised", "case_id": "a", "realised_speedup": 1.5}, path=path)
    violations = ledger.verify_ratchet(_entries(path))
    assert len(violations) == 1
    assert "P1 falsified" in violations[0]["reason"]


def test_ratchet_catches_missing_prediction_and_missing_value(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "realised", "case_id": "no-gate", "realised_speedup": 1.2},
                  path=path)
    ledger.append({"event": "gate", "case_id": "b",
                   "prediction": {"speedup_upper_max": 2.0}}, path=path)
    ledger.append({"event": "realised", "case_id": "b"}, path=path)
    reasons = sorted(v["reason"] for v in ledger.verify_ratchet(_entries(path)))
    assert reasons == ["realised without a prior gate prediction",
                       "realised without realised_speedup"]


def test_ledger_line_is_sorted_json_with_ts(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append({"event": "stop", "case_id": "a"}, path=path)
    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert list(payload) == sorted(payload)
    assert payload["event"] == "stop"
