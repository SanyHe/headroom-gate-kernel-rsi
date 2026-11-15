import json
from copy import deepcopy
from pathlib import Path

import yaml

from gate import decide, rules as rules_mod
from gate.cli import load_case

REPO = Path(__file__).resolve().parents[1]
RTX_CASE = REPO / "cases-rtx" / "swiglu-fusion-b4096-h7168-rtx6000d"

# The only measured case published here is the AutoDL RTX 6000D swiglu fusion.
# The five cases in cases/ are pre-registered templates with no measurement data
# and produce no verdict.
EXPECT = {
    "swiglu-fusion-b4096-h7168-rtx6000d": ("GO", 2.147437, 2.154469, 2.151539),
}

# The published rules file is frozen; a silent edit must fail loudly.
PINNED_RULES_SHA = "eace6d1739a6a8d8e05950eda87593c8cd29e2c24ade0191d6f232e1e48436df"


def _run(case_path=RTX_CASE):
    gate_rules, sha = rules_mod.load_rules()
    return decide.evaluate(load_case(case_path), gate_rules,
                           rules_sha256=sha, rules_version=gate_rules["version"])


def _case(case_path=RTX_CASE):
    return yaml.safe_load(case_path.joinpath("case.yaml").read_text(encoding="utf-8"))


def _measured_template(**overrides):
    """A minimal valid measured case dict built from the RTX case, for mutation tests."""
    case = _case()
    case.update(overrides)
    return case


def test_measured_case_replays():
    for case_id, (verdict, lo, hi, point) in EXPECT.items():
        doc = _run()
        assert doc["verdict"] == verdict, (case_id, doc["reasons"])
        p = doc["prediction"]
        assert abs(p["speedup_upper_min"] - lo) < 1e-4, (case_id, p["speedup_upper_min"])
        assert abs(p["speedup_upper_max"] - hi) < 1e-4, (case_id, p["speedup_upper_max"])
        assert abs(p["speedup_upper_point"] - point) < 1e-4, (case_id, p["speedup_upper_point"])


def test_rules_file_matches_pinned_hash():
    _, sha = rules_mod.load_rules()
    assert sha == PINNED_RULES_SHA


def test_measured_case_pins_the_current_rules_hash():
    _, sha = rules_mod.load_rules()
    case = _case()
    assert case["rules_sha256"] == sha


def test_template_cases_have_no_measurements():
    """Pre-registered templates in cases/ carry no measured_ms and produce no verdict."""
    for path in sorted((REPO / "cases").glob("*/case.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "measured_ms" not in case.get("workload", {}), path
        assert case.get("status") == "pre_registered", path


def test_swiglu_upper_bound_is_consistent_with_band_and_cap():
    """The predicted upper bound is the max of speedup_upper over the band
    endpoints (and the cap kink, if it falls inside the band)."""
    doc = _run()
    floor = doc["floor"]["t_floor_ms"]
    cap = doc["addressable"]["cap_ms"]
    kink = floor + cap
    band = doc["measurement"]
    lo, hi = band["lo_ms"], band["hi_ms"]

    # Replicate the gate's candidate set: band endpoints plus kink if in-band.
    share = 1.0
    def speedup_at(t):
        hr = max(0.0, t - floor)
        applied = min(share * hr, cap, hr)
        return t / (t - applied) if t > applied else float("inf")

    candidates = [speedup_at(lo), speedup_at(band.get("med_ms", (lo + hi) / 2)),
                  speedup_at(hi)]
    if lo < kink < hi:
        candidates.append(speedup_at(kink))
    expected_max = max(candidates)
    assert abs(doc["prediction"]["speedup_upper_max"] - expected_max) < 1e-6


def test_out_of_scope_flag_invalidates():
    bad = _measured_template()
    bad["out_of_scope_checks"]["changes_precision"] = True
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("out of scope: changes_precision" in r for r in doc["reasons"])


def test_below_floor_is_invalid_not_clipped():
    bad = _measured_template()
    floor = bad["workload"].get("empirical_floor_ms", 0.109)
    bad["workload"]["measured_ms"] = [floor * 0.5]  # below the floor
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert "violation" in doc["reasons"][0]


def test_rules_hash_mismatch_invalidates():
    bad = _measured_template()
    bad["rules_sha256"] = "0" * 64
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("rules_sha256 mismatch" in r for r in doc["reasons"])


def test_disallowed_mechanism_class_invalidates():
    bad = _measured_template()
    bad["mechanism"]["class"] = "precision"
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("not allowed" in r for r in doc["reasons"])


def test_missing_required_fields_are_reported():
    bad = _measured_template()
    del bad["workload"]["measured_ms"]
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("workload.measured_ms" in r for r in doc["reasons"])


def test_go_requires_both_bounds_above_line():
    doc = _run()
    assert doc["checks"]["n_ge_min_repeats_go"] is True
    assert doc["checks"]["empirical_floor_present"] is True
    assert doc["checks"]["out_of_scope_clean"] is True
    assert doc["prediction"]["speedup_upper_min"] >= 1.30


def test_verdict_json_is_serialisable():
    doc = _run()
    json.dumps(doc)
    assert doc["kind"] == "hgk_gate_verdict"
    assert doc["rules_version"] == "2.0.0"
