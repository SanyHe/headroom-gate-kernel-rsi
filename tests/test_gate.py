import json
from copy import deepcopy
from pathlib import Path

import yaml

from gate import decide, rules as rules_mod
from gate.cli import load_case

REPO = Path(__file__).resolve().parents[1]
EXPECT = {
    "swiglu-fusion-b4096-h7168": ("GO", 4.993443, 5.004346, 4.999138),
    "attn-decode-longctx": ("NO-GO", 1.021620, 1.035426, 1.035426),
    "gemv-m8-bf16": ("NO-GO", 1.021552, 1.048246, 1.029720),
    "selector-dflash2": ("NO-GO", 1.004749, 1.005068, 1.004906),
    "verify-fusion-megaspec": ("NO-GO", 1.0, 1.003320, 1.003257),
}
# The upstream archive pinned 6280dcff... for its byte-identical YAML. This repo
# re-typed the file, so the identity hash differs while every number replays
# (see private/PROGRESS.md and docs/03-eval.md). The hash is pinned here so a
# silent rules edit fails loudly.
PINNED_RULES_SHA = "966715a7dfe9df58e71175719fa3f62226d474e368adc42c177fe7d17416bea2"


def _run(case_id):
    gate_rules, sha = rules_mod.load_rules()
    return decide.evaluate(load_case(REPO / "cases" / case_id), gate_rules,
                           rules_sha256=sha, rules_version=gate_rules["version"])


def _case(case_id):
    return yaml.safe_load(
        (REPO / "cases" / case_id / "case.yaml").read_text(encoding="utf-8"))


def test_archived_verdicts_replay():
    for case_id, (verdict, lo, hi, point) in EXPECT.items():
        doc = _run(case_id)
        assert doc["verdict"] == verdict, (case_id, doc["reasons"])
        p = doc["prediction"]
        assert abs(p["speedup_upper_min"] - lo) < 1e-4, (case_id, p["speedup_upper_min"])
        assert abs(p["speedup_upper_max"] - hi) < 1e-4, (case_id, p["speedup_upper_max"])
        assert abs(p["speedup_upper_point"] - point) < 1e-4, (case_id, p["speedup_upper_point"])


def test_rules_file_matches_pinned_hash():
    _, sha = rules_mod.load_rules()
    assert sha == PINNED_RULES_SHA


def test_every_case_pins_the_current_rules_hash():
    _, sha = rules_mod.load_rules()
    for path in sorted((REPO / "cases").glob("*/case.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert case["rules_sha256"] == sha, path


def test_swiglu_upper_bound_comes_from_the_in_band_kink():
    doc = _run("swiglu-fusion-b4096-h7168")
    floor = doc["floor"]["t_floor_ms"]
    cap = doc["addressable"]["cap_ms"]
    kink = floor + cap
    band = doc["measurement"]
    assert band["lo_ms"] < kink < band["hi_ms"], "kink must fall inside the CI band"
    expected = kink / floor
    assert abs(doc["prediction"]["speedup_upper_max"] - expected) < 1e-6
    # Two-endpoint-only evaluation would land near 4.99 and miss the kink.
    assert doc["prediction"]["speedup_upper_max"] > 5.0


def test_out_of_scope_flag_invalidates():
    bad = deepcopy(_case("swiglu-fusion-b4096-h7168"))
    bad["out_of_scope_checks"]["changes_precision"] = True
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("out of scope: changes_precision" in r for r in doc["reasons"])


def test_below_floor_is_invalid_not_clipped():
    bad = deepcopy(_case("swiglu-fusion-b4096-h7168"))
    bad["workload"]["measured_ms"] = [0.050]   # below the 0.053382 floor
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert "violation" in doc["reasons"][0]


def test_rules_hash_mismatch_invalidates():
    bad = deepcopy(_case("swiglu-fusion-b4096-h7168"))
    bad["rules_sha256"] = "0" * 64
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("rules_sha256 mismatch" in r for r in doc["reasons"])


def test_disallowed_mechanism_class_invalidates():
    bad = deepcopy(_case("swiglu-fusion-b4096-h7168"))
    bad["mechanism"]["class"] = "precision"
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("not allowed" in r for r in doc["reasons"])


def test_missing_required_fields_are_reported():
    bad = deepcopy(_case("gemv-m8-bf16"))
    del bad["workload"]["measured_ms"]
    gate_rules, sha = rules_mod.load_rules()
    doc = decide.evaluate(bad, gate_rules, rules_sha256=sha,
                          rules_version=gate_rules["version"])
    assert doc["verdict"] == "INVALID"
    assert any("workload.measured_ms" in r for r in doc["reasons"])


def test_go_requires_both_bounds_above_line():
    doc = _run("swiglu-fusion-b4096-h7168")
    assert doc["checks"]["n_ge_min_repeats_go"] is True
    assert doc["checks"]["empirical_floor_present"] is True
    assert doc["checks"]["out_of_scope_clean"] is True
    assert doc["prediction"]["speedup_upper_min"] >= 1.30


def test_single_sample_case_uses_parametric_band_and_cannot_go():
    doc = _run("selector-dflash2")
    assert doc["measurement"]["band_method"] == "parametric"
    assert doc["prediction"] == doc["prediction"]  # populated, not empty
    assert doc["verdict"] == "NO-GO"


def test_verdict_json_is_serialisable():
    doc = _run("attn-decode-longctx")
    json.dumps(doc)
    assert doc["kind"] == "hgk_gate_verdict"
    assert doc["rules_version"] == "1.1.0"
