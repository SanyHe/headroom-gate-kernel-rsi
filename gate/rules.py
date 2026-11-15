"""Load and hash-check the frozen gate rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RULES_PATH = REPO / "protocol" / "gate_rules.yaml"

# A missing key is a hard error, not a default: defaults are how thresholds drift.
REQUIRED = {
    "device": ("hbm_realistic_tbs", "bf16_peak_tflops"),
    "decision": ("go_min_speedup", "veto_below_speedup", "min_headroom_fraction",
                 "min_repeats_go", "max_cv_for_go", "assumed_cv_single_sample",
                 "z90", "require_empirical_floor_for_go"),
    "statistics": ("bootstrap_resamples", "seed", "ci_lower_pct", "ci_upper_pct"),
    "scope": ("forbidden_changes", "allowed_mechanism_classes"),
    "measurement": ("min_samples_for_evidence", "noise_floor_fraction"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_rules(path=None):
    """Return (rules, sha256). Raises on a structurally incomplete file."""
    path = Path(path or RULES_PATH)
    raw = path.read_bytes()
    rules = yaml.safe_load(raw)
    if not isinstance(rules, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    missing = []
    for section, keys in REQUIRED.items():
        if section not in rules or not isinstance(rules[section], dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in rules[section]:
                missing.append(f"{section}.{key}")
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
    if not rules.get("version"):
        raise ValueError(f"{path}: missing version")
    return rules, sha256_bytes(raw)
