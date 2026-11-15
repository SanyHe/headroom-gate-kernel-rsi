"""Command line for the headroom gate.

    python3 -m gate.cli cases/<id>          evaluate one case, write gate_verdict.json
    python3 -m gate.cli --all               evaluate every case under cases/
    python3 -m gate.cli --all --record      ... and append each verdict to the ledger
    python3 -m gate.cli --verify-ledger     P1 ratchet check over the ledger

Exit codes: 0 = clean, 3 = some case INVALID, 4 = P1 violation found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from gate import decide, ledger, rules as rules_mod

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "cases"


def load_case(case_dir):
    path = Path(case_dir) / "case.yaml"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    case.setdefault("id", Path(case_dir).name)
    return case


def evaluate_case(case_dir, gate_rules, rules_sha, rules_version):
    case = load_case(case_dir)
    doc = decide.evaluate(case, gate_rules, rules_sha256=rules_sha,
                          rules_version=rules_version)
    doc["generated_at"] = ledger.now()
    doc["case_sha256"] = rules_mod.sha256_file(Path(case_dir) / "case.yaml")
    (Path(case_dir) / "gate_verdict.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def cmd_one(args):
    gate_rules, rules_sha = rules_mod.load_rules()
    version = gate_rules["version"]
    if args.all:
        case_dirs = sorted(p.parent for p in CASES.glob("*/case.yaml"))
    else:
        case_dirs = [Path(args.case).resolve()]
    if not case_dirs:
        print(f"no cases found under {CASES}", file=sys.stderr)
        return 3

    rc = 0
    for case_dir in case_dirs:
        doc = evaluate_case(case_dir, gate_rules, rules_sha, version)
        print(decide.summarize(doc))
        if args.json:
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        if args.record:
            ledger.append({
                "event": "gate",
                "case_id": doc["case_id"],
                "verdict": doc["verdict"],
                "rules_sha256": rules_sha,
                "rules_version": version,
                "case_sha256": doc["case_sha256"],
                "prediction": doc.get("prediction"),
                "reasons": doc.get("reasons"),
            })
        if doc["verdict"] == "INVALID":
            rc = 3
    return rc


def cmd_verify_ledger():
    entries = ledger.read_all()
    violations = ledger.verify_ratchet(entries)
    state = ledger.state_by_case(entries)
    print(f"ledger entries: {len(entries)} · cases: {len(state)}")
    for case_id, info in sorted(state.items()):
        print(f"  {case_id}: last={info.get('last_event')} "
              f"verdict={info.get('verdict', '-')} "
              f"predicted_upper={info.get('predicted_upper_max', '-')} "
              f"realised={info.get('realised_speedup', '-')}")
    if violations:
        print("\nP1 VIOLATIONS:")
        for v in violations:
            print(f"  {v}")
        return 4
    print("\nP1 ratchet: no violations")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="HGK-RSI headroom gate")
    ap.add_argument("case", nargs="?", help="case directory, e.g. cases/gemv-m8-bf16")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify-ledger", action="store_true")
    args = ap.parse_args(argv)

    if args.verify_ledger:
        return cmd_verify_ledger()
    if not args.case and not args.all:
        ap.error("give a case directory or --all")
    return cmd_one(args)


if __name__ == "__main__":
    sys.exit(main())
