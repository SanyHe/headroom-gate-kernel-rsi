"""Join a case's baseline and candidate evidence into one comparison artifact.

    python3 scripts/compare.py --case cases/<id>
    python3 scripts/compare.py --case cases/<id> --baseline <json> --candidate <json>

Reads the newest baseline_*.json and candidate_*.json under the case's
evidence/ directory unless explicit paths are given. The output is evidence, not
a verdict: the gate remains the only thing that decides.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def latest(case_dir, prefix):
    files = sorted((case_dir / "evidence").glob(f"{prefix}_*.json"))
    if not files:
        raise SystemExit(f"no {prefix}_*.json under {case_dir / 'evidence'}")
    return files[-1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repo_relative(path):
    """Keep published evidence free of absolute host paths."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return Path(path).name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, help="case directory")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    case_dir = (REPO / args.case).resolve() if not Path(args.case).is_absolute() \
        else Path(args.case).resolve()
    baseline_path = Path(args.baseline) if args.baseline else latest(case_dir, "baseline")
    candidate_path = Path(args.candidate) if args.candidate else latest(case_dir, "candidate")
    base, cand = load(baseline_path), load(candidate_path)

    if base.get("case_id") != cand.get("case_id"):
        raise SystemExit(f"case_id mismatch: {base.get('case_id')} vs {cand.get('case_id')}")

    ref_ms = base["reference"]["median_ms"]
    control_ms = cand["control_median_ms"]
    candidate_ms = cand["candidate_median_ms"]
    paired = cand["paired_speedup"]

    result = {
        "kind": "hgk_case_comparison",
        "case_id": case_dir.name,
        "measurement_case_id": cand["case_id"],
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "device": cand.get("device"),
        "torch": cand.get("torch"), "cuda": cand.get("cuda"),
        "triton": cand.get("triton"),
        "shape": cand.get("shape"), "dtype": cand.get("dtype"),
        "ledger_bytes": cand.get("ledger_bytes"),
        "sources": {"baseline": repo_relative(baseline_path),
                    "candidate": repo_relative(candidate_path)},
        "arms": {
            "baseline": {"median_ms": base["baseline"]["median_ms"],
                         "cv": base["baseline"]["cv"],
                         "gbps": base.get("baseline_gbps")},
            "contiguous_baseline": {"median_ms": base["contiguous_baseline"]["median_ms"],
                                    "cv": base["contiguous_baseline"]["cv"]},
            "reference": {"median_ms": ref_ms, "cv": base["reference"]["cv"],
                          "gbps": base.get("reference_gbps")},
            "control": {"median_ms": control_ms, "cv": cand.get("control_cv"),
                        "gbps": cand.get("control_gbps")},
            "candidate": {"median_ms": candidate_ms, "cv": cand.get("candidate_cv"),
                          "gbps": cand.get("candidate_gbps")},
        },
        "headroom_ratio": base.get("headroom_ratio"),
        "contiguous_headroom_ratio": base.get("contiguous_headroom_ratio"),
        "paired_speedup": paired,
        "speedup_of_medians": cand.get("speedup_of_medians"),
        "correctness": cand.get("correctness"),
        "checks": {
            "candidate_at_or_above_reference": candidate_ms >= ref_ms,
            "candidate_ratio_to_reference": candidate_ms / ref_ms,
            "bitwise_fraction": cand["correctness"][
                "round_silu" if cand["kernel"]["round_silu"] else "fp32_silu"
            ]["bitwise_fraction"],
            "paired_ci_excludes_one": paired["ci95"][0] > 1.0,
        },
    }

    out = Path(args.out) if args.out else (
        case_dir / "evidence" /
        f"comparison_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": result["case_id"],
                      "control_ms": control_ms, "candidate_ms": candidate_ms,
                      "reference_ms": ref_ms,
                      "paired_speedup": paired,
                      "checks": result["checks"]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
