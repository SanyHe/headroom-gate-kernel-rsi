"""Robust re-analysis of an e2e artifact: two estimators side by side.

The raw samples are never modified. Per workload this adds the mean-ratio CI,
the median-ratio CI, and a first-touch sensitivity (drop slot 0) so a reader
can see whether one warm sample drives the result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from stats import bootstrap  # noqa: E402


def ci_of(values, fn=bootstrap.mean, resamples=20000, seed=0):
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = [fn([values[rng.randrange(n)] for _ in range(n)])
             for _ in range(resamples)]
    return [bootstrap.percentile(stats, 2.5), bootstrap.percentile(stats, 97.5)]


def block(a, b, drop_first=False):
    if drop_first:
        a, b = a[1:], b[1:]
    r = [x / y for x, y in zip(a, b)]
    return {
        "n": len(r),
        "mean_ratio": bootstrap.mean(r),
        "mean_ci95": ci_of(r),
        "median_ratio": bootstrap.median(r),
        "median_ci95": ci_of(r, fn=bootstrap.median),
        "per_slot": [round(x, 4) for x in r],
        "excluding_first_touch": drop_first,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    doc = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = {}
    for entry in doc["blocks"]:
        a, b = entry["strided_ms"], entry["fused_ms"]
        rows[entry["workload"]] = {
            "full": block(a, b),
            "drop_first": block(a, b, drop_first=True),
        }

    out_doc = {
        "kind": "qwen3_8b_e2e_reanalysis",
        "source": str(args.artifact),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "fused_impl": doc["fused_impl"],
        "device": doc["device"],
        "workloads": rows,
        "note": "mean_ratio is the measurement estimator; median_ratio and the "
                "drop-first variant are sensitivity checks only and must be "
                "reported next to it, never used to re-pick a direction.",
    }
    out = args.out or args.artifact.with_name(
        args.artifact.stem + "_reanalysis.json")
    out.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fused_impl": doc["fused_impl"],
        "workloads": {k: {"mean_ratio": round(v["full"]["mean_ratio"], 4),
                          "mean_ci95": [round(x, 4) for x in v["full"]["mean_ci95"]],
                          "median_ratio": round(v["full"]["median_ratio"], 4),
                          "median_ci95": [round(x, 4) for x in v["full"]["median_ci95"]],
                          "drop_first_mean": round(v["drop_first"]["mean_ratio"], 4)}
                      for k, v in rows.items()}}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
