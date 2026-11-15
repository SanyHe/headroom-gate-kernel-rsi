"""Apply the frozen P3 selection policy to a measured pool.

    python3 -m p3.select --pool p3/pool.json --out p3/selection.json

Pure and deterministic: same pool + same policy bytes -> same selection.
The policy hash frozen at p3/policy.sha256 is checked before anything is
selected, so a selection can never be made under a silently edited policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

POLICY_VERSION = "1.0.0"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO / "p3" / "policy.md"
DEFAULT_HASH = REPO / "p3" / "policy.sha256"
DEFAULT_POOL = REPO / "p3" / "pool.json"
DEFAULT_OUT = REPO / "p3" / "selection.json"

ACC_SLACK = 0.0015          # R2: per-config allowance vs the FP16 reference
POOL_SLACK = 0.0035         # R3: task allowance vs the pool-wide FP16 mean
CAVEAT_MARGIN = 0.002       # R3.b
MIN_FAMILY_TASKS = 3        # R4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_policy_hash(path=DEFAULT_HASH) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def verify_policy(policy_path=DEFAULT_POLICY, hash_path=DEFAULT_HASH):
    actual = sha256_file(policy_path)
    frozen = load_policy_hash(hash_path)
    if actual != frozen:
        raise SystemExit(
            "p3/policy.md does not match p3/policy.sha256: the policy was edited "
            "after the freeze. Re-freeze deliberately (and bump the version) "
            "instead of selecting under a different policy.")
    return actual


def config_id(logical, bits):
    return f"L{logical}:" + ",".join(f"{k}={bits[k]}" for k in sorted(bits))


def total_bits(bits):
    return sum(int(v) for v in bits.values())


def arm_mean(trials, key="fp16_val_acc"):
    return sum(float(t[key]) for t in trials) / len(trials)


def _task_ref(task):
    if "fp16_val_acc" not in task:
        raise ValueError(f"task {task.get('id')}: missing fp16_val_acc")
    return float(task["fp16_val_acc"])


def group_arms(trials):
    arms = defaultdict(list)
    for trial in trials:
        arms[config_id(trial["logical"], trial["bits"])].append(trial)
    return arms


def evaluate_task(task, pool_ref_mean, caveats):
    """Return a task row: selected flag, best config, auditable reasons."""
    task_id = task["id"]
    ref = _task_ref(task)
    reasons, arms = [], group_arms(task.get("trials", []))

    control = [t for t in task.get("trials", []) if int(t["logical"]) == 16]
    if control and any(t.get("diverged") for t in control):
        return _row(task, ref, None, False, ["R1: the FP16 control arm diverged, "
                                             "task excluded"])

    scored = []
    for cid, trials in sorted(arms.items()):
        if any(t.get("diverged") for t in trials):
            reasons.append(f"R1: {cid} excluded (diverged trial)")
            continue
        mean_acc = arm_mean(trials)
        eligible = mean_acc >= ref - ACC_SLACK
        scored.append({
            "config_id": cid,
            "logical": int(trials[0]["logical"]),
            "bits": dict(trials[0]["bits"]),
            "total_bits": total_bits(trials[0]["bits"]),
            "seeds": len(trials),
            "mean_acc": mean_acc,
            "eligible": eligible,
        })
        if not eligible:
            reasons.append(f"R2: {cid} mean {mean_acc:.5f} < ref-{ACC_SLACK} "
                           f"({ref - ACC_SLACK:.5f})")

    eligible = [s for s in scored if s["eligible"]]
    if not eligible:
        return _row(task, ref, None, False, reasons + ["R2: no eligible config"])

    best = max(eligible, key=lambda s: (s["mean_acc"], -s["total_bits"], s["config_id"]))
    reasons.append(f"R2: best {best['config_id']} at {best['mean_acc']:.5f}")

    caveat = caveats.get(task_id)
    pool_floor = pool_ref_mean - POOL_SLACK
    selected = best["mean_acc"] >= pool_floor
    if not selected:
        reasons.append(f"R3: {best['mean_acc']:.5f} < pool mean - {POOL_SLACK} "
                       f"({pool_floor:.5f})")
    if caveat is not None:
        if best["mean_acc"] <= float(caveat) + CAVEAT_MARGIN:
            selected = False
            reasons.append(f"R3.b: {best['mean_acc']:.5f} <= caveat {float(caveat):.5f} "
                           f"+ {CAVEAT_MARGIN}")
        else:
            reasons.append(f"R3.b: above caveat {float(caveat):.5f} + {CAVEAT_MARGIN}")

    return _row(task, ref, best, selected, reasons)


def _row(task, ref, best, selected, reasons):
    return {
        "task_id": task["id"],
        "mechanism_class": task.get("mechanism_class"),
        "fp16_ref_acc": ref,
        "selected": bool(selected),
        "best": None if best is None else {
            "config_id": best["config_id"],
            "logical": best["logical"],
            "bits": best["bits"],
            "total_bits": best["total_bits"],
            "seeds": best["seeds"],
            "mean_acc": round(best["mean_acc"], 6),
            "gain_vs_fp16": round(best["mean_acc"] - ref, 6),
        },
        "reasons": reasons,
    }


def select(pool):
    tasks = pool.get("tasks", [])
    if not tasks:
        raise SystemExit("pool has no tasks; refusing to emit a selection")
    refs = [_task_ref(t) for t in tasks]
    ref_mean = sum(refs) / len(refs)
    rows = [evaluate_task(t, ref_mean, pool.get("fp16_acc_caveat") or {})
            for t in tasks]

    selected = [r for r in rows if r["selected"]]
    by_class = defaultdict(list)
    for row in selected:
        by_class[row["mechanism_class"] or "unknown"].append(row)

    family = None
    if by_class:
        candidates = [(name, items) for name, items in by_class.items()
                      if len(items) >= MIN_FAMILY_TASKS]
        if candidates:
            def rank(item):
                gains = [r["best"]["gain_vs_fp16"] for r in item[1]]
                gains.sort()
                median_gain = gains[len(gains) // 2] if len(gains) % 2 else \
                    0.5 * (gains[len(gains) // 2 - 1] + gains[len(gains) // 2])
                return (len(item[1]), median_gain, item[0])
            name, items = max(candidates, key=rank)
            family = {"mechanism_class": name, "task_count": len(items),
                      "tasks": [r["task_id"] for r in items]}

    selected_classes = sorted({r["mechanism_class"] for r in selected
                               if r["mechanism_class"]})
    joint = _joint_block(pool, selected_classes)

    return {
        "kind": "hgk_p3_selection",
        "policy_version": POLICY_VERSION,
        "policy_sha256": verify_policy(),
        "pool_sha256": None,
        "fp16_ref_mean": round(ref_mean, 6),
        "tasks": rows,
        "family": family,
        "selected_mechanisms": selected_classes,
        "joint": joint,
        "p1_mechanism_classes": selected_classes,
        "notes": [
            "R5: joint values must be measured for the exact combination; the "
            "planning_sum is a sum of per-class impacts and is not the joint value.",
            "R6: only these classes may be cited by P1 over P3.",
        ],
    }


def _joint_block(pool, selected_classes):
    measured = pool.get("joint")
    if measured and sorted(measured.get("mechanisms", [])) == selected_classes \
            and measured.get("value") is not None:
        return {
            "status": "measured",
            "value": float(measured["value"]),
            "mechanisms": selected_classes,
            "source": measured.get("source"),
        }
    impacts = {m: float(v) for m, v in (pool.get("per_class_impact_pct") or {}).items()
               if m in selected_classes}
    planning = sum(impacts.values()) if impacts else None
    return {
        "status": "requires_joint_measurement",
        "value": None,
        "mechanisms": selected_classes,
        "planning_sum_pct": planning,
        "per_class_impact_pct": impacts,
        "source": None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    verify_policy()
    if not args.pool.exists():
        raise SystemExit(f"missing pool: {args.pool} (run p3/measure_pool.py on a GPU)")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    doc = select(pool)
    doc["pool_sha256"] = sha256_file(args.pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"selected {sum(1 for r in doc['tasks'] if r['selected'])}/{len(doc['tasks'])} "
          f"tasks · family={doc['family'] and doc['family']['mechanism_class']} "
          f"· mechanisms={doc['selected_mechanisms']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
