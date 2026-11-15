"""One-shot evaluation of the pre-registered holdout shapes.

The manifest hash and the "one evaluation only" rule are both enforced before
any tensor is allocated. The manifest was frozen after the candidate already
existed, so the result is a held-out-shape check, not a blind sealed read; that
deviation is written into the evidence and the ledger note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE_DIR = REPO / "cases" / "swiglu-fusion-b4096-h7168"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE_DIR))

from gate import ledger  # noqa: E402
from stats import bootstrap  # noqa: E402
import kernel  # noqa: E402

CASE_ID = HERE.name


def device_facts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.sm,"
             "clocks.mem,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        return {"rows": [ln.strip() for ln in out.stdout.strip().splitlines()
                         if ln.strip()]}
    except Exception as exc:
        return {"error": repr(exc)}


def time_one(fn, evict):
    evict()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e))


def check_guards(manifest_path: Path, sha_path: Path):
    pinned = sha_path.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual != pinned:
        raise SystemExit(f"sealed manifest hash mismatch: pinned {pinned}, file {actual}")
    prior = [e for e in ledger.read_all()
             if e.get("case_id") == CASE_ID and e.get("event") == "sealed_eval"]
    if prior:
        raise SystemExit(f"sealed set already evaluated at {prior[0].get('ts')}; "
                         "the protocol allows one read")
    return actual


def evaluate_holdout(spec, manifest, args, evict):
    rows = int(spec["batch_tokens"])
    hidden = int(spec.get("hidden", manifest["hidden"]))
    torch.manual_seed(manifest["seed"])
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device=args.device)
    ref = kernel.reference(x)
    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=args.device)
    round_silu = bool(manifest["candidate"]["round_silu"])
    block = int(manifest["candidate"]["block"])
    candidate = lambda: kernel.silu_and_mul(x, out, block=block,  # noqa: E731
                                            round_silu=round_silu)
    control = lambda: kernel.reference(x)  # noqa: E731

    candidate()
    torch.cuda.synchronize()
    equal = int((out == ref).sum().item())
    elements = out.numel()

    end = time.time() + args.ramp_s
    while time.time() < end:
        evict()
        control()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        evict(); torch.cuda.synchronize(); control()
        evict(); torch.cuda.synchronize(); candidate()
    torch.cuda.synchronize()

    control_ms, candidate_ms = [], []
    for i in range(args.pairs):
        if i % 2 == 0:
            control_ms.append(time_one(control, evict))
            candidate_ms.append(time_one(candidate, evict))
        else:
            candidate_ms.append(time_one(candidate, evict))
            control_ms.append(time_one(control, evict))

    paired = bootstrap.paired_speedup_ci(control_ms, candidate_ms,
                                         resamples=args.resamples, seed=0)
    accepted = (equal == elements) and paired.lo > 1.0
    return {
        "id": spec["id"],
        "shape": {"batch_tokens": rows, "hidden": hidden},
        "elements": elements,
        "bitwise_equal": equal,
        "bitwise_fraction": equal / elements,
        "control_median_ms": bootstrap.median(control_ms),
        "candidate_median_ms": bootstrap.median(candidate_ms),
        "control_cv": bootstrap.cv(control_ms),
        "candidate_cv": bootstrap.cv(candidate_ms),
        "paired_speedup": paired.as_dict(),
        "accepted": accepted,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--flush-mb", type=int, default=None)
    ap.add_argument("--ramp-s", type=float, default=None)
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    manifest_path = HERE / "sealed_manifest.json"
    manifest_sha = check_guards(manifest_path, HERE / "sealed_manifest.sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proto = manifest["protocol"]
    args.pairs = args.pairs or int(proto["pairs"])
    args.warmup = args.warmup or int(proto["warmup"])
    args.flush_mb = args.flush_mb or (int(proto["l2_flush_bytes"]) >> 20)
    args.ramp_s = args.ramp_s if args.ramp_s is not None else float(proto["clock_ramp_s"])
    args.resamples = args.resamples or 20000

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.cuda.set_device(torch.device(args.device))

    flush = torch.empty((args.flush_mb << 20) // 4, dtype=torch.float32,
                        device=args.device)
    evict = lambda: torch.sum(flush)  # noqa: E731
    evict(); torch.cuda.synchronize()
    before = device_facts()

    holdouts = [evaluate_holdout(spec, manifest, args, evict)
                for spec in manifest["holdouts"]]
    all_accepted = all(h["accepted"] for h in holdouts)
    after = device_facts()

    result = {
        "kind": "swiglu_sealed_holdout_evaluation",
        "case_id": CASE_ID,
        "manifest_case_id": manifest["case_id"],
        "manifest_sha256": manifest_sha,
        "declared_in": manifest["declared_in"],
        "sealed_status": "post_hoc_freeze",
        "sealed_note": manifest["note"],
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": kernel.triton.__version__,
        "dtype": manifest["dtype"],
        "protocol": {"pairs": args.pairs, "warmup": args.warmup,
                     "l2_flush_bytes": args.flush_mb << 20,
                     "l2_flush_mode": proto["l2_flush_mode"],
                     "ramp_s": args.ramp_s, "order": proto["order"],
                     "window": proto["window"], "resamples": args.resamples,
                     "seed": 0},
        "holdouts": holdouts,
        "all_accepted": all_accepted,
        "device_facts_before": before, "device_facts_after": after,
    }
    out = args.out or (HERE / "evidence" /
                       f"sealed_eval_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    ledger.append({
        "event": "sealed_eval",
        "case_id": CASE_ID,
        "evidence": str(out.relative_to(REPO)),
        "manifest_sha256": manifest_sha,
        "sealed_status": "post_hoc_freeze",
        "all_accepted": all_accepted,
        "holdouts": [{k: h[k] for k in ("id", "bitwise_fraction", "accepted")}
                     | {"speedup": h["paired_speedup"]["mean_diff"],
                        "ci95": h["paired_speedup"]["ci95"]}
                     for h in holdouts],
        "note": "Manifest frozen 2026-11-15 after the candidate existed; the "
                "shapes were pre-registered but this is a held-out-shape check, "
                "not a blind sealed read.",
    })

    print(json.dumps({"manifest_sha256": manifest_sha,
                      "all_accepted": all_accepted,
                      "holdouts": [
                          {"id": h["id"],
                           "speedup": round(h["paired_speedup"]["mean_diff"], 4),
                           "ci95": h["paired_speedup"]["ci95"],
                           "bitwise_fraction": h["bitwise_fraction"],
                           "accepted": h["accepted"]} for h in holdouts]},
                     indent=2))
    print(f"written: {out}")
    return 0 if all_accepted else 1


if __name__ == "__main__":
    sys.exit(main())
