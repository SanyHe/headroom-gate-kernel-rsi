"""Paired A/B measurement of the fused silu_and_mul against the shipped path."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from gate import byte_model  # noqa: E402
from stats import bootstrap  # noqa: E402
import kernel  # noqa: E402

B, H = 4096, 7168


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


def correctness(x, ref, block):
    hidden = x.shape[1] // 2
    out = torch.empty((x.shape[0], hidden), dtype=x.dtype, device=x.device)
    report = {}
    for round_silu in (True, False):
        kernel.silu_and_mul(x, out, block=block, round_silu=round_silu)
        torch.cuda.synchronize()
        equal = (out == ref)
        diff = (out.float() - ref.float()).abs()
        report["round_silu" if round_silu else "fp32_silu"] = {
            "bitwise_equal": int(equal.sum().item()),
            "elements": out.numel(),
            "bitwise_fraction": equal.float().mean().item(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--ramp-s", type=float, default=3.0)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(0)

    x = torch.randn(B, 2 * H, dtype=torch.bfloat16, device=args.device)
    cand_out = torch.empty((B, H), dtype=torch.bfloat16, device=args.device)
    ref = kernel.reference(x)
    torch.cuda.synchronize()

    checks = correctness(x, ref, args.block)
    chosen_round = (checks["round_silu"]["bitwise_equal"] >=
                    checks["fp32_silu"]["bitwise_equal"])

    control = lambda: kernel.reference(x)  # noqa: E731
    candidate = lambda: kernel.silu_and_mul(x, cand_out, block=args.block,  # noqa: E731
                                            round_silu=chosen_round)

    flush = torch.empty((args.flush_mb << 20) // 4, dtype=torch.float32,
                        device=args.device)
    evict = lambda: torch.sum(flush)  # noqa: E731
    evict(); torch.cuda.synchronize()
    before = device_facts()

    end = time.time() + args.ramp_s
    while time.time() < end:
        evict(); control()
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        evict(); torch.cuda.synchronize(); control()
        evict(); torch.cuda.synchronize(); candidate()
    torch.cuda.synchronize()

    control_ms, candidate_ms = [], []
    for i in range(args.pairs):
        if i % 2 == 0:                     # alternate order inside each pair
            control_ms.append(time_one(control, evict))
            candidate_ms.append(time_one(candidate, evict))
        else:
            candidate_ms.append(time_one(candidate, evict))
            control_ms.append(time_one(control, evict))

    after = device_facts()
    paired = bootstrap.paired_speedup_ci(control_ms, candidate_ms,
                                         resamples=args.resamples, seed=0)
    med_c, med_k = bootstrap.median(control_ms), bootstrap.median(candidate_ms)
    ledger = {"activations": B * 2 * H * 2, "output": B * H * 2, "weights": 0}
    total = byte_model.sum_bytes(ledger)

    result = {
        "kind": "swiglu_fused_paired_measurement",
        "case_id": "swiglu-fusion-b4096-h7168",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": kernel.triton.__version__,
        "dtype": "bf16", "shape": {"batch_tokens": B, "hidden": H},
        "kernel": {"block": args.block, "num_warps": 4, "round_silu": chosen_round,
                   "grid": [B, (H + args.block - 1) // args.block]},
        "correctness": checks,
        "protocol": {"pairs": args.pairs, "warmup": args.warmup,
                     "l2_flush_bytes": args.flush_mb << 20, "l2_flush_mode": "read",
                     "ramp_s": args.ramp_s,
                     "order": "alternating (A,B) on even pairs, (B,A) on odd pairs",
                     "window": "CUDA events around one launch, after an L2 flush and a sync",
                     "resamples": args.resamples, "seed": 0},
        "control_ms": control_ms, "candidate_ms": candidate_ms,
        "control_median_ms": med_c, "candidate_median_ms": med_k,
        "control_cv": bootstrap.cv(control_ms),
        "candidate_cv": bootstrap.cv(candidate_ms),
        "paired_speedup": paired.as_dict(),
        "speedup_of_medians": med_c / med_k,
        "control_gbps": total / (med_c * 1e-3) / 1e9,
        "candidate_gbps": total / (med_k * 1e-3) / 1e9,
        "ledger_bytes": total,
        "device_facts_before": before, "device_facts_after": after,
    }
    out = args.out or (HERE / "evidence" /
                       f"candidate_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "round_silu": chosen_round,
        "bitwise_round": checks["round_silu"]["bitwise_equal"],
        "bitwise_fp32": checks["fp32_silu"]["bitwise_equal"],
        "elements": checks["round_silu"]["elements"],
        "control_median_ms": med_c, "candidate_median_ms": med_k,
        "paired_speedup": result["paired_speedup"],
        "candidate_gbps": result["candidate_gbps"]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
