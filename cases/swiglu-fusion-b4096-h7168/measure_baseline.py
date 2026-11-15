"""Baseline + same-bytes reference for the SwiGLU fusion case.

Measures the shipped path, the contiguous variant, and a same-bytes reference on
the current host. Evidence only: it never rewrites case.yaml or the ledger.
"""

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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from gate import byte_model  # noqa: E402

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


def timed(fn, repeats, warmup, evict, ramp_s):
    end = time.time() + ramp_s
    while time.time() < end:
        evict(); fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        evict(); torch.cuda.synchronize(); fn()
    torch.cuda.synchronize()
    return [time_one(fn, evict) for _ in range(repeats)]


def summ(samples):
    med = statistics.median(samples)
    return {"samples_ms": samples, "median_ms": med,
            "cv": (statistics.stdev(samples) / statistics.fmean(samples))
                  if len(samples) > 1 else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--ramp-s", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(0)

    x = torch.randn(B, 2 * H, dtype=torch.bfloat16, device=args.device)
    gate, up = x[:, :H], x[:, H:]
    a = torch.randn(B, H, dtype=torch.bfloat16, device=args.device)
    b = torch.randn(B, H, dtype=torch.bfloat16, device=args.device)
    flush = torch.empty((args.flush_mb << 20) // 4, dtype=torch.float32,
                        device=args.device)
    evict = lambda: torch.sum(flush)  # noqa: E731
    evict(); torch.cuda.synchronize()
    before = device_facts()

    baseline = summ(timed(lambda: F.silu(gate) * up, args.repeats, args.warmup,
                          evict, args.ramp_s))
    contiguous = summ(timed(lambda: F.silu(a) * b, args.repeats, args.warmup,
                            evict, args.ramp_s))
    reference = summ(timed(lambda: torch.add(a, b), args.repeats, args.warmup,
                           evict, args.ramp_s))
    silu_only = summ(timed(lambda: F.silu(a), args.repeats, args.warmup,
                           evict, args.ramp_s))
    after = device_facts()

    ledger = {"activations": B * 2 * H * 2, "output": B * H * 2, "weights": 0}
    total = byte_model.sum_bytes(ledger)
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = None
    result = {
        "kind": "swiglu_baseline_and_reference",
        "case_id": "swiglu-fusion-b4096-h7168",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "vllm": vllm_version,
        "dtype": "bf16", "shape": {"batch_tokens": B, "hidden": H},
        "protocol": {"warmup": args.warmup, "repeats": args.repeats,
                     "l2_flush_bytes": args.flush_mb << 20, "l2_flush_mode": "read",
                     "ramp_s": args.ramp_s,
                     "window": "CUDA events around one launch, after an L2 flush and a sync"},
        "byte_ledger": ledger, "ledger_bytes": total,
        "baseline": baseline, "contiguous_baseline": contiguous,
        "reference": reference, "silu_only": silu_only,
        "empirical_floor_ms": reference["median_ms"],
        "empirical_floor_source_arm": "torch.add(a,b): 2 reads + 1 write, same bytes",
        "baseline_gbps": total / (baseline["median_ms"] * 1e-3) / 1e9,
        "reference_gbps": total / (reference["median_ms"] * 1e-3) / 1e9,
        "headroom_ratio": baseline["median_ms"] / reference["median_ms"],
        "contiguous_headroom_ratio": contiguous["median_ms"] / reference["median_ms"],
        "device_facts_before": before, "device_facts_after": after,
    }
    out = args.out or (HERE / "evidence" /
                       f"baseline_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("device", "vllm", "empirical_floor_ms", "baseline_gbps",
                       "reference_gbps", "headroom_ratio",
                       "contiguous_headroom_ratio")}, indent=2))
    for name in ("baseline", "contiguous_baseline", "reference"):
        print(f"{name}: median {result[name]['median_ms']:.5f} ms "
              f"cv {result[name]['cv']:.4f}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
