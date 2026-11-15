"""Three-arm dispatch measurement: naive, Triton, and a compiled CUDA op.

The spec's third arm is a locally compiled cuda_ext. This host has no nvcc and
both the configured mirror and PyPI cannot supply one, so the third arm is the
image's own prebuilt fused op (vLLM torch.ops._C.silu_and_mul). It is a real
compiled CUDA kernel over the same buffer, and it is labelled as a substitute
in the evidence; the locally built cuda_ext remains unbuilt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_DIR = REPO / "cases" / "swiglu-fusion-b4096-h7168"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE_DIR))

from gate import ledger  # noqa: E402
from stats import bootstrap  # noqa: E402
import kernel  # noqa: E402

CASE_ID = "swiglu-fusion-b4096-h7168-rtx6000d"
SHAPES = [
    {"id": "registered", "batch_tokens": 4096, "hidden": 7168},
    {"id": "prefill", "batch_tokens": 4096, "hidden": 12288},
    {"id": "decode-b256", "batch_tokens": 256, "hidden": 12288},
    {"id": "decode-b1", "batch_tokens": 1, "hidden": 12288},
]


def _device_facts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,"
             "temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        return {"rows": [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]}
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


def cuda_op_available():
    try:
        import vllm._custom_ops  # noqa: F401
        return hasattr(torch.ops._C, "silu_and_mul")
    except Exception:
        return False


def measure_shape(spec, use_cuda_op, flush, evict, pairs, warmup, ramp_s):
    rows, hidden = spec["batch_tokens"], spec["hidden"]
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device="cuda")
    triton_out = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    cuda_out = torch.empty_like(triton_out)

    def naive():
        return kernel.reference(x)

    def triton():
        return kernel.silu_and_mul(x, triton_out, block=1024, round_silu=True)

    def cuda():
        torch.ops._C.silu_and_mul(cuda_out, x)
        return cuda_out

    ref = naive()
    triton()
    torch.cuda.synchronize()
    checks = {
        "triton_bitwise": int((triton_out == ref).sum().item()),
        "elements": triton_out.numel(),
    }
    if use_cuda_op:
        cuda()
        torch.cuda.synchronize()
        checks["cuda_bitwise"] = int((cuda_out == ref).sum().item())
        checks["cuda_vs_triton_bitwise"] = int((cuda_out == triton_out).sum().item())
        checks["max_abs_diff"] = (cuda_out.float() - ref.float()).abs().max().item()

    arms = {"naive": naive, "triton": triton}
    if use_cuda_op:
        arms["cuda"] = cuda

    end = time.time() + ramp_s
    while time.time() < end:
        for fn in arms.values():
            evict()
            fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        for fn in arms.values():
            evict()
            fn()
            torch.cuda.synchronize()
    torch.cuda.synchronize()

    samples = {name: [] for name in arms}
    order = list(arms.items())
    for i in range(pairs):
        sequence = order if i % 2 == 0 else list(reversed(order))
        for name, fn in sequence:
            samples[name].append(time_one(fn, evict))

    medians = {name: bootstrap.median(v) for name, v in samples.items()}
    out = {
        "id": spec["id"],
        "shape": {"batch_tokens": rows, "hidden": hidden},
        "traffic_mb": (rows * 2 * hidden * 2 + rows * hidden * 2) / 1024 / 1024,
        "correctness": checks,
        "samples_ms": samples,
        "median_ms": medians,
    }
    if use_cuda_op:
        out["naive_vs_cuda"] = bootstrap.paired_speedup_ci(
            samples["naive"], samples["cuda"], resamples=20000, seed=0).as_dict()
        out["triton_vs_cuda"] = bootstrap.paired_speedup_ci(
            samples["triton"], samples["cuda"], resamples=20000, seed=0).as_dict()
        out["naive_vs_triton"] = bootstrap.paired_speedup_ci(
            samples["naive"], samples["triton"], resamples=20000, seed=0).as_dict()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--ramp-s", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(0)
    use_cuda_op = cuda_op_available()
    third_arm = ("vllm_prebuilt_torch_ops_C_silu_and_mul" if use_cuda_op else "unavailable")

    flush = torch.empty((args.flush_mb << 20) // 4, dtype=torch.float32, device="cuda")
    evict = lambda: torch.sum(flush)  # noqa: E731
    before = _device_facts()
    rows = [measure_shape(s, use_cuda_op, flush, evict, args.pairs, args.warmup,
                          args.ramp_s) for s in SHAPES]

    result = {
        "kind": "swiglu_dispatch",
        "case_id": CASE_ID,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": kernel.triton.__version__,
        "third_arm": third_arm,
        "third_arm_note": (
            "cuda_ext/fused_silu_mul.cu was not built: no nvcc on the host and no "
            "wheel available from the configured mirror or PyPI. The third arm is "
            "the image's prebuilt compiled CUDA fused op instead."),
        "protocol": {"pairs": args.pairs, "warmup": args.warmup,
                     "l2_flush_bytes": args.flush_mb << 20, "l2_flush_mode": "read",
                     "ramp_s": args.ramp_s,
                     "order": "all arms in order on even pairs, reversed on odd pairs",
                     "window": "CUDA events around one launch, after a flush and a sync",
                     "resamples": 20000, "seed": 0},
        "shapes": rows,
        "device_facts_before": before,
    }
    out = args.out or (HERE / "evidence" /
                       f"dispatch_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    try:
        evidence_rel = str(out.resolve().relative_to(REPO))
    except ValueError:
        evidence_rel = str(out)
    summary = [{
        "id": r["id"], "shape": r["shape"],
        "naive_ms": round(r["median_ms"]["naive"], 5),
        "triton_ms": round(r["median_ms"]["triton"], 5),
        "cuda_ms": round(r["median_ms"].get("cuda", float("nan")), 5),
        "naive_vs_cuda": round(r.get("naive_vs_cuda", {}).get("mean_diff", float("nan")), 3),
        "triton_vs_cuda": round(r.get("triton_vs_cuda", {}).get("mean_diff", float("nan")), 3),
        "bitwise": f'{r["correctness"].get("triton_bitwise")}/{r["correctness"]["elements"]}',
    } for r in rows]
    event = ledger.append({
        "event": "dispatch_result",
        "case_id": CASE_ID,
        "evidence": evidence_rel,
        "third_arm": third_arm,
        "shapes": summary,
        "note": "third arm is the image's prebuilt CUDA op; cuda_ext not built (no nvcc)",
    })

    print(json.dumps({"third_arm": third_arm, "rows": summary,
                      "ledger_ts": event["ts"]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
