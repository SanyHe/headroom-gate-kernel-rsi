"""Optional hardware probes used to populate the gate's device constants."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

GB = 1 << 30


def _device_facts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,"
             "temperature.gpu,power.draw,persistence_mode",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        return {"rows": [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]}
    except Exception as exc:
        return {"error": repr(exc)}


def _time_ms(torch, fn, repeats, warmup, evict=None):
    for _ in range(warmup):
        if evict is not None:
            evict()
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if evict is not None:
            evict()
            torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        samples.append(float(s.elapsed_time(e)))
    return samples


def _evict_fn(torch, mb=256):
    buf = torch.empty((mb << 20) // 4, dtype=torch.float32, device="cuda")
    if buf.numel() == 0:
        return None
    return lambda: torch.sum(buf)


def hbm_copy_probe(size_gb=2.0, repeats=10, warmup=3):
    """Sustained copy bandwidth; counts read+write bytes together."""
    import torch
    n = int(size_gb * GB) // 4
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    evict = _evict_fn(torch)
    samples = _time_ms(torch, lambda: dst.copy_(src), repeats, warmup, evict)
    med = statistics.median(samples)
    return {"probe": "hbm_copy", "size_gb": size_gb, "median_ms": med,
            "tbs_bidirectional": (2 * n * 4) / (med * 1e-3) / 1e12,
            "samples_ms": samples}


def hbm_read_probe(size_gb=4.0, repeats=10, warmup=3):
    """Read-only reduction bandwidth: the ceiling that bounds read-bound ops."""
    import torch
    n = int(size_gb * GB) // 4
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    evict = _evict_fn(torch)
    samples = _time_ms(torch, lambda: torch.sum(x), repeats, warmup, evict)
    med = statistics.median(samples)
    return {"probe": "hbm_read_sum", "size_gb": size_gb, "median_ms": med,
            "tbs_read": (n * 4) / (med * 1e-3) / 1e12, "samples_ms": samples}


def gemm_probe(m, k, n, dtype="bf16", repeats=10, warmup=5):
    import torch
    td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    a = torch.randn(m, k, dtype=td, device="cuda")
    b = torch.randn(k, n, dtype=td, device="cuda")
    evict = _evict_fn(torch)
    samples = _time_ms(torch, lambda: torch.matmul(a, b), repeats, warmup, evict)
    med = statistics.median(samples)
    return {"probe": "gemm", "shape": {"m": m, "k": k, "n": n}, "dtype": dtype,
            "median_ms": med, "tflops": (2 * m * k * n) / (med * 1e-3) / 1e12,
            "samples_ms": samples}


def stream_probe(traffic_mb=176, repeats=10, warmup=3):
    """Three-stream (2 reads + 1 write) bandwidth at an operator-scale size."""
    import torch
    n = int(traffic_mb * 1024 * 1024 / 3) // 2  # bf16 elements per buffer
    a = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(a)
    evict = _evict_fn(torch)
    samples = _time_ms(torch, lambda: torch.add(a, b, out=out), repeats, warmup,
                       evict)
    med = statistics.median(samples)
    return {"probe": "stream_add", "traffic_mb": traffic_mb, "median_ms": med,
            "tbs": (3 * n * 2) / (med * 1e-3) / 1e12, "samples_ms": samples}


PLAN = [
    ("hbm_copy_2GB", hbm_copy_probe, dict(size_gb=2.0)),
    ("hbm_copy_4GB", hbm_copy_probe, dict(size_gb=4.0)),
    ("hbm_read_4GB", hbm_read_probe, dict(size_gb=4.0)),
    # Operator-scale working sets: 176 MB of traffic matches the SwiGLU case.
    ("hbm_copy_176MB_traffic", hbm_copy_probe, dict(size_gb=176 / 1024 / 2)),
    ("hbm_copy_256MB_traffic", hbm_copy_probe, dict(size_gb=256 / 1024 / 2)),
    ("hbm_read_176MB", hbm_read_probe, dict(size_gb=176 / 1024)),
    ("stream_176MB", stream_probe, dict(traffic_mb=176)),
    ("stream_1GB", stream_probe, dict(traffic_mb=1024)),
    ("gemm_bf16_4096", gemm_probe, dict(m=4096, k=4096, n=4096)),
    ("gemm_bf16_8192", gemm_probe, dict(m=8192, k=8192, n=8192)),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="actually run on GPU")
    ap.add_argument("--out", type=Path, default=Path("evidence/device_probes.json"))
    args = ap.parse_args(argv)

    if not args.execute:
        print("plan (pass --execute to run):")
        for name, _, kwargs in PLAN:
            print(f"  {name}: {kwargs}")
        print(f"would write {args.out}")
        return 0

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; probes need a GPU")
    torch.manual_seed(0)
    results = {"kind": "hgk_device_probes", "device_facts": _device_facts(),
               "torch": torch.__version__}
    for name, fn, kwargs in PLAN:
        results[name] = fn(**kwargs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "device_facts"}, indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
