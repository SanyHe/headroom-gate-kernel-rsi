"""Cross-check the device constant outside the registered 176 MB shape.

gate_rules.yaml sets hbm_realistic_tbs = 1.613 from a 176 MB three-stream
working set. This script measures the same three-stream traffic and the fused
kernel at 302 MB and 470 MB to see whether that constant survives a larger
working set, or whether it is operator-scale only.
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
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE_DIR = REPO / "cases" / "swiglu-fusion-b4096-h7168"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE_DIR))

from gate import probes  # noqa: E402
from stats import bootstrap  # noqa: E402
import kernel  # noqa: E402

SHAPES = [
    {"batch_tokens": 4096, "hidden": 7168},
    {"batch_tokens": 4096, "hidden": 12288},
    {"batch_tokens": 8192, "hidden": 7168},
]


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


def measure_shape(shape, flush, evict, pairs=20, warmup=10):
    rows, hidden = shape["batch_tokens"], shape["hidden"]
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    ref = kernel.reference(x)
    kernel.silu_and_mul(x, out, block=1024, round_silu=True)
    torch.cuda.synchronize()
    bitwise = int((out == ref).sum().item())
    elements = out.numel()

    control = lambda: kernel.reference(x)  # noqa: E731
    candidate = lambda: kernel.silu_and_mul(x, out, block=1024, round_silu=True)  # noqa: E731

    end = time.time() + 3.0
    while time.time() < end:
        evict(); control(); evict(); candidate()
    torch.cuda.synchronize()
    for _ in range(warmup):
        evict(); control(); torch.cuda.synchronize()
        evict(); candidate(); torch.cuda.synchronize()
    torch.cuda.synchronize()

    control_ms, candidate_ms = [], []
    for i in range(pairs):
        if i % 2 == 0:
            control_ms.append(time_one(control, evict))
            candidate_ms.append(time_one(candidate, evict))
        else:
            candidate_ms.append(time_one(candidate, evict))
            control_ms.append(time_one(control, evict))

    paired = bootstrap.paired_speedup_ci(control_ms, candidate_ms,
                                         resamples=20000, seed=0)
    traffic_bytes = rows * 2 * hidden * 2 + rows * hidden * 2
    traffic_mb = traffic_bytes / 1024 / 1024
    med_c = bootstrap.median(control_ms)
    med_k = bootstrap.median(candidate_ms)
    stream = probes.stream_probe(traffic_mb=int(round(traffic_mb)),
                                 repeats=10, warmup=3)
    return {
        "shape": {"batch_tokens": rows, "hidden": hidden},
        "traffic_mb": traffic_mb,
        "bitwise_equal": bitwise,
        "elements": elements,
        "control_median_ms": med_c,
        "candidate_median_ms": med_k,
        "paired_speedup": paired.as_dict(),
        "control_tbs": traffic_bytes / (med_c * 1e-3) / 1e12,
        "candidate_tbs": traffic_bytes / (med_k * 1e-3) / 1e12,
        "same_bytes_stream_tbs": stream["tbs"],
        "same_bytes_stream_median_ms": stream["median_ms"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", type=Path,
                    default=REPO / "protocol" / "gate_rules.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(0)
    rules = yaml.safe_load(args.rules.read_text())
    constant = float(rules["device"]["hbm_realistic_tbs"])

    flush = torch.empty((256 << 20) // 4, dtype=torch.float32, device="cuda")
    evict = lambda: torch.sum(flush)  # noqa: E731
    before = device_facts()

    rows = [measure_shape(s, flush, evict) for s in SHAPES]
    for row in rows:
        row["stream_over_constant"] = row["same_bytes_stream_tbs"] / constant
        row["candidate_over_stream"] = row["candidate_tbs"] / row["same_bytes_stream_tbs"]

    result = {
        "kind": "swiglu_shape_crosscheck",
        "case_id": "swiglu-fusion-b4096-h7168-rtx6000d",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": kernel.triton.__version__,
        "rules": str(args.rules.relative_to(REPO)),
        "hbm_realistic_tbs": constant,
        "shapes": rows,
        "device_facts_before": before,
    }
    out = args.out or (HERE / "evidence" /
                       f"shape_crosscheck_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "hbm_realistic_tbs": constant,
        "rows": [
            {"shape": r["shape"], "traffic_mb": round(r["traffic_mb"]),
             "candidate_ms": round(r["candidate_median_ms"], 5),
             "candidate_tbs": round(r["candidate_tbs"], 3),
             "stream_tbs": round(r["same_bytes_stream_tbs"], 3),
             "stream_over_constant": round(r["stream_over_constant"], 3),
             "bitwise": f'{r["bitwise_equal"]}/{r["elements"]}',
             "paired": round(r["paired_speedup"]["mean_diff"], 3)}
            for r in rows]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
