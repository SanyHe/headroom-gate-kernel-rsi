"""Exploratory timing of P3 candidate operators: shipped call vs same-bytes reference.

Writes only p3/evidence/pool_<ts>.json. This is not a registered P3 run: there
are no low-bit arms and no accuracy trials, so it cannot drive p3.select. It is
also not a gate event, so nothing is appended to the ledger.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

VOCAB, HID = 151936, 7168  # Qwen3-8B embedding / hidden sizes


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


def build(dev):
    """Return list of (id, shipped_fn, reference_fn, read_bytes, write_bytes,
    reference_traffic_bytes, reachable)."""
    rows = []

    x = torch.randn(64, HID, dtype=torch.bfloat16, device=dev)
    w = torch.ones(HID, dtype=torch.bfloat16, device=dev)
    src = torch.randn(64, HID, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    rows.append(("rmsnorm_b64_h7168", lambda: F.rms_norm(x, (HID,), w, 1e-6),
                 lambda: dst.copy_(src), 64 * HID * 2, 64 * HID * 2,
                 2 * 64 * HID * 2, True))

    y = torch.randn(64, HID, dtype=torch.bfloat16, device=dev)
    ln_w = torch.ones(HID, dtype=torch.bfloat16, device=dev)
    ln_b = torch.zeros(HID, dtype=torch.bfloat16, device=dev)
    a2 = torch.randn(64, HID, dtype=torch.bfloat16, device=dev)
    b2 = torch.randn(64, HID, dtype=torch.bfloat16, device=dev)
    rows.append(("layernorm_b64_h7168",
                 lambda: F.layer_norm(y, (HID,), ln_w, ln_b, 1e-6),
                 lambda: torch.add(a2, b2), 2 * 64 * HID * 2, 64 * HID * 2,
                 3 * 64 * HID * 2, True))

    gx = torch.randn(4096, 2 * HID, dtype=torch.bfloat16, device=dev)
    aa = torch.randn(4096, HID, dtype=torch.bfloat16, device=dev)
    bb = torch.randn(4096, HID, dtype=torch.bfloat16, device=dev)
    rows.append(("silu_mul_b4096_h14336",
                 lambda: F.silu(gx[:, :HID]) * gx[:, HID:],
                 lambda: torch.add(aa, bb), 2 * 4096 * HID * 2, 4096 * HID * 2,
                 3 * 4096 * HID * 2, True))

    lg = torch.randn(8, VOCAB, dtype=torch.bfloat16, device=dev)
    cs = torch.randn(8, VOCAB, dtype=torch.bfloat16, device=dev)
    cd = torch.empty_like(cs)
    rows.append(("softmax_b8_v151936", lambda: torch.softmax(lg, -1),
                 lambda: cd.copy_(cs), 8 * VOCAB * 2, 8 * VOCAB * 2,
                 2 * 8 * VOCAB * 2, True))
    rows.append(("argmax_b8_v151936", lambda: torch.argmax(lg, -1),
                 lambda: torch.sum(lg), 8 * VOCAB * 2, 8 * 8,
                 8 * VOCAB * 2, True))

    c1 = torch.randn(1 << 20, dtype=torch.float32, device=dev)
    c2 = torch.randn(1 << 20, dtype=torch.float32, device=dev)
    c3 = torch.empty_like(c2)
    rows.append(("cumsum_1m_fp32", lambda: torch.cumsum(c1, 0),
                 lambda: c3.copy_(c2), (1 << 20) * 4, (1 << 20) * 4,
                 2 * (1 << 20) * 4, True))

    att = torch.randn(32, 32768, dtype=torch.bfloat16, device=dev)
    at2 = torch.randn(32, 32768, dtype=torch.bfloat16, device=dev)
    at3 = torch.empty_like(at2)
    rows.append(("softmax_32x32768", lambda: torch.softmax(att, -1),
                 lambda: at3.copy_(at2), 32 * 32768 * 2, 32 * 32768 * 2,
                 2 * 32 * 32768 * 2, True))

    emb = torch.randn(VOCAB, HID, dtype=torch.bfloat16, device=dev)
    idx = torch.randint(0, VOCAB, (4096,), device=dev)
    gsrc = torch.randn(4096, HID, dtype=torch.bfloat16, device=dev)
    gdst = torch.empty_like(gsrc)
    rows.append(("embed_gather_4096x7168", lambda: F.embedding(idx, emb),
                 lambda: gdst.copy_(gsrc), 4096 * HID * 2, 4096 * HID * 2,
                 2 * 4096 * HID * 2, False))
    return rows


def measure(fn, rounds, warmup, evict):
    for _ in range(warmup):
        evict()
        torch.cuda.synchronize()
        fn()
    torch.cuda.synchronize()
    return [time_one(fn, evict) for _ in range(rounds)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--flush-mb", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(0)
    flush = torch.empty((args.flush_mb << 20) // 4, dtype=torch.float32,
                        device=args.device)
    evict = lambda: torch.sum(flush)  # noqa: E731
    evict()
    torch.cuda.synchronize()

    members, before = [], device_facts()
    for op, shipped, reference, rb, wb, rtb, reachable in build(args.device):
        s = measure(shipped, args.rounds, args.warmup, evict)
        r = measure(reference, args.rounds, args.warmup, evict)
        med_s, med_r = statistics.median(s), statistics.median(r)
        members.append({
            "op": op,
            "read_bytes": rb, "write_bytes": wb, "traffic_bytes": rb + wb,
            "reference_traffic_bytes": rtb, "reference_reachable": reachable,
            "shipped": {"samples_ms": s, "median_ms": med_s,
                        "cv": statistics.stdev(s) / statistics.fmean(s)},
            "reference": {"samples_ms": r, "median_ms": med_r,
                          "cv": statistics.stdev(r) / statistics.fmean(r)},
            "empirical_floor_ms": med_r,
            "headroom_ratio": med_s / med_r,
            "shipped_gbps": (rb + wb) / (med_s * 1e-3) / 1e9,
            "reference_gbps": rtb / (med_r * 1e-3) / 1e9,
        })
        print(f"{op}: shipped {med_s:.5f} ms  ref {med_r:.5f} ms  "
              f"headroom {med_s / med_r:.3f}x", flush=True)

    out = args.out or (HERE / "evidence" /
                       f"pool_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "p3_pool_measurement", "generated_at": datetime.now(
            timezone.utc).astimezone().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "protocol": {"rounds": args.rounds, "warmup": args.warmup,
                     "l2_flush_bytes": args.flush_mb << 20,
                     "l2_flush_mode": "read",
                     "window": "CUDA events around one launch"},
        "members": members, "device_facts_before": before,
        "device_facts_after": device_facts(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
