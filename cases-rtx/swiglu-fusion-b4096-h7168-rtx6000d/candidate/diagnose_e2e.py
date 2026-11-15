"""Explain the negative e2e result.

Two questions, measured separately:

1. In a real prefill (4096 tokens, Qwen3-8B), how much GPU time does the MLP
   activation actually take per arm, and how much of the layer/step does it
   account for? If the isolated 2x does not show up here, the model-level
   result is not a timing artefact.
2. Does L2 state explain it? Time the same activation at 4096x12288 twice:
   once after a full L2 flush (the isolated-benchmark condition) and once
   immediately after a same-size producer write (the real-MLP condition).

Writes one evidence JSON. Diagnostic only: no ledger event.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_DIR = REPO / "cases" / "swiglu-fusion-b4096-h7168"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE_DIR))

import kernel  # noqa: E402
from e2e_qwen3_8b import FusedSwiGLU, load_model, set_mode  # noqa: E402


def pair():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="path to a local Qwen3-8B checkpoint")
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--isolated-reps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(0)
    model = load_model(args.model, "cuda")
    inter, hidden = model.config.intermediate_size, model.config.hidden_size
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (1, args.prefill_len), device="cuda")

    # Instrument each FusedSwiGLU: one event pair around the activation only.
    activation_events = []

    def make_forward(orig):
        def forward(self, x):
            y = self.gate_up(x)
            y2 = y.view(-1, 2 * self.inter)
            s, e = pair()
            s.record()
            if self.mode == "triton":
                a2 = kernel.silu_and_mul(y2)
            elif self.mode == "cuda":
                a2 = torch.empty((y2.shape[0], self.inter), dtype=y2.dtype,
                                 device=y2.device)
                self.cuda_op(a2, y2)
            else:
                a2 = F.silu(y2[:, : self.inter]) * y2[:, self.inter:]
            e.record()
            activation_events.append((s, e))
            return self.down_proj(a2.view(*y.shape[:-1], self.inter))
        return forward

    FusedSwiGLU.forward = make_forward(FusedSwiGLU.forward)

    in_model = {}
    for mode in ("strided", "cuda", "triton"):
        set_mode(model, mode)
        totals, layers = [], []
        for _ in range(args.reps):
            activation_events.clear()
            torch.cuda.synchronize()
            s, e = pair()
            s.record()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            e.record()
            torch.cuda.synchronize()
            totals.append(float(s.elapsed_time(e)))
            layers.append(sum(float(a.elapsed_time(b)) for a, b in activation_events))
        in_model[mode] = {
            "prefill_ms": totals,
            "activation_ms_per_prefill": layers,
            "activation_count": len(activation_events),
            "activation_share_of_prefill": [a / t for a, t in zip(layers, totals)],
        }

    # CPU enqueue vs GPU drain: a wall-clock split of the same prefill.
    import time
    enqueue = {}
    for mode in ("strided", "cuda", "triton"):
        set_mode(model, mode)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        rows = []
        for _ in range(4):
            t0 = time.perf_counter()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            rows.append({"enqueue_ms": (t1 - t0) * 1e3, "drain_ms": (t2 - t1) * 1e3,
                         "wall_ms": (t2 - t0) * 1e3})
        enqueue[mode] = rows

    # Ablation: remove the activation entirely; what is left is the floor.
    set_mode(model, "strided")
    saved_forward = FusedSwiGLU.forward

    def no_activation(self, x):
        return self.down_proj(self.gate_up(x)[..., : self.inter])

    FusedSwiGLU.forward = no_activation
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    ablation = []
    for _ in range(4):
        t0 = time.perf_counter()
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ablation.append({"enqueue_ms": (t1 - t0) * 1e3,
                         "drain_ms": (t2 - t1) * 1e3,
                         "wall_ms": (t2 - t0) * 1e3})
    FusedSwiGLU.forward = saved_forward

    # Where the CPU time goes, CPU activity only (GPU profiling serialises here).
    from torch.profiler import ProfilerActivity, profile
    set_mode(model, "cuda")
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    evs = prof.key_averages()
    cpu_top = [{"op": e.key, "calls": e.count,
                "self_cpu_ms": e.self_cpu_time_total / 1000}
               for e in sorted(evs, key=lambda e: e.self_cpu_time_total,
                               reverse=True)[:6]]
    cpu_total_ms = sum(e.self_cpu_time_total for e in evs) / 1000

    # Isolated: cold (flush) vs warm (producer write right before, no flush).
    rows = args.prefill_len
    x = torch.randn(rows, 2 * inter, dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, inter), dtype=torch.bfloat16, device="cuda")
    flush = torch.empty((256 << 20) // 4, dtype=torch.float32, device="cuda")
    evict = lambda: torch.sum(flush)  # noqa: E731
    producer = lambda: torch.add(x, 1.0)  # noqa: E731

    def naive():
        return F.silu(x[:, :inter]) * x[:, inter:]

    def cuda_op():
        torch.ops._C.silu_and_mul(out, x)

    def triton():
        return kernel.silu_and_mul(x)

    def timed(fn, cold):
        if cold:
            evict()
        else:
            producer()
        torch.cuda.synchronize()
        s, e = pair()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e))

    isolated = {}
    for name, fn in (("naive", naive), ("cuda", cuda_op), ("triton", triton)):
        for _ in range(5):
            timed(fn, cold=True)
        cold = [timed(fn, cold=True) for _ in range(args.isolated_reps)]
        warm = [timed(fn, cold=False) for _ in range(args.isolated_reps)]
        isolated[name] = {
            "cold_ms": cold,
            "cold_median_ms": sorted(cold)[len(cold) // 2],
            "warm_ms": warm,
            "warm_median_ms": sorted(warm)[len(warm) // 2],
        }

    doc = {
        "kind": "e2e_negative_result_diagnosis",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "prefill_len": args.prefill_len,
        "shape": {"batch_tokens": rows, "hidden": hidden, "intermediate": inter},
        "in_model": in_model,
        "enqueue_vs_drain": enqueue,
        "activation_ablation": ablation,
        "cpu_profile": {"total_ms": cpu_total_ms, "top": cpu_top},
        "isolated": isolated,
        "isolated_traffic_mb":
            (rows * 2 * inter * 2 + rows * inter * 2) / 1024 / 1024,
        "note": "cold = 256 MiB L2 flush before the call (the isolated-benchmark "
                "condition); warm = a same-size producer write immediately "
                "before, no flush (the real-MLP condition).",
    }
    out_path = args.out or (HERE / "evidence" /
                            f"negative_diagnosis_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "in_model": {m: {
            "prefill_ms": [round(v, 1) for v in d["prefill_ms"]],
            "activation_ms": [round(v, 3) for v in d["activation_ms_per_prefill"]],
            "share_pct": [round(100 * s, 3) for s in d["activation_share_of_prefill"]],
            "calls": d["activation_count"],
        } for m, d in in_model.items()},
        "isolated": {m: {"cold_ms": round(d["cold_median_ms"], 4),
                         "warm_ms": round(d["warm_median_ms"], 4)}
                     for m, d in isolated.items()},
        "wall_split_ms": {m: {"enqueue": round(sorted(r["enqueue_ms"] for r in v)[2], 1),
                              "drain": round(sorted(r["drain_ms"] for r in v)[2], 1),
                              "wall": round(sorted(r["wall_ms"] for r in v)[2], 1)}
                          for m, v in enqueue.items()},
        "ablation_none": {"enqueue": round(sorted(r["enqueue_ms"] for r in ablation)[2], 1),
                          "drain": round(sorted(r["drain_ms"] for r in ablation)[2], 1),
                          "wall": round(sorted(r["wall_ms"] for r in ablation)[2], 1)},
        "cpu_profile_total_ms": round(cpu_total_ms, 1),
        "cpu_profile_top": [{"op": r["op"], "calls": r["calls"],
                             "ms": round(r["self_cpu_ms"], 1)} for r in cpu_top[:3]],
    }, indent=2))
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
