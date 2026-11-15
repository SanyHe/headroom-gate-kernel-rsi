"""End-to-end Qwen3-8B: how much of the operator gain survives the model.

One merged gate_up GEMM per MLP in serving layout, then the activation under
test. Only the activation changes between the two arms; the GEMM and every
other part of the model are identical.

Workloads (reported separately, never mixed):
  prefill 4096 tokens, decode b=1 for 32 steps, decode b=256 for 32 steps.
Cache construction is outside the timing window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_DIR = REPO / "cases" / "swiglu-fusion-b4096-h7168"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE_DIR))

from gate import ledger  # noqa: E402
from stats import bootstrap  # noqa: E402
import kernel  # noqa: E402

CASE_ID = "swiglu-fusion-b4096-h7168-rtx6000d"


class FusedSwiGLU(torch.nn.Module):
    """Serving-style MLP: one merged gate_up GEMM, then the activation under test."""

    def __init__(self, gate_up_w, down_w, inter):
        super().__init__()
        hidden = gate_up_w.shape[1]
        self.inter = inter
        self.gate_up = torch.nn.Linear(hidden, 2 * inter, bias=False,
                                       device=gate_up_w.device, dtype=gate_up_w.dtype)
        with torch.no_grad():
            self.gate_up.weight.copy_(gate_up_w)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False,
                                         device=down_w.device, dtype=down_w.dtype)
        with torch.no_grad():
            self.down_proj.weight.copy_(down_w)
        self.mode = "triton"
        self.cuda_op = None

    def forward(self, x):
        y = self.gate_up(x)
        y2 = y.view(-1, 2 * self.inter)
        if self.mode == "triton":
            a2 = kernel.silu_and_mul(y2)
        elif self.mode == "cuda":
            a2 = torch.empty((y2.shape[0], self.inter), dtype=y2.dtype,
                             device=y2.device)
            self.cuda_op(a2, y2)
        else:
            a2 = F.silu(y2[:, : self.inter]) * y2[:, self.inter:]
        return self.down_proj(a2.view(*y.shape[:-1], self.inter))


def set_mode(model, mode):
    for layer in model.model.layers:
        layer.mlp.mode = mode


def timed(fn, flush):
    flush()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.inference_mode():
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e))


def device_facts():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,"
             "temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        return {"rows": [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]}
    except Exception as exc:
        return {"error": repr(exc)}


def load_model(path, impl):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.eval()
    inter = model.config.intermediate_size
    for layer in model.model.layers:
        old = layer.mlp
        merged = torch.cat([old.gate_proj.weight.detach(),
                            old.up_proj.weight.detach()], dim=0)
        layer.mlp = FusedSwiGLU(merged, old.down_proj.weight.detach(), inter)
        if impl == "cuda":
            import vllm._custom_ops  # noqa: F401
            layer.mlp.cuda_op = torch.ops._C.silu_and_mul
        del old, merged
    torch.cuda.empty_cache()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="path to a local Qwen3-8B checkpoint")
    ap.add_argument("--fused-impl", choices=("triton", "cuda"), default="triton")
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--batch-decode", type=int, default=256)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--operator-speedup", type=float, default=1.91595,
                    help="realised operator-level speedup for the translation block")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = "cuda:0"
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    before = device_facts()
    model = load_model(args.model, args.fused_impl)
    cfg = model.config
    vocab = cfg.vocab_size

    ids = torch.randint(0, vocab, (1, args.prefill_len), device=dev)
    tok = torch.randint(0, vocab, (1, 1), device=dev)
    ids_b = torch.randint(0, vocab, (args.batch_decode, 1), device=dev)
    tok_b = torch.randint(0, vocab, (args.batch_decode, 1), device=dev)
    cps = [torch.tensor([args.prefill_len + k], device=dev)
           for k in range(args.decode_steps)]
    cps_b = [torch.tensor([1 + k], device=dev) for k in range(args.decode_steps)]

    run_prefill = lambda: model(input_ids=ids, use_cache=False)  # noqa: E731

    def fresh_cache():
        with torch.inference_mode():
            return model(input_ids=ids, use_cache=True,
                         logits_to_keep=1).past_key_values

    def decode_steps(cache):
        for step in range(args.decode_steps):
            model(input_ids=tok, past_key_values=cache,
                  cache_position=cps[step], use_cache=True, logits_to_keep=1)

    def batch_cache():
        with torch.inference_mode():
            return model(input_ids=ids_b, use_cache=True,
                         logits_to_keep=1).past_key_values

    def batch_steps(cache):
        for step in range(args.decode_steps):
            model(input_ids=tok_b, past_key_values=cache,
                  cache_position=cps_b[step], use_cache=True, logits_to_keep=1)

    # Correctness first, bitwise, separate from timing.
    ids_small = torch.randint(0, vocab, (1, 128), device=dev)
    set_mode(model, "strided")
    with torch.inference_mode():
        logits_s = model(input_ids=ids_small, use_cache=False).logits.float()
    set_mode(model, args.fused_impl)
    with torch.inference_mode():
        logits_t = model(input_ids=ids_small, use_cache=False).logits.float()
    correctness = {
        "logits_bitwise_equal": bool(torch.equal(logits_s, logits_t)),
        "max_abs_diff": float((logits_s - logits_t).abs().max().item()),
        "argmax_agree": int((logits_s.argmax(-1) == logits_t.argmax(-1)).sum().item()),
        "positions": int(logits_s.shape[1]),
    }

    flush = torch.empty((256 << 20) // 4, dtype=torch.float32, device=dev)
    evict = lambda: torch.sum(flush)  # noqa: E731
    evict()
    torch.cuda.synchronize()

    pre = {"strided": [], args.fused_impl: []}
    dec = {"strided": [], args.fused_impl: []}
    bdec = {"strided": [], args.fused_impl: []}
    for rep in range(args.reps):
        order = ("strided", args.fused_impl) if rep % 2 == 0 else \
            (args.fused_impl, "strided")
        for mode in order:
            set_mode(model, mode)
            pre[mode].append(timed(run_prefill, evict))
            cache = fresh_cache()
            dec[mode].append(timed(lambda: decode_steps(cache), evict))
            cache_b = batch_cache()
            bdec[mode].append(timed(lambda: batch_steps(cache_b), evict))
            print(f"rep {rep} {mode}: prefill {pre[mode][-1]:.2f} ms, "
                  f"decode {dec[mode][-1]:.2f} ms, batch {bdec[mode][-1]:.2f} ms",
                  flush=True)

    fk = args.fused_impl

    def block(name, a, b):
        ci = bootstrap.paired_speedup_ci(a, b, resamples=20000, seed=0)
        return {
            "workload": name,
            "strided_ms": a, "fused_ms": b,
            "median_strided_ms": bootstrap.median(a),
            "median_fused_ms": bootstrap.median(b),
            "median_ratio": bootstrap.median(a) / bootstrap.median(b),
            "speedup": ci.mean_diff, "ci95": [ci.lo, ci.hi],
            "per_slot_ratio": [x / y for x, y in zip(a, b)],
        }

    blocks = [block("prefill_4096", pre["strided"], pre[fk]),
              block("decode_b1_32", dec["strided"], dec[fk]),
              block(f"decode_b{args.batch_decode}_32", bdec["strided"], bdec[fk])]
    by_name = {b["workload"]: b for b in blocks}

    result = {
        "kind": "qwen3_8b_e2e",
        "case_id": CASE_ID,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "fused_impl": fk,
        "model": Path(args.model).name,
        "model_config": {"hidden_size": cfg.hidden_size,
                         "intermediate_size": cfg.intermediate_size,
                         "num_layers": cfg.num_hidden_layers,
                         "vocab_size": vocab},
        "workload": {"prefill_len": args.prefill_len,
                     "decode_steps": args.decode_steps,
                     "batch_decode": args.batch_decode},
        "protocol": {"reps": args.reps,
                     "l2_flush_bytes": 256 << 20, "l2_flush_mode": "read",
                     "window": "CUDA events around the workload; cache build outside",
                     "order": "strided/fused alternate per rep; fused first on odd reps",
                     "resamples": 20000, "seed": 0},
        "correctness": correctness,
        "blocks": blocks,
        "translation": {
            "operator_speedup": args.operator_speedup,
            "prefill_speedup": by_name["prefill_4096"]["speedup"],
            "decode_speedup": by_name["decode_b1_32"]["speedup"],
            "batch_decode_speedup": by_name[f"decode_b{args.batch_decode}_32"]["speedup"],
            "prefill_share_of_operator_gain":
                by_name["prefill_4096"]["speedup"] / args.operator_speedup,
            "decode_share_of_operator_gain":
                by_name["decode_b1_32"]["speedup"] / args.operator_speedup,
            "batch_decode_share_of_operator_gain":
                by_name[f"decode_b{args.batch_decode}_32"]["speedup"] /
                args.operator_speedup,
        },
        "device_facts_before": before,
        "device_facts_after": device_facts(),
    }
    out = args.out or (HERE / "evidence" /
                       f"e2e_{fk}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not args.no_ledger:
        try:
            evidence_rel = str(out.resolve().relative_to(REPO))
        except ValueError:
            evidence_rel = str(out)
        ledger.append({
            "event": "e2e_result",
            "case_id": CASE_ID,
            "fused_impl": fk,
            "device": result["device"],
            "prefill_speedup": result["translation"]["prefill_speedup"],
            "decode_speedup": result["translation"]["decode_speedup"],
            "batch_decode_speedup": result["translation"]["batch_decode_speedup"],
            "prefill_ci95": by_name["prefill_4096"]["ci95"],
            "decode_ci95": by_name["decode_b1_32"]["ci95"],
            "batch_decode_ci95": by_name[f"decode_b{args.batch_decode}_32"]["ci95"],
            "logits_bitwise_equal": correctness["logits_bitwise_equal"],
            "evidence": evidence_rel,
        })

    print(json.dumps({
        "fused_impl": fk,
        "correctness": correctness,
        "prefill": {k: by_name["prefill_4096"][k] for k in ("median_ratio", "speedup", "ci95")},
        "decode_b1": {k: by_name["decode_b1_32"][k] for k in ("median_ratio", "speedup", "ci95")},
        "batch_decode": {k: by_name[f"decode_b{args.batch_decode}_32"][k]
                         for k in ("median_ratio", "speedup", "ci95")},
    }, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
