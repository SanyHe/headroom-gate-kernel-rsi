"""Fused silu_and_mul over an interleaved [B, 2H] bf16 buffer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, H, ROUND_SILU: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H

    gate = tl.load(x_ptr + row * 2 * H + col, mask=mask, other=0.0)
    up = tl.load(x_ptr + row * 2 * H + H + col, mask=mask, other=0.0)

    g = gate.to(tl.float32)
    si = g * tl.sigmoid(g)
    if ROUND_SILU:
        si = si.to(tl.bfloat16).to(tl.float32)   # shipped path rounds silu to bf16
    out = si * up.to(tl.float32)
    tl.store(out_ptr + row * H + col, out.to(tl.bfloat16), mask=mask)


def silu_and_mul(x, out=None, block=1024, round_silu=True):
    rows, two_h = x.shape
    if two_h % 2:
        raise ValueError(f"last dim must be even, got {two_h}")
    hidden = two_h // 2
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"expected bf16, got {x.dtype}")
    if out is None:
        out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
    grid = (rows, triton.cdiv(hidden, block))
    _silu_and_mul_kernel[grid](x, out, hidden, ROUND_SILU=round_silu,
                               BLOCK=block, num_warps=4)
    return out


def reference(x):
    """What a serving stack ships today."""
    import torch.nn.functional as F
    hidden = x.shape[1] // 2
    return F.silu(x[:, :hidden]) * x[:, hidden:]
