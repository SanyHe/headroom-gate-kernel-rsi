"""Build/load the production-style fused SwiGLU CUDA op (JIT, cached by torch)."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_op = None


def load_op():
    """Compile the extension for sm_120 explicitly.

    The arch list is pinned instead of queried: the RTX 6000D on this host is
    sm_120, and ``_get_cuda_arch_flags`` raises when the device query is
    unavailable, which otherwise kills the build before ninja ever runs.

    CUDA_HOME is set because the image ships CUDA 13.0 under /usr/local/cuda but
    does not put ``nvcc`` on PATH.
    """
    global _op
    if _op is None:
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from torch.utils.cpp_extension import load
        _op = load(
            name="hgk_silu_mul",
            sources=[str(HERE / "fused_silu_mul.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _op
