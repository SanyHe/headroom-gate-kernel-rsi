"""Analytic byte and FLOP ledger for operator-level workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

DTYPE_BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


def dtype_size(dtype):
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}") from None


def gemm_io_bytes(m, k, n, dtype="bf16", out_dtype=None):
    """Compulsory traffic of one dense GEMM/GEMV call: read A, read B, write C."""
    if min(m, k, n) <= 0:
        raise ValueError(f"non-positive shape: m={m} k={k} n={n}")
    b = dtype_size(dtype)
    o = dtype_size(out_dtype or dtype)
    return {"weights": k * n * b, "activations": m * k * b, "output": m * n * o}


def gemv_io_bytes(k, n, dtype="bf16"):
    return gemm_io_bytes(1, k, n, dtype)


def sum_bytes(ledger):
    unknown = set(ledger) - {"weights", "activations", "output"}
    if unknown:
        raise ValueError(f"unknown byte-ledger fields: {sorted(unknown)}")
    return int(sum(int(v) for v in ledger.values()))


@dataclass(frozen=True)
class Floor:
    bytes_total: int
    flops: int
    bandwidth_tbs: float
    tflops: float
    t_mem_ms: float
    t_compute_ms: float
    t_floor_ms: float
    regime: str
    arithmetic_intensity: float
    analytic_floor_ms: float | None = None
    empirical_floor_ms: float | None = None
    floor_source: str = "analytic"

    def as_dict(self):
        return asdict(self)


def roofline(bytes_total, flops, bandwidth_tbs, tflops, balanced_ratio=1.10):
    """max(t_mem, t_compute); balanced_ratio only labels the regime."""
    if bytes_total <= 0:
        raise ValueError("bytes_total must be > 0 (a streaming floor needs bytes)")
    if bandwidth_tbs <= 0 or tflops <= 0:
        raise ValueError("bandwidth_tbs and tflops must be > 0")

    t_mem = bytes_total / (bandwidth_tbs * 1e12) * 1e3
    t_compute = (flops / (tflops * 1e12) * 1e3) if flops and flops > 0 else 0.0
    t_floor = max(t_mem, t_compute)

    if t_compute <= 0 or t_mem > t_compute * balanced_ratio:
        regime = "memory"
    elif t_compute > t_mem * balanced_ratio:
        regime = "compute"
    else:
        regime = "balanced"

    return Floor(
        bytes_total=int(bytes_total), flops=int(flops or 0),
        bandwidth_tbs=float(bandwidth_tbs), tflops=float(tflops),
        t_mem_ms=round(t_mem, 6), t_compute_ms=round(t_compute, 6),
        t_floor_ms=round(t_floor, 6), regime=regime,
        arithmetic_intensity=round((flops or 0) / bytes_total, 6),
    )


def with_empirical_floor(floor, empirical_floor_ms):
    """Fold a measured same-bytes floor into an analytic roofline.

    Direction matters: raising a floor *reduces* headroom, so this can never
    invent a gain and cannot break P1. A lower empirical value is ignored.
    """
    if empirical_floor_ms is None:
        return floor
    empirical = float(empirical_floor_ms)
    if empirical <= 0:
        raise ValueError("empirical_floor_ms must be > 0")
    raised = empirical > floor.t_floor_ms
    return Floor(
        bytes_total=floor.bytes_total, flops=floor.flops,
        bandwidth_tbs=floor.bandwidth_tbs, tflops=floor.tflops,
        t_mem_ms=floor.t_mem_ms, t_compute_ms=floor.t_compute_ms,
        t_floor_ms=round(max(floor.t_floor_ms, empirical), 6),
        regime=floor.regime, arithmetic_intensity=floor.arithmetic_intensity,
        analytic_floor_ms=floor.t_floor_ms, empirical_floor_ms=round(empirical, 6),
        floor_source="empirical" if raised else "analytic",
    )


def gemm_flops(m, k, n):
    return 2 * int(m) * int(k) * int(n)


def gbps(bytes_total, ms):
    return bytes_total / (ms * 1e-3) / 1e9
