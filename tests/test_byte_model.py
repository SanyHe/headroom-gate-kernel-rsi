import pytest

from gate import byte_model


def test_dtype_sizes():
    assert byte_model.dtype_size("bf16") == 2
    assert byte_model.dtype_size("fp8") == 1
    with pytest.raises(ValueError):
        byte_model.dtype_size("int4")


def test_gemm_io_bytes_and_sum():
    ledger = byte_model.gemm_io_bytes(8, 5120, 20480, "bf16")
    assert ledger["weights"] == 5120 * 20480 * 2
    assert ledger["activations"] == 8 * 5120 * 2
    assert ledger["output"] == 8 * 20480 * 2
    assert byte_model.sum_bytes(ledger) == 210124800


def test_gemv_is_the_m1_case():
    ledger = byte_model.gemv_io_bytes(5120, 20480)
    assert ledger["weights"] == 209715200
    assert byte_model.sum_bytes(ledger) == 209715200 + 10240 + 40960


def test_sum_bytes_rejects_unknown_fields():
    with pytest.raises(ValueError):
        byte_model.sum_bytes({"weights": 1, "scratch": 2})


def test_gemm_io_bytes_rejects_non_positive_shapes():
    with pytest.raises(ValueError):
        byte_model.gemm_io_bytes(0, 1, 1)


def test_roofline_memory_bound_swiglu():
    ledger = {"activations": 117440512, "output": 58720256, "weights": 0}
    floor = byte_model.roofline(byte_model.sum_bytes(ledger), 176160768, 3.30, 138.0)
    assert floor.bytes_total == 176160768
    assert floor.regime == "memory"
    assert floor.t_mem_ms == pytest.approx(0.0533818, rel=1e-5)
    assert floor.t_compute_ms == pytest.approx(0.001277, abs=1e-6)
    assert floor.t_floor_ms == pytest.approx(0.053382, abs=1e-6)


def test_roofline_compute_bound_labels_regime():
    # 4096^3 BF16 GEMM: 137 GFLOP over 268 MB -> compute bound on this host.
    floor = byte_model.roofline(3 * 4096 * 4096 * 2, byte_model.gemm_flops(4096, 4096, 4096),
                                3.30, 138.0)
    assert floor.regime == "compute"
    assert floor.t_compute_ms > floor.t_mem_ms


def test_empirical_floor_can_only_raise():
    base = byte_model.roofline(134217728, 536870912, 3.30, 138.0)
    raised = byte_model.with_empirical_floor(base, 0.061424)
    assert raised.floor_source == "empirical"
    assert raised.t_floor_ms == 0.061424
    assert raised.analytic_floor_ms == base.t_floor_ms

    ignored = byte_model.with_empirical_floor(base, 0.030)
    assert ignored.floor_source == "analytic"
    assert ignored.t_floor_ms == base.t_floor_ms

    assert byte_model.with_empirical_floor(base, None) is base


def test_empirical_floor_must_be_positive():
    base = byte_model.roofline(1024, 0, 3.30, 138.0)
    with pytest.raises(ValueError):
        byte_model.with_empirical_floor(base, 0.0)


def test_roofline_rejects_empty_stream():
    with pytest.raises(ValueError):
        byte_model.roofline(0, 0, 3.30, 138.0)
    with pytest.raises(ValueError):
        byte_model.roofline(1024, 0, 0.0, 138.0)


def test_gbps_round_trip():
    assert byte_model.gbps(1e9, 1.0) == pytest.approx(1000.0)
