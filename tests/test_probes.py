import sys

from gate import probes


def test_plan_prints_without_touching_cuda(capsys):
    assert probes.main([]) == 0
    out = capsys.readouterr().out
    assert "hbm_copy_2GB" in out
    assert "gemm_bf16_8192" in out
    assert "--execute" in out
    # The plan path must never import torch: this test runs on a GPU-less host.
    assert "torch" not in sys.modules


def test_plan_lists_every_probe_with_kwargs():
    names = [name for name, _, _ in probes.PLAN]
    assert len(names) == len(set(names)) == 5
    for _, fn, kwargs in probes.PLAN:
        assert callable(fn) and isinstance(kwargs, dict)
