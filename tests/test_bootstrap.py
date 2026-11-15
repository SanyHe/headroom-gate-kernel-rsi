import random

import pytest

from stats import bootstrap


def test_median_and_mean():
    assert bootstrap.median([3, 1, 2]) == 2
    assert bootstrap.median([4, 1, 3, 2]) == 2.5
    assert bootstrap.mean([1, 2, 3]) == 2
    with pytest.raises(ValueError):
        bootstrap.median([])
    with pytest.raises(ValueError):
        bootstrap.mean([])


def test_stdev_and_cv():
    assert bootstrap.stdev([5.0]) == 0.0
    assert bootstrap.cv([5.0]) == 0.0
    assert bootstrap.cv([1.0, 1.0, 1.0]) == 0.0
    assert bootstrap.cv([1.0, 3.0]) > 0.0


def test_percentile_endpoints_and_interpolation():
    values = [1, 2, 3, 4]
    assert bootstrap.percentile(values, 0) == 1
    assert bootstrap.percentile(values, 100) == 4
    assert bootstrap.percentile(values, 50) == 2.5
    with pytest.raises(ValueError):
        bootstrap.percentile(values, 101)


def test_bootstrap_median_ci_is_deterministic_given_seed():
    samples = [1.0, 1.1, 0.95, 1.05, 1.2]
    a = bootstrap.bootstrap_median_ci(samples, resamples=500, seed=7)
    b = bootstrap.bootstrap_median_ci(samples, resamples=500, seed=7)
    assert a == b
    assert a.lo <= a.med <= a.hi
    assert a.n == len(samples)


def test_parametric_band_is_conservative_on_the_fast_side():
    band = bootstrap.parametric_band([10.0], z=1.6448536269514722, assumed_cv=0.02)
    assert band.med == 10.0
    assert band.lo == pytest.approx(10.0 / 1.032897, rel=1e-5)
    assert band.hi == pytest.approx(10.0 * 1.032897, rel=1e-5)


def test_measurement_interval_switches_on_repeat_count():
    band, method = bootstrap.measurement_interval([1.0] * 4, min_repeats=5,
                                                  resamples=100)
    assert method == "parametric"
    band, method = bootstrap.measurement_interval([1.0] * 5, min_repeats=5,
                                                  resamples=100)
    assert method == "bootstrap"


def test_paired_bootstrap_diff_detects_a_positive_shift():
    rng = random.Random(0)
    control = [1.0 + rng.random() * 0.01 for _ in range(20)]
    candidate = [c + 0.5 + rng.random() * 0.01 for c in control]
    result = bootstrap.paired_bootstrap_diff(control, candidate, resamples=2000, seed=0)
    assert result.excludes_zero
    assert result.lo > 0
    assert result.mean_diff > 0.4


def test_paired_bootstrap_diff_on_identical_arms_straddles_zero():
    rng = random.Random(1)
    control = [1.0 + rng.random() * 0.01 for _ in range(20)]
    candidate = list(control)
    result = bootstrap.paired_bootstrap_diff(control, candidate, resamples=2000, seed=0)
    assert abs(result.mean_diff) < 1e-12
    assert not result.excludes_zero


def test_paired_speedup_ci_ratio_semantics():
    control = [2.0] * 10
    candidate = [1.0] * 10
    result = bootstrap.paired_speedup_ci(control, candidate, resamples=500, seed=0)
    assert result.mean_diff == pytest.approx(2.0)
    assert result.excludes_zero


def test_paired_helpers_reject_bad_input():
    with pytest.raises(ValueError):
        bootstrap.paired_bootstrap_diff([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        bootstrap.paired_bootstrap_diff([], [])
    with pytest.raises(ValueError):
        bootstrap.paired_speedup_ci([1.0], [0.0])
    with pytest.raises(ValueError):
        bootstrap.bootstrap_median_ci([])


def test_paired_result_dict_rounding():
    result = bootstrap.paired_speedup_ci([2.0], [1.0], resamples=100, seed=0)
    payload = result.as_dict()
    assert payload["n"] == 1
    assert payload["excludes_zero"] is True
    assert len(payload["ci95"]) == 2
