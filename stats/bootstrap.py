"""Small, dependency-light statistics used by the gate and by case evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass


def median(values):
    if not values:
        raise ValueError("median of empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def mean(values):
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def cv(values):
    """Coefficient of variation (sample sd / mean). 0 for n<2."""
    m = mean(values)
    return 0.0 if m == 0 else stdev(values) / abs(m)


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass(frozen=True)
class Interval:
    lo: float
    med: float
    hi: float
    n: int


def bootstrap_median_ci(samples, resamples=10000, seed=0, lo_pct=5.0, hi_pct=95.0):
    """Percentile bootstrap of the median. Deterministic given the seed."""
    if not samples:
        raise ValueError("bootstrap of empty sample list")
    rng = random.Random(seed)
    n = len(samples)
    meds = [median([samples[rng.randrange(n)] for _ in range(n)])
            for _ in range(resamples)]
    return Interval(lo=percentile(meds, lo_pct), med=median(samples),
                    hi=percentile(meds, hi_pct), n=n)


def parametric_band(samples, z=1.6448536269514722, assumed_cv=0.02):
    """Band for small n, conservative on the fast side:
    t_lo = med/(1+z*cv), not med*(1-z*cv)."""
    med = median(samples)
    hi = med * (1.0 + z * assumed_cv)
    lo = med / (1.0 + z * assumed_cv)
    return Interval(lo=lo, med=med, hi=hi, n=len(samples))


def measurement_interval(samples, *, min_repeats=5, resamples=10000, seed=0,
                         lo_pct=5.0, hi_pct=95.0, assumed_cv=0.02,
                         z=1.6448536269514722):
    """Returns (interval, method); method is 'bootstrap' or 'parametric'."""
    if not samples:
        raise ValueError("no samples")
    if len(samples) >= min_repeats:
        return bootstrap_median_ci(samples, resamples, seed, lo_pct, hi_pct), "bootstrap"
    return parametric_band(samples, z=z, assumed_cv=assumed_cv), "parametric"


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_diff: float
    lo: float
    hi: float
    excludes_zero: bool
    method: str
    resamples: int

    def as_dict(self):
        return {
            "n": self.n,
            "mean_diff": round(self.mean_diff, 6),
            "ci95": [round(self.lo, 6), round(self.hi, 6)],
            "excludes_zero": self.excludes_zero,
            "method": self.method,
            "resamples": self.resamples,
        }


def paired_bootstrap_diff(control, candidate, resamples=10000, seed=0,
                          lo_pct=2.5, hi_pct=97.5):
    """Bootstrap CI of mean(candidate - control) over paired trials (P3)."""
    if len(control) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(control)} vs {len(candidate)}")
    if not control:
        raise ValueError("no pairs")
    diffs = [c - b for b, c in zip(control, candidate)]
    n = len(diffs)
    rng = random.Random(seed)
    means = [mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    lo, hi = percentile(means, lo_pct), percentile(means, hi_pct)
    return PairedResult(n=n, mean_diff=mean(diffs), lo=lo, hi=hi,
                        excludes_zero=(lo > 0.0 or hi < 0.0),
                        method="paired_percentile_bootstrap", resamples=resamples)


def paired_speedup_ci(control_ms, candidate_ms, resamples=10000, seed=0):
    """Paired CI of the per-trial ratio control/candidate (1.0 = no change)."""
    if len(control_ms) != len(candidate_ms):
        raise ValueError("paired comparison needs equal lengths")
    ratios = [b / c for b, c in zip(control_ms, candidate_ms) if c > 0]
    if len(ratios) != len(control_ms):
        raise ValueError("candidate times must be > 0")
    n = len(ratios)
    rng = random.Random(seed)
    means = [mean([ratios[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    lo, hi = percentile(means, 2.5), percentile(means, 97.5)
    return PairedResult(n=n, mean_diff=mean(ratios), lo=lo, hi=hi,
                        excludes_zero=(lo > 1.0 or hi < 1.0),
                        method="paired_ratio_bootstrap", resamples=resamples)
