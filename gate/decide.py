"""The headroom gate: case dict in, verdict dict out.

Pure function of (case, rules). No I/O, no clock, no GPU -- so an archived case
replays to the same verdict forever.
"""

from __future__ import annotations

from gate import byte_model
from stats import bootstrap

SCHEMA_VERSION = "1.0.0"

REQUIRED_WORKLOAD = ("kind", "shapes", "dtype", "bytes", "measured_ms",
                     "measurement_source")
REQUIRED_MECHANISM = ("name", "class", "removable_bytes",
                      "addressable_share_claim", "justification")


def validate_case(case):
    """Return human-readable invalidity reasons (empty list = valid)."""
    reasons = []
    for key in ("id", "hypothesis", "workload", "mechanism"):
        if key not in case:
            reasons.append(f"missing field: {key}")
    if reasons:
        return reasons

    wl, mech = case["workload"], case["mechanism"]
    if not isinstance(wl, dict) or not isinstance(mech, dict):
        return ["workload and mechanism must be mappings"]

    for key in REQUIRED_WORKLOAD:
        if key not in wl:
            reasons.append(f"missing field: workload.{key}")
    for key in REQUIRED_MECHANISM:
        if key not in mech:
            reasons.append(f"missing field: mechanism.{key}")
    if reasons:
        return reasons

    if wl["dtype"] not in byte_model.DTYPE_BYTES:
        reasons.append(f"workload.dtype {wl['dtype']!r} is not a known dtype")

    try:
        byte_model.sum_bytes(wl["bytes"])
    except (TypeError, ValueError) as exc:
        reasons.append(f"workload.bytes invalid: {exc}")

    samples = wl["measured_ms"]
    if not isinstance(samples, (list, tuple)) or not samples:
        reasons.append("workload.measured_ms must be a non-empty list")
    elif any((not isinstance(v, (int, float))) or v <= 0 for v in samples):
        reasons.append("workload.measured_ms must contain positive numbers")

    share = mech["addressable_share_claim"]
    if not isinstance(share, (int, float)) or not 0.0 <= float(share) <= 1.0:
        reasons.append("mechanism.addressable_share_claim must be in [0, 1]")

    if not isinstance(mech["removable_bytes"], int) or mech["removable_bytes"] < 0:
        reasons.append("mechanism.removable_bytes must be a non-negative int")

    removable_ms = mech.get("removable_ms")
    if removable_ms is not None:
        if not isinstance(removable_ms, (int, float)) or float(removable_ms) < 0:
            reasons.append("mechanism.removable_ms must be a non-negative number")
        if not mech.get("removable_ms_source"):
            reasons.append("mechanism.removable_ms requires "
                           "mechanism.removable_ms_source")

    empirical_ms = wl.get("empirical_floor_ms")
    if empirical_ms is not None:
        if not isinstance(empirical_ms, (int, float)) or float(empirical_ms) <= 0:
            reasons.append("workload.empirical_floor_ms must be positive")
        if not wl.get("empirical_floor_source"):
            reasons.append("workload.empirical_floor_ms requires "
                           "workload.empirical_floor_source")

    checks = case.get("out_of_scope_checks") or {}
    for flag in ("changes_precision", "changes_model",
                 "changes_acceptance", "changes_output_distribution"):
        if checks.get(flag) is True:
            reasons.append(f"out of scope: {flag}=true")

    return reasons


def evaluate(case, rules, rules_sha256=None, rules_version=None):
    dev, dec, stat, scope = (rules["device"], rules["decision"],
                             rules["statistics"], rules["scope"])

    invalid = validate_case(case)
    if rules_sha256 is not None:
        expected = case.get("rules_sha256")
        if expected is not None and expected != rules_sha256:
            invalid.append(f"rules_sha256 mismatch: case={str(expected)[:12]} "
                           f"rules={rules_sha256[:12]}")
    mech_class = case.get("mechanism", {}).get("class")
    if mech_class and mech_class not in scope["allowed_mechanism_classes"]:
        invalid.append(f"mechanism.class {mech_class!r} not allowed")
    if invalid:
        return _verdict(case, rules, rules_sha256, rules_version,
                        "INVALID", invalid, {})

    wl, mech = case["workload"], case["mechanism"]
    bw = float(dev["hbm_realistic_tbs"])
    tflops = float(dev["bf16_peak_tflops"])

    bytes_total = byte_model.sum_bytes(wl["bytes"])
    floor = byte_model.with_empirical_floor(
        byte_model.roofline(bytes_total, int(wl.get("flops") or 0), bw, tflops),
        wl.get("empirical_floor_ms"),
    )

    samples = [float(v) for v in wl["measured_ms"]]
    interval, method = bootstrap.measurement_interval(
        samples,
        min_repeats=int(dec["min_repeats_go"]),
        resamples=int(stat["bootstrap_resamples"]),
        seed=int(stat["seed"]),
        lo_pct=float(stat["ci_lower_pct"]),
        hi_pct=float(stat["ci_upper_pct"]),
        assumed_cv=float(dec["assumed_cv_single_sample"]),
        z=float(dec["z90"]),
    )
    observed_cv = bootstrap.cv(samples) if len(samples) >= 2 else None

    # A measurement below the floor means the ledger or the reference kernel is
    # wrong; clipping would manufacture negative headroom.
    if interval.med < floor.t_floor_ms * 0.999:
        if floor.floor_source == "empirical":
            reasons = [
                "reference-kernel violation: measured median "
                f"{interval.med:.4f} ms < empirical floor {floor.t_floor_ms:.4f} ms",
                "a kernel cannot beat one that only moves the same bytes; re-measure "
                "the reference kernel, do not clip"]
        else:
            reasons = [
                "roofline violation: measured median "
                f"{interval.med:.4f} ms < floor {floor.t_floor_ms:.4f} ms",
                "the byte/FLOP ledger is inconsistent with this host; fix the ledger, "
                "do not clip"]
        return _verdict(case, rules, rules_sha256, rules_version, "INVALID", reasons, {
            "floor": floor.as_dict(),
            "measurement": _measurement_block(interval, method, observed_cv, wl),
        })

    byte_cap_ms = float(mech["removable_bytes"]) / (bw * 1e12) * 1e3
    ms_cap = float(mech.get("removable_ms") or 0.0)
    cap_ms = max(byte_cap_ms, ms_cap)
    share = float(mech["addressable_share_claim"])

    def at(t):
        hr = max(0.0, t - floor.t_floor_ms)
        claimed = share * hr
        applied = min(claimed, cap_ms, hr)   # accounting caps, headroom caps
        return {
            "t_ms": t,
            "headroom_ms": hr,
            "headroom_fraction": (hr / t) if t > 0 else 0.0,
            "claimed_ms": claimed,
            "applied_ms": applied,
            "capped_by_accounting": applied < claimed - 1e-12,
            "best_case_ms": t - applied,
            "speedup_upper": t / (t - applied) if t > applied else float("inf"),
        }

    mid = at(interval.med)

    # speedup(t) rises while headroom-limited and falls once the cap binds, so
    # the extreme can sit at a kink inside the band. Evaluate kinks explicitly.
    kink_ms = floor.t_floor_ms + cap_ms
    points = {interval.lo, interval.med, interval.hi}
    for value in (floor.t_floor_ms, kink_ms):
        if interval.lo < value < interval.hi:
            points.add(value)
    evaluated = [at(t) for t in sorted(points)]

    speedup_max = max(e["speedup_upper"] for e in evaluated)
    speedup_min = min(e["speedup_upper"] for e in evaluated)
    speedup_point = mid["speedup_upper"]

    trace = []
    if speedup_max < float(dec["veto_below_speedup"]):
        verdict = "NO-GO"
        trace.append(f"veto: most favourable defensible speedup {speedup_max:.4f}x "
                     f"< veto line {dec['veto_below_speedup']}x")
    elif len(samples) < int(dec["min_repeats_go"]):
        verdict = "INCONCLUSIVE"
        trace.append(f"n={len(samples)} < min_repeats_go={dec['min_repeats_go']}: "
                     "a green light needs real repeats, a veto does not")
    elif speedup_min >= float(dec["go_min_speedup"]):
        if dec.get("require_empirical_floor_for_go", True) and \
                floor.empirical_floor_ms is None:
            verdict = "INCONCLUSIVE"
            trace.append("no same-bytes empirical floor: an analytic floor may veto "
                         "but may never grant a GO")
        elif mid["headroom_fraction"] < float(dec["min_headroom_fraction"]):
            verdict = "INCONCLUSIVE"
            trace.append(f"headroom fraction {mid['headroom_fraction']:.4f} < "
                         f"min {dec['min_headroom_fraction']}")
        elif observed_cv is not None and observed_cv > float(dec["max_cv_for_go"]):
            verdict = "INCONCLUSIVE"
            trace.append(f"cv {observed_cv:.4f} > max_cv_for_go {dec['max_cv_for_go']}")
        else:
            verdict = "GO"
            trace.append(f"conservative upper bound {speedup_min:.4f}x >= go line "
                         f"{dec['go_min_speedup']}x")
    else:
        verdict = "INCONCLUSIVE"
        trace.append(f"conservative bound {speedup_min:.4f}x < go line "
                     f"{dec['go_min_speedup']}x, generous bound {speedup_max:.4f}x "
                     f">= veto line: more information needed")

    reasons = list(trace)
    if floor.floor_source == "empirical":
        reasons.append(f"floor raised from analytic {floor.analytic_floor_ms:.4f} ms "
                       f"to {floor.t_floor_ms:.4f} ms by the same-bytes reference")
    if len(samples) < int(dec["min_repeats_go"]):
        reasons.append(f"measurement band is {method}, n={len(samples)}, "
                       f"source={wl['measurement_source']}")
    if mid["capped_by_accounting"]:
        reasons.append(f"claim capped by accounting: claimed {mid['claimed_ms']:.4f} ms "
                       f"> allowed {cap_ms:.4f} ms (bytes {byte_cap_ms:.4f} / "
                       f"time {ms_cap:.4f})")

    checks = {
        "n_samples": len(samples),
        "n_ge_min_repeats_go": len(samples) >= int(dec["min_repeats_go"]),
        "cv": None if observed_cv is None else round(observed_cv, 6),
        "cv_le_max": observed_cv is None or observed_cv <= float(dec["max_cv_for_go"]),
        "headroom_fraction_ge_min": mid["headroom_fraction"] >= float(
            dec["min_headroom_fraction"]),
        "empirical_floor_present": floor.empirical_floor_ms is not None,
        "floor_source": floor.floor_source,
        "accounting_caps_claim": mid["capped_by_accounting"],
        "out_of_scope_clean": True,
    }

    return _verdict(case, rules, rules_sha256, rules_version, verdict, reasons, {
        "checks": checks,
        "floor": floor.as_dict(),
        "measurement": _measurement_block(interval, method, observed_cv, wl),
        "headroom": {
            "median_ms": mid["headroom_ms"],
            "fraction": mid["headroom_fraction"],
            "at_fast_end_ms": at(interval.lo)["headroom_ms"],
            "at_slow_end_ms": at(interval.hi)["headroom_ms"],
        },
        "addressable": {
            "removable_bytes": int(mech["removable_bytes"]),
            "byte_cap_ms": byte_cap_ms,
            "removable_ms": ms_cap,
            "removable_ms_source": mech.get("removable_ms_source"),
            "cap_ms": cap_ms,
            "claimed_share": share,
            "claimed_ms_median": mid["claimed_ms"],
            "applied_ms_median": mid["applied_ms"],
            "effective_share_median": (mid["applied_ms"] / mid["headroom_ms"])
                                      if mid["headroom_ms"] > 0 else 0.0,
        },
        "prediction": {
            "speedup_upper_min": speedup_min,
            "speedup_upper_point": speedup_point,
            "speedup_upper_max": speedup_max,
            "best_case_ms_min": min(e["best_case_ms"] for e in evaluated),
            "best_case_ms_point": mid["best_case_ms"],
            "best_case_ms_max": max(e["best_case_ms"] for e in evaluated),
            "interval_method": method,
        },
        "decision_trace": trace,
    })


def _measurement_block(interval, method, observed_cv, wl):
    return {
        "n": interval.n,
        "median_ms": interval.med,
        "lo_ms": interval.lo,
        "hi_ms": interval.hi,
        "cv": None if observed_cv is None else round(observed_cv, 6),
        "band_method": method,
        "source": wl["measurement_source"],
        "raw_samples_ms": list(wl["measured_ms"]),
    }


def _verdict(case, rules, rules_sha256, rules_version, verdict, reasons, detail):
    doc = {
        "kind": "hgk_gate_verdict",
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("id", "<unknown>"),
        "title": case.get("title", ""),
        "verdict": verdict,
        "reasons": reasons,
        "rules_version": rules_version or rules.get("version"),
        "rules_sha256": rules_sha256,
        "hypothesis": case.get("hypothesis", ""),
        "mechanism": {
            "name": case.get("mechanism", {}).get("name"),
            "class": case.get("mechanism", {}).get("class"),
            "justification": case.get("mechanism", {}).get("justification"),
        },
    }
    doc.update(detail)
    return doc


def summarize(doc):
    p = doc.get("prediction", {})
    if p:
        return (f"{doc['case_id']}: {doc['verdict']} · upper bound "
                f"[{p['speedup_upper_min']:.3f}, {p['speedup_upper_max']:.3f}]x "
                f"(point {p['speedup_upper_point']:.3f}x)")
    return f"{doc['case_id']}: {doc['verdict']} · {'; '.join(doc.get('reasons', [])[:2])}"
