# 03 — Evaluation: two claims, five replays, one honest deviation

## 1. Two claims

**P1 — the ratchet (mechanically checked).** No realised speedup may exceed the
predicted upper bound that its verdict stated. `gate/cli.py --verify-ledger`
rebuilds the latest gate prediction per case from the ledger and compares every
`realised` event against it; a violation exits 4.

Strictly, P1 presupposes P2's ordering discipline: predict first, measure second,
never the other way round.

**P2 — disjointness (structural).** Evaluation operators never appear in measured
cases. The evaluation set lives in a sealed list outside
`protocol/gate_rules.yaml`; `sealed_eval.py` runs only that list, and `cases/` may
not contain the same operator. The GPU evaluation scripts are not landed yet, so
P2 is currently declared, not executed.

## 2. Local replay of the five archived cases

```bash
python3 -m gate.cli --all
python3 -m pytest tests/test_gate.py -q
```

| Case | Verdict | Upper bound 5–95 % | Point | Floor source |
|---|---|---|---|---|
| `swiglu-fusion-b4096-h7168` | **GO** | 4.993443 – 5.004346 | 4.999138 | analytic 0.053382 (empirical 0.053104 is lower, ignored) |
| `attn-decode-longctx` | NO-GO | 1.021620 – 1.035426 | 1.035426 | empirical 0.061424 (analytic 0.040675 is too low) |
| `gemv-m8-bf16` | NO-GO | 1.021552 – 1.048246 | 1.029720 | empirical 0.080317 (analytic 0.063550) |
| `selector-dflash2` | NO-GO | 1.004749 – 1.005068 | 1.004906 | analytic (single sample, parametric band) |
| `verify-fusion-megaspec` | NO-GO | 1.000000 – 1.003320 | 1.003257 | analytic (single sample, parametric band) |

These match the archived verdicts bit for bit. They are the repo's regression
baseline: `tests/test_gate.py::test_archived_verdicts_replay` asserts all 15 values
to 1e-4.

How to read the four NO-GOs:

- `attn-decode-longctx`: the analytic roofline shows 1.56× of "headroom", but a
  same-bytes sum already takes 0.0614 ms against 0.0637 ms measured — 3.4 % real.
  The hypothesis was falsified by its own measurement and is preserved verbatim.
- `gemv-m8-bf16`: 0.1 % headroom, below the 2 % minimum, and zero removable bytes.
- Both end-to-end cases: the mechanism covers 0.3–0.5 % of a step and is
  latency-bound, so the byte ledger binds tighter than the time ledger.

## 3. The one deviation from the archive: the rules hash

The archive records `protocol/gate_rules.yaml` at sha256 `6280dcff…`. The file as
re-created here hashes to
`966715a7dfe9df58e71175719fa3f62226d474e368adc42c177fe7d17416bea2`.

Cause: whitespace and line endings could not be reproduced byte-for-byte from the
document copy, so the original bytes are unrecoverable. The fix follows spec §4.1 —
**only the identity hash changes, no value does**:

- all five `case.yaml` files re-pin `rules_sha256` to `966715a7…`;
- thresholds, device constants, mechanism claims and `measured_ms` are untouched;
- verdicts and all 15 numbers replay identically (table above);
- `tests/test_gate.py` asserts both "current hash == pinned value" and "every case
  pins the same hash", so any future rules edit fails immediately.

This is the only inconsistency a reader needs to know about, and it is written
here rather than buried in a commit message.

## 4. Test coverage

`python3 -m pytest tests -q` — 71 tests:

| File | Coverage |
|---|---|
| `tests/test_gate.py` | five archived replays, in-band kink, out-of-scope/missing-field/hash/mechanism-class invalidation, single-sample path |
| `tests/test_byte_model.py` | byte ledger, roofline regimes, empirical floor raises only, negative cases |
| `tests/test_bootstrap.py` | median/percentile, bootstrap determinism, conservative parametric end, paired tests |
| `tests/test_ledger.py` | append-only write, corrupt-line location, state rebuild, P1 violation detection |
| `tests/test_cli.py` | single/all, `--record`, `--verify-ledger` exit codes, INVALID exit 3 |
| `tests/test_runmeta.py` | init/status/heartbeat/checkpoint, atomic writes, field preservation |
| `tests/test_p3_select.py` | R1–R4, joint-measurement requirement, policy freeze and tamper detection |
| `tests/test_probes.py` | the probe plan prints on a GPU-less host and never imports torch |

## 5. Boundary of reproducibility

Locally you can reproduce the *verdict*, not the *measurement*. Verdicts depend on
the samples in `case.yaml`, which came from the H20 host; every sample used is
embedded in the case file and the original raw archive is private, not
redistributed. Re-deriving samples requires landing the `blocked` GPU scripts and
following [02 — Measurement](./02-measurement.md).
