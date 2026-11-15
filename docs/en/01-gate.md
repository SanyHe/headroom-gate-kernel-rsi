# 01 — The gate: price the ceiling before touching a kernel

The gate is a pure function in `gate/decide.py`: `(case, rules) -> verdict`. It
reads no clock, touches no GPU, and writes no files, so a measured case replays
to the same verdict forever.

## 1. Input: one pre-registered case

`cases/<id>/case.yaml` is written before measurement:

| Field | Meaning |
|---|---|
| `workload.bytes` | compulsory traffic, split into `weights / activations / output` |
| `workload.flops` | FLOPs actually executed, not the theoretical count |
| `workload.measured_ms` | every timed sample; medians and bands derive from these |
| `workload.empirical_floor_ms` | same-bytes reference-kernel time (optional, required for GO) |
| `mechanism.removable_bytes` | bytes the mechanism claims to delete |
| `mechanism.removable_ms` | time it claims to delete, with a source |
| `mechanism.addressable_share_claim` | fraction of headroom it covers, in [0, 1] |
| `out_of_scope_checks` | precision/model/acceptance/output distribution; any `true` makes the case INVALID |

## 2. The floor: the larger of two hard constraints

```
t_mem     = bytes / (BW_eff · 1e12) · 1e3
t_compute = flops / (FLOP_eff · 1e12) · 1e3
floor     = max(t_mem, t_compute)
```

`BW_eff = 1.613 TB/s` is the measured operator-scale sustained copy bandwidth on
the AutoDL RTX 6000D host, not a spec-sheet number; `FLOP_eff = 136.8 TFLOPS` is
the measured BF16 dense-GEMM peak. Both live in `protocol/gate_rules.yaml`,
frozen by hash.

**An empirical floor may only raise.** Read-dominated operators cannot reach copy
bandwidth below ~300 MB (measured read ceiling there is 0.95–2.57 TB/s), so the
analytic floor invents headroom that does not exist. A case may therefore record
the time a reference kernel needs to move the *same* bytes, and the gate uses
`max(analytic, empirical)`. Because the floor only rises, P1 (predicted ≥
realised) stays safe by construction.

## 3. Addressable: three-way clamp

```
headroom  = max(0, t_measured - floor)
claimed   = share · headroom
byte_cap  = removable_bytes / (BW_eff · 1e12) · 1e3
applied   = min(claimed, byte_cap, removable_ms, headroom)
```

The byte ledger is physics: no mechanism removes more time than the bytes it
touches. The time ledger is the second wall: a latency-bound mechanism (a 0.11 ms
selector) can only yield 0.11 ms. Whenever `applied < claimed` the verdict records
`capped_by_accounting` instead of quietly taking the claim.

## 4. The band: kinks must be evaluated

With n ≥ 5 the gate uses a 5–95 % percentile bootstrap (10 000 resamples, fixed
seed). Below that it degrades to a parametric band around the median, conservative
on the fast side (`med / (1 + z·cv)`, not `med·(1 - z·cv)`).

`speedup(t) = t / (t - applied(t))` rises while headroom-limited and falls once a
cap binds, so the maximum can sit at the interior kink `floor + cap_ms`, not at
either endpoint. The gate adds every kink inside the band to the evaluation set.
The swiglu case is exactly this: endpoint-only evaluation gives ~4.99 and misses
the real 5.004.

## 5. Three verdicts

| Condition (in order) | Verdict |
|---|---|
| generous end < `veto_below_speedup` (1.15×) | `NO-GO` |
| n < `min_repeats_go` (5) | `INCONCLUSIVE` |
| conservative end ≥ `go_min_speedup` (1.30×) | continue |
| ↳ no same-bytes empirical floor | `INCONCLUSIVE` |
| ↳ headroom fraction < 2 % | `INCONCLUSIVE` |
| ↳ CV > 2 % | `INCONCLUSIVE` |
| otherwise | `GO` |
| none of the above | `INCONCLUSIVE` |

**An analytic floor may veto, never grant.** That is the core of v1.1.0: analytic
floors are systematically pessimistic for read-bound work, so only a same-bytes
reference kernel proves the headroom is real.

## 6. `INVALID` means the data is broken, not that the idea lost

- `measured_median < floor` (0.1 % tolerance): the ledger or the reference kernel
  is wrong. The gate refuses to clip, because clipping manufactures positive
  headroom and turns an error into a green light.
- Missing fields, unknown dtype, out-of-range share, disallowed mechanism class,
  or a flipped out-of-scope flag.
- `rules_sha256` does not match the current rules file.

## 7. Code

- Thresholds: [gate_rules.yaml](../protocol/gate_rules.yaml)
- Ledger and roofline: [byte_model.py](../gate/byte_model.py)
- Decision: [decide.py](../gate/decide.py)
- Bands and paired tests: [bootstrap.py](../stats/bootstrap.py)
