# P3 — pre-registered selection policy (frozen 2026-11-14)

This file is frozen **before** the pool is measured. Its sha256 is written to
`p3/policy.sha256` and embedded in every `selection.json`, so a selection made
under a different policy is detectable. Editing this file after the freeze
invalidates the selection; bump `policy_version` and re-freeze instead.

Scope: P3 is an *executed* synthetic sweep (four `_rsi` operators, bare-metal
Triton/PTX, fixed random data, no model change). It is closed-loop, not a
benchmark: its target is the **pool's** best-low-bits selection, never a
paper leaderboard.

## 1. Pool

- Operators: `rmsnorm`, `softmax`, `layernorm`, `gemv`.
- Per operator: shapes / seeds of the pool, recorded in `p3/pool.json`
  (`task_id`, `seeds`, `fp16_val_acc`, `trials[]`).
- Every trial records `logical`, `bits` per component, `seed`, `fp16_val_acc`,
  `diverged`, `tok_per_s`. `fp16_val_acc` is always evaluated in FP16 on the
  same synthetic task, so a lower-bits arm is compared to the identical
  numerical target.
- `p3/pool.json` is written by `p3/measure_pool.py`; it must contain the raw
  trials, not a summary.

## 2. Rules (applied in this order)

**R1 — divergence veto.** A config arm with any diverged trial is excluded. If
the FP16 control arm of a task diverged, the whole task is excluded.

**R2 — accuracy.** The best config of a task is the arm with the highest mean
`fp16_val_acc` over its seeds subject to

    mean_acc(config) >= fp16_ref_acc(task) - 0.0015

Ties break by lower total bits, then by config id.

**R3 — task selection.** A task is selected only if

    best_mean >= mean(fp16_ref_acc over all tasks) - 0.0035
    best_mean >  fp16_acc_caveat[task] + 0.002        (when a caveat exists)

A caveat is a recorded pre-existing gap of the FP16 path on that task; without
R3.b a broken FP16 reference would make every arm "good enough".

**R4 — family.** Group selected tasks by `mechanism_class`. Adopt the class
with the most selected tasks provided it has at least 3; ties break by larger
median gain, then by class name. If no class reaches 3, fall back to task-level
selection and record `"family": null` (the pool is then reported as thin, not
as a success).

**R5 — joint impact.** Any headline number for a class combination must be a
*measured joint* run of exactly that combination on the same host. Summing
per-component impacts is allowed only as a planning estimate and must be
labelled as such; it may not be reported as the joint value.

**R6 — claim.** P1 over P3 may only cite mechanism classes that P3 selected
(`p1_mechanism_classes`). A class selected only by a paper is not citable.

## 3. Output

`p3/selection.json` (via `python3 -m p3.select`):

    {kind, policy_version, policy_sha256, pool_sha256, generated_at,
     fp16_ref_mean, tasks[], family, selected_mechanisms[], joint{}, p1_mechanism_classes[]}

Every task row carries its own `reasons[]`, including the rejected ones, so a
reader can audit the selection without rerunning it.

## 4. What P3 may not do

- Change precision/model/acceptance of a P1/P2 case.
- Reuse a P1 case's measurements as pool evidence.
- Run on a different GPU class than the pool's recorded `device` and still be
  combined with P1 numbers.
