# AGENTS.md — Headroom-Gate Kernel RSI

> This is the harness entry file. Read it first at the start of every session.
> It defines how the agent works on this repo, so the work stays reliable across
> sessions.

## 1. Startup Workflow (before writing any code)

1. Read [`private/trae-progress.md`](./private/trae-progress.md) — **Current
   Verified State** and **Next best action**.
2. Read [`private/feature_list.json`](./private/feature_list.json) — find the one
   feature that is `in_progress`, or the highest-priority `not_started` feature.
   Only **one** may be `in_progress`.
3. Read the spec section for that feature in `private/REPRODUCE.md`, plus
   [`docs/01-gate.md`](./docs/01-gate.md) and [`docs/03-eval.md`](./docs/03-eval.md).
4. Run `./init.sh`. It replays the archived verdicts and runs the CPU suite. If it
   fails, fix the baseline before anything else.

> `private/` is gitignored and never published. A fresh clone has no harness
> state; recreate it from the release notes instead of guessing.

## 2. Working Rules

- **One feature at a time.** Never start a feature while another is `in_progress`.
- **Evidence before `passing`.** A feature is `passing` only when its verification
  commands were run, their output is recorded in `evidence`, and the full CPU
  suite is still green.
- **No GPU locally.** Timing runs happen only on the GPU host. If a feature needs
  a GPU, mark it `blocked` with the reason — never `passing`.
- **The gate is frozen by hash.** `protocol/gate_rules.yaml` may not be edited to
  make a case pass. A changed file invalidates every verdict under it; bump
  `version`, re-freeze, and re-register cases deliberately.
- **Floors may only go up.** An empirical floor is folded in with `max(...)`.
  Never lower a floor, never clip a measurement that falls below it — that case
  is `INVALID` by construction.
- **Cases are append-only.** Do not rewrite `measured_ms` or a mechanism claim
  after a verdict. Record `post_measurement_note` and a new ledger event instead.
- **No silent threshold drift.** Thresholds come from `gate_rules.yaml`; missing
  keys are hard errors, not defaults.
- **Scope discipline.** Implement the selected feature only. No speculative
  modules for later phases.
- **Code comments in English.** No emojis in code or docs.
- **No secrets, no private hostnames, no tokens in the repo.**

## 3. Definition of Done

1. `private/feature_list.json` says `passing` with concrete evidence: command,
   output summary, file paths, and the sha256 of any generated artifact.
2. `private/trae-progress.md` has a session record: Goal, Completed, Verification,
   Evidence, Known risks, Next best action.
3. `python3 -m pytest tests/ -q` passes.
4. `python3 -m gate.cli --all` replays the archived verdicts unchanged.
5. Nothing half-finished is left unrecorded.

## 4. Commands & Paths

| Purpose | Command |
|---|---|
| Bootstrap + verify (no GPU) | `./init.sh` |
| Evaluate all cases | `python3 -m gate.cli --all` |
| Evaluate one case | `python3 -m gate.cli cases/swiglu-fusion-b4096-h7168` |
| Record verdicts to the ledger | `python3 -m gate.cli --all --record` |
| P1 ratchet check | `python3 -m gate.cli --verify-ledger` |
| Tests | `python3 -m pytest tests/ -q` |
| Device probe plan (GPU needed to execute) | `python3 -m gate.probes` |
| Select from a measured pool | `python3 -m p3.select --pool p3/pool.json` |
| Detached run | `bash scripts/detach.sh <job> <cmd...>` |
| Run status / stop / resume | `bash scripts/check_run.sh <run_dir>` · `bash scripts/stop_run.sh <run_dir>` · `bash scripts/resume.sh <run_dir>` |

Key paths:
- Frozen thresholds: [`protocol/gate_rules.yaml`](./protocol/gate_rules.yaml)
- Byte/FLOP ledger: [`gate/byte_model.py`](./gate/byte_model.py)
- Decision logic: [`gate/decide.py`](./gate/decide.py)
- Statistics: [`stats/bootstrap.py`](./stats/bootstrap.py)
- Append-only ledger: [`ledger.jsonl`](./ledger.jsonl)
- Cases: [`cases/`](./cases/)
- P3 policy: [`p3/policy.md`](./p3/policy.md)

## 5. Compute

All timing runs happen on third-party cloud GPU hardware. Published
measurements come from the AutoDL RTX 6000D host under
`protocol/gate_rules.yaml` (version 2.0.0). The five cases in `cases/` are
pre-registered templates with no measurement data; they must be re-measured on
third-party cloud hardware before they can produce a verdict. Local work is
limited to: the gate, the ledger, statistics, case registration, the run-script
contract, the P3 selector, tests, and docs. Anything that needs `torch.cuda` is
`blocked` until a GPU session.

## 6. End-of-Session Checklist

- [ ] `./init.sh` still passes.
- [ ] `private/feature_list.json` reflects reality (no false `passing`).
- [ ] `private/trae-progress.md` updated with the session record.
- [ ] Any blocker is written into the feature's `notes` and the progress file.
- [ ] No `case.yaml` edited after its verdict was recorded.
