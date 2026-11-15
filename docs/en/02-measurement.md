# 02 — Measurement protocol and the detached-run contract

A verdict can only be as good as its measurement. This section fixes two things:
how time is taken, and how a long run stays decidable after the terminal, SSH
session, or laptop is gone.

## 1. Timing protocol (shared by every script)

- **CUDA events around a single launch** — not `time.time()`, not a loop average.
- **Flush L2 before each repeat** (read a 256 MB buffer); otherwise the second
  operator hits the cache the first one left warm.
- **Warmup and repeats are recorded separately**; defaults are 10 warmup and 20
  repeats.
- **Clocks recorded, never pinned** (`clocks: recorded-only`), so a lock is never
  mistaken for an algorithmic gain.
- **A 1 % noise floor**: anything smaller is "not measurable" here and must not be
  reported as a gain, in either direction.
- **Keep the raw samples.** `measured_ms` is a list, not a median. Bands, CV and
  P1 are all recomputed from samples.

An evidence file must answer: which shapes and dtype, how many warmup/repeats,
which timer, what median, and what a same-bytes reference kernel costs.
`measurement_source` in `case.yaml` points at that file and field.

## 2. Why `case.yaml` is append-only

Pre-registration only means something if the hypothesis cannot be rewritten after
the numbers arrive:

- `measured_ms`, `removable_*` and `addressable_share_claim` are not edited after a
  verdict;
- hindsight goes into `post_measurement_note` (the attn and gemv cases both keep
  their falsified original hypotheses);
- new verdicts are appended to `ledger.jsonl`; old ones stay as history.

## 3. Detached-run contract

```
QUEUED ──► RUNNING ──► SUCCEEDED | FAILED | CANCELLED
```

| Script | Role |
|---|---|
| `scripts/detach.sh <job> <cmd...>` | starts the supervisor in a new session via `setsid`, stdin closed, `set -m` so SIGINT stays trappable; prints the run dir |
| `scripts/job_supervisor.sh` | launches the child, forwards signals, writes `status.json`, heartbeats every 5 s |
| `scripts/check_run.sh <run_dir>` | status + heartbeat age + log tails; exit 0/1/2 for done/failed/alive |
| `scripts/stop_run.sh <run_dir>` | SIGINT, wait 30 s; then SIGTERM, wait 10 s; **never kill -9** |
| `scripts/resume.sh <run_dir>` | replays argv from `command.json`, `resume_count + 1`; refuses RUNNING/QUEUED |
| `scripts/lib/runmeta.py` | `init / status / heartbeat / checkpoint`, all writes atomic |

`status.json` is the only truth. Success is never inferred from "the terminal
came back".

A run directory should contain `command.txt`, `command.json`, `environment.json`
(host, Python, git commit, visible GPUs), `input_manifest.json` (sha256 per input),
`status.json`, `heartbeat.json`, `checkpoint.json`, `stdout.log`, `stderr.log`, and
`artifacts/`.

## 4. Platform differences (measured, not assumed)

- On Linux `setsid` creates an independent session and `ps -o sid=` works.
- macOS has no `setsid`; the scripts fall back to `nohup`. BSD `ps` only knows
  `sess`, so the supervisor validates that the session field is numeric and records
  0 otherwise. Local macOS use is a contract smoke test only.
- A naive `cmd &` makes the child inherit "SIGINT ignored", which is exactly why a
  graceful stop never lands. The supervisor launches the child under `set -m`, and
  `stop_run.sh`'s SIGINT then reaches the operator.

## 5. Current boundary

The GPU-side scripts are not in this repo yet (`measure_baseline.py`,
`measure_candidate.py`, `sealed_eval.py`, `kernel.py`, `measure_dispatch.py`,
`e2e_qwen3_8b.py`, `analyze_e2e.py`, `scripts/compare.py`,
`p3/measure_pool.py`). They are tracked as `blocked`, not described as done. The
five archived verdicts are therefore replayed from recorded samples, not
re-measured here.
