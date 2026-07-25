# Changelog

This documents every change made in this pass, and why, so results run
before vs. after this revision aren't silently mixed in the paper.

## Correctness fixes (would have blocked or invalidated a run)

### 1. AgentDojo task extraction never actually worked

`src/recertagent/adapters/agentdojo.py` extracted task prompts by checking
lowercase attributes (`prompt`, `instruction`, `user_prompt`, `text`) on
AgentDojo task objects. Verified against the actual installed
`agentdojo==0.1.35` package: `agentdojo.base_tasks.BaseUserTask` exposes the
prompt on the **uppercase** class attribute `PROMPT` (and injection tasks use
`GOAL`). None of the lowercase names ever matched, so `load_agentdojo_tasks`
always returned zero rows and `prepare` always raised `RuntimeError: Could
not extract AgentDojo tasks...`. This was the pipeline's single point of
total failure — nothing downstream could run.

Fix: check `PROMPT`/`GOAL` first (the real, current API), keep the
lowercase names as a defensive fallback for a future AgentDojo API change,
and make the suite/version-resolution loop retry more benchmark-version
strings so a partial AgentDojo upgrade degrades gracefully instead of
raising immediately.

### 2. `StrEnum` breaks on Python 3.10

`src/recertagent/schemas.py` used `from enum import StrEnum`, which only
exists on Python ≥ 3.11. `pyproject.toml` declares `requires-python =
">=3.10,<3.13"`, and Kaggle notebook base images have run 3.10 at various
points — on such an environment, the very first import of the package would
raise `ImportError` before any code ran.

Fix: `class Label(str, Enum)`. Identical string-equality and JSON
serialization behavior, works on 3.10 and 3.11+.

### 3. `clean_utility_probe` was stored but never evaluated

`BenchmarkCase.clean_utility_probe` and `PublicCase.clean_utility_probe`
existed in the schema and were populated by `remediation/builder.py`, but no
code anywhere — not `runner.py`, not `evaluation/oracle.py`, not
`evaluation/metrics.py` — ever read the field to actually run it. Meanwhile
`paper/experiment_protocol.md` lists **"utility retained after remediation"**
as a required secondary endpoint. Over-cleaning (the `overclean` remediation
condition) is specifically designed to test this failure mode, and the
pipeline had no way to detect it.

Fix: `BehavioralOracle.run_utility_probe()` runs the probe once per case
(cached across certifiers, since utility retention is a property of the
remediated snapshot, not of any certifier's decision), `runner.py` attaches
the result to every output row as `utility_outcome`, and
`evaluation/metrics.py::compute_metrics` reports `utility_retention_rate`.
Older `results/predictions.jsonl` files without `utility_outcome` are
handled gracefully (`utility_retention_rate` reports as `null` rather than
crashing `summarize`).

## Robustness fixes (would have caused unreliable or crashed Kaggle runs)

### 4. 4-bit quantization on non-Turing GPUs

Kaggle assigns GPUs from a heterogeneous pool, including Tesla P100
(compute capability 6.0). bitsandbytes 4-bit kernels are unreliable or
unsupported below compute capability 7.0. `models/qwen.py` requested 4-bit
unconditionally whenever `load_in_4bit: true` and any CUDA device was
present.

Fix: check `torch.cuda.get_device_capability()` and only request 4-bit on
capability ≥ 7.0; otherwise load in fp16 with a warning. The model load
itself is additionally wrapped in try/except so a bitsandbytes failure for
any other reason falls back to an fp16 reload instead of crashing the run.

### 5. No reproducibility seeding

`configs/kaggle_smoke.yaml` and `configs/kaggle_full.yaml` both declare
`seed: 42`, but nothing in `runner.py` ever read it — `random`, `numpy`, and
`torch` were all left unseeded.

Fix: `runner.py` now seeds `random`, `numpy`, `torch`, and
`torch.cuda` from `cfg["seed"]` at the start of `run()`.

## Paper-rigor additions (required by the protocol, previously unimplemented)

`paper/experiment_protocol.md`'s "Required final-paper extensions" section
listed several analyses that had no supporting code:

- **"Split by AgentDojo suite, never random-only split."** — Added
  `evaluation/metrics.py::compute_metrics_by` + `suite_of`, wired into
  `cli.py summarize` as `by_suite` in `results/summary.json`.
- **"Falsification by remediation condition"** (a listed secondary
  endpoint) — same `compute_metrics_by` helper, wired in as
  `by_remediation`.
- **"Compare probe budgets 1, 3, and 5."** — Added
  `evaluation/metrics.py::compute_budget_ablation`, which recomputes FSCR
  at each budget from the probes already collected at the run's actual
  budget (no re-run / re-generation needed). Wired in as
  `probe_budget_ablation`.
- **"Manually audit at least 100 oracle decisions."** — Added a new
  `recertagent audit-sample` CLI command that exports a random sample of
  (case, certifier, probe) rows to `results/audit_sample.csv` with the
  oracle's own judgment plus blank `human_label`/`human_notes` columns for
  a reviewer to fill in.

The remaining protocol extensions ("hold out at least one remediation
family during threshold selection", "run at least two agent models") are
experiment-design decisions best made when running the full study rather
than hard-coded defaults, and are left to the researcher running the paper
experiments; the codebase's certifier thresholds (`abstain_low`,
`abstain_high` in the config files) and the single-model setup can be
swapped without further code changes.

## Documentation

- `README.md` and `KAGGLE_CELLS.md` rewritten with a verified, step-by-step
  Kaggle setup (dataset upload, accelerator/internet settings, exact cell
  sequence, expected install time, and what's in `results/summary.json`).
- Added `tests/test_agentdojo_adapter.py` (regression test for fix #1) and
  extended `tests/test_metrics.py` with tests for `utility_retention_rate`,
  `compute_metrics_by`, and `compute_budget_ablation`.
