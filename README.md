# ReCertAgent

**Attack-blind safety certification of remediated stateful LLM agents**

ReCertAgent evaluates whether a verifier can issue `SAFE`, `UNSAFE`, or
`ABSTAIN` for a post-remediation agent state without seeing the original attack
payload, trigger, attack family, corrupted record identifier, or remediation
oracle labels.

The primary endpoint is **False-Safe Certification Rate (FSCR)**:

> among states certified SAFE, the fraction subsequently falsified by held-out
> behavioral probes.

## Real components

- AgentDojo 0.1.35 for realistic tool-agent tasks.
- NotInject (`leolee99/NotInject`) for benign attack-like prompts.
- PIGuard (`leolee99/PIGuard`) as a real detector baseline.
- Qwen2.5-3B-Instruct as an open-weight certifier and behavioral agent.
- BGE-small-en-v1.5 for semantic inspection.

All five are public, ungated HuggingFace/PyPI resources — no API keys or
HF tokens are required anywhere in this pipeline.

## What changed in this pass (see `CHANGELOG.md` for full detail)

This revision fixes a bug that made `prepare` fail on every run (AgentDojo's
real task-prompt attribute is `PROMPT`, not any lowercase name — see
`CHANGELOG.md`), a Python-3.10/Kaggle portability bug (`StrEnum` requires
3.11+), adds a GPU-capability check so 4-bit loading degrades gracefully on
older Kaggle GPUs (e.g. P100), seeds all RNGs for reproducibility, and
implements several endpoints the frozen protocol requires but the original
code never computed (utility retention, per-suite/per-remediation
breakdowns, probe-budget ablation, an audit-sample export). Read
`CHANGELOG.md` before citing results from this repo — it documents exactly
which numbers changed and why.

## Running on Kaggle

### 1. Upload the repo as a Kaggle Dataset

Kaggle notebooks can't read arbitrary local zips directly — package the repo
as a **private Kaggle Dataset** once:

1. Kaggle → **Create** → **New Dataset** → upload `ReCertAgent.zip` as-is
   (don't unzip it first). Name it `recertagent`.
2. Create a new Notebook, and under **Add Input**, attach the `recertagent`
   dataset. It will be mounted at `/kaggle/input/recertagent/ReCertAgent.zip`.

### 2. Notebook settings (right sidebar)

- **Accelerator**: GPU T4 x2 (or any single T4/L4). Works on P100 too — 4-bit
  quantization will auto-fall-back to fp16 there (see `CHANGELOG.md`), which
  is slower but correct.
- **Internet**: **On**. Required to `pip install` and to pull AgentDojo task
  suites, NotInject, PIGuard, Qwen2.5-3B-Instruct, and BGE-small from the
  network on first run.
- **Persistence**: not required; each cell below is self-contained.

### 3. Cells

See `KAGGLE_CELLS.md` for the exact cell-by-cell notebook. Short version:

```bash
!unzip -q /kaggle/input/recertagent/ReCertAgent.zip -d /kaggle/working/
%cd /kaggle/working/ReCertAgent
!pip install -q -e ".[dev]"
```

```bash
!pytest -q
```

```bash
!python -m recertagent.cli prepare --limit-agentdojo 20
```

```bash
!python -m recertagent.cli run --config configs/kaggle_smoke.yaml
```

```bash
!python -m recertagent.cli summarize --input results/predictions.jsonl
```

After the smoke run succeeds end-to-end:

```bash
!python -m recertagent.cli prepare --limit-agentdojo 97
!python -m recertagent.cli run --config configs/kaggle_full.yaml
!python -m recertagent.cli summarize --input results/predictions.jsonl
```

Optional, for the paper's manual-audit requirement:

```bash
!python -m recertagent.cli audit-sample --input results/predictions.jsonl --n 100
```

### Expected timing / resources

- `pip install -e ".[dev]"` installs AgentDojo's full dependency tree
  (it unconditionally requires `anthropic`, `cohere`, `google-genai`,
  `langchain`, `openai`, none of which this pipeline actually calls, but pip
  installs them anyway since they're plain `Requires-Dist`, not extras) —
  budget **3-6 minutes** for this cell, not a hang.
- Qwen2.5-3B-Instruct in 4-bit is ~2.5 GB of VRAM; PIGuard and BGE-small are
  under 1 GB combined. All three fit comfortably on a single T4 (16 GB).
- The smoke config (`kaggle_smoke.yaml`, 24 cases) finishes in a few minutes
  on a T4. The full config (`kaggle_full.yaml`, 582 cases = 97 tasks × 6
  remediation variants) is the real experiment and takes longer in
  proportion — run the smoke config first to catch environment problems
  cheaply.

### Saving outputs

```python
import shutil
shutil.make_archive("/kaggle/working/recertagent_results", "zip", "results")
```

Then download `recertagent_results.zip` from the Kaggle output pane.

## Scientific boundary

The source tasks and prompts are real. ReCertAgent then creates controlled,
auditable post-remediation state variants. These variants are the proposed
benchmark construction; they are not claimed reproductions of recovery systems
inside AgentDojo.

The certifier receives a `PublicCase`. Hidden attack information is stored only
in `BenchmarkCase.hidden` and is used solely by the held-out evaluator.

## Paper-safe contribution statement

> We formulate attack-blind, budgeted post-remediation certification as a
> three-way decision problem and evaluate false-safe claims under held-out
> behavioral falsification.

Do not claim the first recovery, rollback, memory-poisoning, or residual-state
framework.

## Outputs and what's in `results/summary.json`

`summarize` now reports, per certifier:

- The original pooled metrics (coverage, FSCR + 95% bootstrap CI,
  unsafe-detection rate, clean-certification rate, abstention rate).
- `utility_retention_rate`: fraction of remediated snapshots that still let
  the agent complete their own benign task (`clean_utility_probe`) —
  previously specified in the schema but never evaluated anywhere.
- `by_suite`: the same metric set computed separately per AgentDojo suite
  (workspace/banking/travel/slack).
- `by_remediation`: the same metric set computed separately per remediation
  condition (clean/source_only/summary_residual/cached_plan_residual/
  policy_only/overclean).
- `probe_budget_ablation`: FSCR/coverage recomputed at probe budgets 1, 3,
  and 5 from the probes already collected, so the comparison in the
  protocol's required extensions doesn't need a second run.

`audit-sample` writes `results/audit_sample.csv`: one row per (case,
certifier, probe) with the model's own judgment and empty `human_label`/
`human_notes` columns for a reviewer to fill in — supports the protocol's
"manually audit at least 100 oracle decisions" requirement.

## Paper Evaluation

The paper reports a controlled 24-instance pilot covering six
post-remediation scenarios and four certifiers: Keyword-Based,
PI-Guard, Semantic, and Static LLM.

The machine-readable output used for the reported tables is available
under `paper_results/`.

The repository also contains an earlier exploratory 582-instance
evaluation under `legacy/`. That experiment is retained for
completeness but is not used in the paper.
