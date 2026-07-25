# Kaggle notebook cells

Prerequisites (see `README.md` for details):

- Upload `ReCertAgent.zip` **as-is** (don't unzip) as a private Kaggle
  Dataset named `recertagent`, and attach it to the notebook via **Add
  Input**.
- Notebook settings: **Accelerator = GPU** (T4 x2, single T4, or L4;
  P100 also works, just slower — see `CHANGELOG.md`), **Internet = On**.

## Cell 1: install

```python
!unzip -q /kaggle/input/recertagent/ReCertAgent.zip -d /kaggle/working/
%cd /kaggle/working/ReCertAgent
!pip install -q -e ".[dev]"
```

This installs AgentDojo's full dependency tree (it unconditionally pulls in
`anthropic`, `cohere`, `google-genai`, `langchain`, `openai` even though none
of them are used here — they're plain dependencies, not optional extras).
Expect 3-6 minutes, not a hang.

## Cell 2: tests

```python
!pytest -q
```

Should print `XX passed`. If this fails, nothing later in the notebook is
worth trusting — stop and check the traceback before continuing.

## Cell 3: data preparation (smoke)

```python
!python -m recertagent.cli prepare --limit-agentdojo 20
```

Downloads AgentDojo suites and the NotInject dataset and builds
`data/processed/cases.jsonl`. If this raises `RuntimeError: Could not
extract AgentDojo tasks...`, it means AgentDojo's API changed again since
0.1.35 — the error message includes diagnostics from every attempted
import path/signature to help debug it.

## Cell 4: smoke experiment

```python
!python -m recertagent.cli run --config configs/kaggle_smoke.yaml
```

Downloads Qwen2.5-3B-Instruct, PIGuard, and BGE-small-en-v1.5, then runs all
four certifiers over 24 post-remediation cases. Writes
`results/predictions.jsonl`.

## Cell 5: summary

```python
!python -m recertagent.cli summarize --input results/predictions.jsonl
```

Writes `results/summary.json` with, per certifier: overall metrics, a
bootstrap 95% CI on FSCR, utility retention, per-suite breakdown,
per-remediation-condition breakdown, and a probe-budget-1/3/5 ablation.

## Cell 6 (optional): manual-audit export

```python
!python -m recertagent.cli audit-sample --input results/predictions.jsonl --n 100
```

Writes `results/audit_sample.csv` — one row per (case, certifier, probe)
with the oracle's own judgment and blank `human_label`/`human_notes`
columns, for the paper's "manually audit at least 100 oracle decisions"
requirement.

## Cell 7: full run

Once the smoke run above succeeds end-to-end, rerun with the full
configuration:

```python
!python -m recertagent.cli prepare --limit-agentdojo 97
!python -m recertagent.cli run --config configs/kaggle_full.yaml
!python -m recertagent.cli summarize --input results/predictions.jsonl
!python -m recertagent.cli audit-sample --input results/predictions.jsonl --n 100
```

## Cell 8: save outputs

```python
import shutil
shutil.make_archive("/kaggle/working/recertagent_results", "zip", "results")
```

Download `recertagent_results.zip` from the Kaggle output pane — it
contains `predictions.jsonl`, `summary.json`, and `audit_sample.csv`.
