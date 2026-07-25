import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from recertagent.adapters.agentdojo import load_agentdojo_tasks
from recertagent.adapters.notinject import load_notinject
from recertagent.remediation.builder import build_cases
from recertagent.io import write_jsonl
from recertagent.runner import run
from recertagent.evaluation.metrics import (
    compute_metrics,
    bootstrap_fscr,
    compute_metrics_by,
    compute_budget_ablation,
    suite_of,
)

def prepare(args):
    tasks = load_agentdojo_tasks(args.limit_agentdojo)
    benign = load_notinject()
    cases = build_cases(tasks, benign)
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} cases to {args.output}")

def _load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]

def summarize(args):
    rows = _load_rows(args.input)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["decision"]["certifier"]].append(row)

    summary = {}
    for name, group in grouped.items():
        entry = compute_metrics(group)
        entry["fscr_95ci"] = bootstrap_fscr(group, samples=args.bootstrap_samples)
        # Required final-paper analyses from paper/experiment_protocol.md
        # ("Split by AgentDojo suite, never random-only split", "falsification
        # by remediation condition", "Compare probe budgets 1, 3, and 5").
        # These were listed in the protocol but never computed anywhere in
        # the original pipeline; the overall pooled metrics above hide
        # exactly the kind of per-suite / per-condition failure modes a
        # reviewer would ask about.
        entry["by_suite"] = compute_metrics_by(group, suite_of)
        entry["by_remediation"] = compute_metrics_by(group, lambda r: r["remediation"])
        entry["probe_budget_ablation"] = compute_budget_ablation(group)
        summary[name] = entry

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

def audit_sample(args):
    """Export a random sample of oracle decisions to CSV for manual review.

    Directly implements the protocol's "Manually audit at least 100 oracle
    decisions" requirement, which had no supporting tooling at all. Produces
    one row per (case, certifier, probe) with an empty `human_label` column
    for an annotator to fill in, plus the model's own judgment for
    comparison (agreement/kappa can be computed from this afterward).
    """
    rows = _load_rows(args.input)
    records = []
    for row in rows:
        for outcome in row.get("held_out_outcomes", []):
            records.append({
                "case_id": row["case_id"],
                "source_id": row["source_id"],
                "remediation": row["remediation"],
                "certifier": row["decision"]["certifier"],
                "certifier_label": row["decision"]["label"],
                "probe": outcome["prompt"],
                "response": outcome["response"],
                "oracle_unsafe": outcome["unsafe"],
                "oracle_evidence": "; ".join(outcome.get("evidence", [])),
                "human_label": "",
                "human_notes": "",
            })

    rng = random.Random(args.seed)
    rng.shuffle(records)
    sample = records[: args.n]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0].keys()) if sample else [])
        writer.writeheader()
        writer.writerows(sample)
    print(f"Wrote {len(sample)} rows (of {len(records)} available) to {output}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--limit-agentdojo", type=int, default=97)
    p.add_argument("--output", default="data/processed/cases.jsonl")
    p.set_defaults(func=prepare)

    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/kaggle_smoke.yaml")
    r.set_defaults(func=lambda args: run(args.config))

    s = sub.add_parser("summarize")
    s.add_argument("--input", default="results/predictions.jsonl")
    s.add_argument("--output", default="results/summary.json")
    s.add_argument("--bootstrap-samples", type=int, default=2000)
    s.set_defaults(func=summarize)

    a = sub.add_parser("audit-sample")
    a.add_argument("--input", default="results/predictions.jsonl")
    a.add_argument("--output", default="results/audit_sample.csv")
    a.add_argument("--n", type=int, default=100)
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=audit_sample)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
