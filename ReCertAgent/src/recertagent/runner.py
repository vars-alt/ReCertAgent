import random
from pathlib import Path
import json
import numpy as np
import torch
import yaml
from tqdm import tqdm

from recertagent.io import read_jsonl
from recertagent.schemas import BenchmarkCase, PublicCase, Label
from recertagent.models.qwen import QwenModel
from recertagent.models.piguard import PIGuard
from recertagent.models.embedder import SemanticInspector
from recertagent.certifiers.baselines import (
    KeywordCertifier,
    PIGuardCertifier,
    SemanticCertifier,
    LLMStaticCertifier,
)
from recertagent.evaluation.oracle import BehavioralOracle


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    exp = cfg["experiment"]
    seed = cfg.get("seed", 42)
    _seed_everything(seed)

    cases = read_jsonl(cfg["input_path"], BenchmarkCase)[:exp["max_cases"]]

    qcfg = cfg["models"]["qwen"]
    qwen = QwenModel(
        qcfg["name"],
        load_in_4bit=qcfg["load_in_4bit"],
        max_new_tokens=qcfg["max_new_tokens"],
    )
    oracle = BehavioralOracle(qwen)
    utility_cache = {}

    certifiers = []
    for name in exp["certifiers"]:
        if name == "keyword":
            certifiers.append(KeywordCertifier())
        elif name == "piguard":
            certifiers.append(PIGuardCertifier(
                PIGuard(cfg["models"]["piguard"]["name"]),
                exp["abstain_low"],
                exp["abstain_high"],
            ))
        elif name == "semantic":
            certifiers.append(SemanticCertifier(
                SemanticInspector(cfg["models"]["embedder"]["name"])
            ))
        elif name == "llm_static":
            certifiers.append(LLMStaticCertifier(qwen))
        else:
            raise ValueError(f"Unknown certifier: {name}")

    output = Path(cfg["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in tqdm(cases, desc="Post-remediation states"):
            public = PublicCase.from_private(case)

            if case.case_id not in utility_cache:
                utility_cache[case.case_id] = oracle.run_utility_probe(case).model_dump()
            utility_outcome = utility_cache[case.case_id]

            for certifier in certifiers:
                decision = certifier.certify(public)
                outcomes = []
                if decision.label == Label.SAFE:
                    for probe in case.held_out_probes[:exp["held_out_probe_budget"]]:
                        outcomes.append(oracle.run_probe(case, probe).model_dump())
                row = {
                    "case_id": case.case_id,
                    "source_id": case.source_id,
                    "remediation": case.remediation,
                    "actually_safe": case.hidden.actually_safe,
                    "decision": decision.model_dump(mode="json"),
                    "held_out_outcomes": outcomes,
                    "falsified": any(x["unsafe"] for x in outcomes),
                    "utility_outcome": utility_outcome,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
