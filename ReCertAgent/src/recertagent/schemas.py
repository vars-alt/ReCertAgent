from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

# Plain `(str, Enum)` rather than `enum.StrEnum` on purpose: StrEnum only
# exists on Python >= 3.11, but Kaggle notebooks (and this project's own
# pyproject.toml, which allows >=3.10) may run 3.10, where `from enum import
# StrEnum` raises ImportError before anything else in the package can run.
# `(str, Enum)` gives identical str-equality/serialization behavior on every
# supported version.
class Label(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    ABSTAIN = "ABSTAIN"

class MemoryItem(BaseModel):
    item_id: str
    text: str
    kind: str
    parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class AgentSnapshot(BaseModel):
    snapshot_id: str
    task: str
    memory: list[MemoryItem] = Field(default_factory=list)
    summaries: list[MemoryItem] = Field(default_factory=list)
    cached_plans: list[MemoryItem] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)

class HiddenTruth(BaseModel):
    attack_family: str
    attack_payload: str
    trigger_concepts: list[str]
    poisoned_ids: list[str]
    residual_ids: list[str]
    forbidden_actions: list[str]
    actually_safe: bool

class BenchmarkCase(BaseModel):
    case_id: str
    source_dataset: str
    source_id: str
    remediation: str
    snapshot: AgentSnapshot
    hidden: HiddenTruth
    held_out_probes: list[str]
    clean_utility_probe: str

class PublicCase(BaseModel):
    case_id: str
    source_dataset: str
    source_id: str
    remediation: str
    snapshot: AgentSnapshot
    clean_utility_probe: str

    @classmethod
    def from_private(cls, case: BenchmarkCase):
        return cls(
            case_id=case.case_id,
            source_dataset=case.source_dataset,
            source_id=case.source_id,
            remediation=case.remediation,
            snapshot=case.snapshot,
            clean_utility_probe=case.clean_utility_probe,
        )

class Decision(BaseModel):
    case_id: str
    certifier: str
    label: Label
    unsafe_probability: float
    reason: str
    probes_used: int = 0
    evidence: list[str] = Field(default_factory=list)

class ProbeOutcome(BaseModel):
    prompt: str
    response: str
    unsafe: bool
    evidence: list[str] = Field(default_factory=list)

class UtilityOutcome(BaseModel):
    """Result of running the case's `clean_utility_probe` against the
    remediated snapshot. This implements the "utility retained after
    remediation" secondary endpoint listed in paper/experiment_protocol.md,
    which was previously specified in the schema but never evaluated
    anywhere in the pipeline."""
    prompt: str
    response: str
    refused: bool
    preserved: bool
