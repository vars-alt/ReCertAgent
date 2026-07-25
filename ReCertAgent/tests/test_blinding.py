from recertagent.schemas import (
    AgentSnapshot,
    BenchmarkCase,
    HiddenTruth,
    PublicCase,
)

def test_public_case_removes_hidden_attack_information():
    private = BenchmarkCase(
        case_id="case",
        source_dataset="source",
        source_id="1",
        remediation="clean",
        snapshot=AgentSnapshot(snapshot_id="snapshot", task="task"),
        hidden=HiddenTruth(
            attack_family="family",
            attack_payload="SECRET_PAYLOAD",
            trigger_concepts=["trigger"],
            poisoned_ids=["poison"],
            residual_ids=[],
            forbidden_actions=["external_export"],
            actually_safe=True,
        ),
        held_out_probes=["probe"],
        clean_utility_probe="task",
    )
    public = PublicCase.from_private(private)
    assert "SECRET_PAYLOAD" not in public.model_dump_json()
    assert not hasattr(public, "hidden")
