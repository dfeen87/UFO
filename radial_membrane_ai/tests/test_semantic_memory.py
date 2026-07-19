"""
Comprehensive unit and integration tests for the Policy-Bound Semantic Memory layer.
Achieving 100% line coverage.
"""

from __future__ import annotations
import time
import pytest
import numpy as np

from radial_membrane_ai.residuals import ResidualLedger, ResidualRecord
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.ufo_engine import SingleAgentEngine, MultiAgentEngine
from radial_membrane_ai.semantic_memory.policy import MemoryPolicy, AdmissibilityContext, PolicyContext
from radial_membrane_ai.semantic_memory.curvature import MemoryCurvatureState
from radial_membrane_ai.semantic_memory.core import (
    MemoryRecord,
    MemoryEvent,
    AgentViewConfig,
    AgentSemanticMemory,
    MeshSemanticMemory,
    write_memory,
    read_memory,
    list_memory,
    promote_memory,
    read_shared_memory,
    list_shared_memory
)
from radial_membrane_ai.semantic_memory.integration import (
    bind_to_agent,
    bind_to_mesh,
    on_tick_start,
    on_tick_end,
    on_sao_promotion
)


def test_memory_policy() -> None:
    # 1. Test check_write_admissible when quarantined
    policy = MemoryPolicy(admissibility_required=True, sensitivity="high", governor_override_allowed=False)
    record = MemoryRecord(
        key="k1", value="v1", tags={"tag1"}, created_at=time.time(), updated_at=time.time(),
        origin_agent_id="agent_1", policy_envelope=policy
    )
    context_q = AdmissibilityContext(coherence=0.9, quarantined=True, cost_factor=1.0, stability_energy=1.0)
    assert policy.check_write_admissible(record, context_q) is False

    # 2. Test check_write_admissible with low coherence
    context_low_coh = AdmissibilityContext(coherence=0.3, quarantined=False, cost_factor=1.0, stability_energy=1.0)
    assert policy.check_write_admissible(record, context_low_coh) is False

    # 3. Test check_write_admissible with high stability energy and no override allowed
    context_unstable = AdmissibilityContext(coherence=0.9, quarantined=False, cost_factor=1.0, stability_energy=5.0)
    assert policy.check_write_admissible(record, context_unstable) is False

    # 4. Test check_write_admissible with high stability energy with override allowed
    policy_override = MemoryPolicy(admissibility_required=True, sensitivity="high", governor_override_allowed=True)
    assert policy_override.check_write_admissible(record, context_unstable) is True

    # 5. Test check_write_admissible sensitivity cases
    policy_medium = MemoryPolicy(sensitivity="medium")
    assert policy_medium.check_write_admissible(record, AdmissibilityContext(0.45, False, 1.0, 1.0)) is False
    assert policy_medium.check_write_admissible(record, AdmissibilityContext(0.55, False, 1.0, 1.0)) is True

    # Test sensitivity high with coherence between 0.4 and 0.7
    policy_high = MemoryPolicy(sensitivity="high")
    assert policy_high.check_write_admissible(record, AdmissibilityContext(0.5, False, 1.0, 1.0)) is False

    # 6. Test check_read_allowed when reader quarantined
    policy_read = MemoryPolicy(read_scope="local")
    p_context_q = PolicyContext(coherence=0.9, quarantined_agents=["agent_1"])
    assert policy_read.check_read_allowed(record, "agent_1", p_context_q) is False

    # 7. Test check_read_allowed local scope match/mismatch
    p_context_ok = PolicyContext(coherence=0.9, quarantined_agents=[])
    assert policy_read.check_read_allowed(record, "agent_1", p_context_ok) is True
    assert policy_read.check_read_allowed(record, "agent_2", p_context_ok) is False

    # 8. Test check_read_allowed shared scope
    policy_shared = MemoryPolicy(read_scope="shared", sensitivity="high")
    assert policy_shared.check_read_allowed(record, "agent_2", PolicyContext(0.2, [])) is False
    assert policy_shared.check_read_allowed(record, "agent_2", PolicyContext(0.5, [])) is False
    assert policy_shared.check_read_allowed(record, "agent_2", PolicyContext(0.7, [])) is True

    # 9. Test check_read_allowed global scope
    policy_global = MemoryPolicy(read_scope="global", sensitivity="high")
    assert policy_global.check_read_allowed(record, "agent_2", PolicyContext(0.3, [])) is False
    assert policy_global.check_read_allowed(record, "agent_2", PolicyContext(0.5, [])) is True


def test_memory_curvature_state() -> None:
    state = MemoryCurvatureState()
    assert state.curvature == 0.0
    assert state.tension == 0.0

    state.update_on_write()
    assert state.access_count == 1
    assert state.curvature == 0.1

    state.update_on_read()
    assert state.access_count == 2
    assert state.curvature == pytest.approx(0.12)

    state.update_on_conflict()
    assert state.conflict_count == 1
    assert state.tension == 0.25

    # Test clamping
    state.curvature = 4.95
    state.update_on_write(size_factor=1.0)
    assert state.curvature == 5.0

    state.tension = 4.9
    state.update_on_conflict(conflict_intensity=1.0)
    assert state.tension == 5.0

    # Test decay
    state.decay(rate=0.1)
    assert state.curvature == 4.9
    assert state.tension == 4.95


def test_core_apis_agent_memory() -> None:
    ledger = ResidualLedger()
    agent_mem = AgentSemanticMemory(agent_id="a1")
    policy = MemoryPolicy()
    context = AdmissibilityContext(coherence=0.8, quarantined=False, cost_factor=1.0, stability_energy=1.0)

    # 1. Success Write
    res = write_memory(agent_mem, "k1", "value1", {"tagA"}, policy, context, global_ledger=ledger)
    assert res.success is True
    assert res.record is not None
    assert res.record.key == "k1"
    assert res.record.value == "value1"

    # 2. Overwrite and check residual preservation
    res_over = write_memory(agent_mem, "k1", "value1_updated", {"tagA", "tagB"}, policy, context, global_ledger=ledger)
    assert res_over.success is True
    assert len(agent_mem.residuals) == 1
    assert agent_mem.residuals[0].details["overwritten_value"] == "value1"
    assert len(ledger.records) == 1
    assert ledger.records[0].error_type == "memory_overwrite_residual"

    # 3. Read Match
    p_context = PolicyContext(coherence=0.8, quarantined_agents=[])
    rec_read = read_memory(agent_mem, "k1", p_context)
    assert rec_read is not None
    assert rec_read.value == "value1_updated"

    # 4. Read Miss
    assert read_memory(agent_mem, "non_existent", p_context) is None

    # 5. Read Rejected
    policy_local = MemoryPolicy(read_scope="local")
    write_memory(agent_mem, "k2", "val2", set(), policy_local, context, global_ledger=ledger)
    # Querying agent is a1, which is allowed
    rec_rejected = read_memory(agent_mem, "k2", p_context)
    assert rec_rejected is not None

    # Write a record that fails write admissibility to cover core.py:156-167
    policy_strict = MemoryPolicy(admissibility_required=True, sensitivity="high")
    context_low_coh = AdmissibilityContext(coherence=0.1, quarantined=False, cost_factor=1.0, stability_energy=1.0)
    res_rejected_write = write_memory(agent_mem, "k3", "val3", set(), policy_strict, context_low_coh, ledger)
    assert res_rejected_write.success is False

    # Local read rejected to cover core.py:228-230
    # Try to read k2 from AgentSemanticMemory using agent_id matching mismatches
    agent_mem_other = AgentSemanticMemory(agent_id="other_agent")
    agent_mem_other.local_store["k2"] = agent_mem.local_store["k2"] # origin is a1, querying is other_agent
    assert read_memory(agent_mem_other, "k2", p_context) is None

    # 6. List Memory with filters
    records_all = list_memory(agent_mem)
    assert len(records_all) == 2

    records_filtered = list_memory(agent_mem, filter_tags={"tagB"})
    assert len(records_filtered) == 1
    assert records_filtered[0].key == "k1"


def test_core_apis_mesh_memory() -> None:
    ledger = ResidualLedger()
    agent_mem = AgentSemanticMemory(agent_id="a1")
    mesh_mem = MeshSemanticMemory()
    policy = MemoryPolicy()
    context = AdmissibilityContext(coherence=0.8, quarantined=False, cost_factor=1.0, stability_energy=1.0)

    # 1. Promote non-existent
    res_fail = promote_memory(mesh_mem, agent_mem, "k_miss", context, ledger)
    assert res_fail.success is False

    # 2. Promote success
    write_memory(agent_mem, "k1", "v1", {"t1"}, policy, context, ledger)
    res_promo = promote_memory(mesh_mem, agent_mem, "k1", context, ledger)
    assert res_promo.success is True
    assert res_promo.p_sao == pytest.approx(0.2)

    # 3. Promote rejected (with low coherence)
    context_low = AdmissibilityContext(coherence=0.2, quarantined=False, cost_factor=1.0, stability_energy=1.0)
    res_fail_policy = promote_memory(mesh_mem, agent_mem, "k1", context_low, ledger)
    assert res_fail_policy.success is False
    assert any(r.error_type == "memory_promotion_rejected" for r in ledger.records)

    # 4. Shared memory read & read miss
    p_context = PolicyContext(coherence=0.8, quarantined_agents=[])
    assert read_shared_memory(mesh_mem, "a1", "non_existent", p_context) is None

    rec_shared = read_shared_memory(mesh_mem, "a1", "k1", p_context)
    assert rec_shared is not None
    assert rec_shared.value == "v1"

    # 5. Shared memory read rejected via view
    mesh_mem.agent_views["a2"] = MeshSemanticMemory().agent_views.get("a2", AgentViewConfig("a2"))
    mesh_mem.agent_views["a2"].read_granted = False
    assert read_shared_memory(mesh_mem, "a2", "k1", p_context) is None

    # Shared read rejected via policy check to cover core.py:357-359
    policy_strict_read = MemoryPolicy(read_scope="shared", sensitivity="high")
    write_memory(agent_mem, "k_strict_read", "v_strict_read", set(), policy_strict_read, context, ledger)
    promote_memory(mesh_mem, agent_mem, "k_strict_read", context, ledger)
    p_context_low_coh = PolicyContext(coherence=0.2, quarantined_agents=[])
    assert read_shared_memory(mesh_mem, "a1", "k_strict_read", p_context_low_coh) is None

    # 6. List shared memories with tag filtering
    write_memory(agent_mem, "k2", "v2", {"t2"}, policy, context, ledger)
    promote_memory(mesh_mem, agent_mem, "k2", context, ledger)

    recs_all = list_shared_memory(mesh_mem)
    assert len(recs_all) == 3

    recs_filtered = list_shared_memory(mesh_mem, filter_tags={"t2"})
    assert len(recs_filtered) == 1
    assert recs_filtered[0].key == "k2"


def test_sao_promotion_residual_overwrite_in_mesh() -> None:
    ledger = ResidualLedger()
    agent_mem = AgentSemanticMemory(agent_id="a1")
    mesh_mem = MeshSemanticMemory()
    policy = MemoryPolicy()
    context = AdmissibilityContext(coherence=0.8, quarantined=False, cost_factor=1.0, stability_energy=1.0)

    write_memory(agent_mem, "k1", "val_first", {"t1"}, policy, context, ledger)
    promote_memory(mesh_mem, agent_mem, "k1", context, ledger)

    # Overwrite in agent and promote again to trigger overwrite residual in mesh
    write_memory(agent_mem, "k1", "val_second", {"t1"}, policy, context, ledger)
    promote_memory(mesh_mem, agent_mem, "k1", context, ledger)

    assert any(r.error_type == "mesh_memory_overwrite_residual" for r in ledger.records)


def test_integration_and_hooks() -> None:
    # 1. Bind to agent and mesh
    agent = UFOAgent("a1")
    bind_to_agent(agent)
    assert hasattr(agent, "semantic_memory")
    assert isinstance(agent.semantic_memory, AgentSemanticMemory)

    # 2. Single-Agent Engine tick hooks integration
    engine_single = SingleAgentEngine()
    # Execute step & check memory hooks are integrated
    engine_single.tick(task_value=0.5, excitation=np.ones(12) * 0.1)

    # Trigger high tension in single agent memory
    engine_single.semantic_memory.curvature_state.tension = 2.0
    # Next tick should produce intervention warning
    engine_single.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert any("Memory Tension Escalation" in inter for inter in engine_single.interventions)

    # 3. Multi-Agent Engine tick hooks integration
    engine_multi = MultiAgentEngine(n_agents=2)
    engine_multi.tick(task_value=0.5, excitation=np.ones(12) * 0.1)

    # Trigger high tension in multi agent mesh memory
    engine_multi.mesh_memory.curvature_state.tension = 2.0
    engine_multi.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert any("Global Shared Memory Tension Warning" in inter for inter in engine_multi.interventions)


def test_integration_on_sao_promotion_hook() -> None:
    engine_multi = MultiAgentEngine(n_agents=2)
    agent_l = engine_multi.agents[0]
    agent_r = engine_multi.agents[1]

    # Write a key to local memory of agent_l
    policy = MemoryPolicy()
    context = AdmissibilityContext(coherence=0.9, quarantined=False, cost_factor=1.0, stability_energy=1.0)
    write_memory(agent_l.semantic_memory, "k_shared", "secret_data", {"tagA"}, policy, context)

    # Trigger a successful SAO promotion tick to verify auto promotion
    # Let's mock the execution of promotion to return "ascend"
    engine_multi.mesh_governance.execute_sao_promotion = (
        lambda a, b, **kwargs: ("ascend", 0.0, np.zeros(12))  # type: ignore
    )

    # Run tick
    engine_multi.tick(task_value=0.5, excitation=np.ones(12) * 0.1)

    # Verify that the key got auto-promoted to mesh semantic memory
    rec = read_shared_memory(
        engine_multi.mesh_memory,
        agent_r.agent_id,
        "k_shared",
        PolicyContext(coherence=0.8, quarantined_agents=[])
    )
    assert rec is not None
    assert rec.value == "secret_data"


def test_on_sao_promotion_edge_cases() -> None:
    # Test on_sao_promotion with missing mesh_memory, missing agent, etc.
    class DummyEngine:
        pass

    engine = DummyEngine()
    assert on_sao_promotion("agent_1", "key", engine) is False

    # Bind to mesh
    bind_to_mesh(engine)
    assert on_sao_promotion("agent_1", "key", engine) is False

    # Test integration.py: 103, 114-115, 122 coverage
    # Add dummy target agent with no semantic_memory
    class DummyAgent:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id

    engine.agents = [DummyAgent("agent_1")] # DummyAgent has no semantic_memory, covers 103
    assert on_sao_promotion("agent_1", "key", engine) is False

    # Add semantic_memory to DummyAgent
    agent_1 = DummyAgent("agent_1")
    agent_1.semantic_memory = None # covers 106
    engine.agents = [agent_1]
    assert on_sao_promotion("agent_1", "key", engine) is False

    # Full mock for 114-115, 122
    class MockEngine:
        def __init__(self) -> None:
            self.mesh_memory = MeshSemanticMemory()
            self.coherence_history = [0.9]
            self.v_history = [2.0]
            agent = DummyAgent("agent_1")
            agent.semantic_memory = AgentSemanticMemory("agent_1")
            # Write a key
            policy = MemoryPolicy()
            context = AdmissibilityContext(coherence=0.9, quarantined=False, cost_factor=1.0, stability_energy=1.0)
            write_memory(agent.semantic_memory, "key", "val", set(), policy, context)
            self.agents = [agent]

    mock_eng = MockEngine()
    assert on_sao_promotion("agent_1", "key", mock_eng) is True
