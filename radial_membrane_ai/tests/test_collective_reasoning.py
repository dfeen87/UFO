# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Comprehensive Unit Tests for Policy-Bound Collective Reasoning Governance Layer.
"""

from __future__ import annotations
import numpy as np
import pytest
import time
from typing import Any

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.residuals import ResidualLedger
from radial_membrane_ai.semantic_memory.core import MemoryRecord, MeshSemanticMemory, write_memory
from radial_membrane_ai.semantic_memory.policy import MemoryPolicy, AdmissibilityContext
from radial_membrane_ai.semantic_memory.integration import bind_to_agent

# Collective Reasoning imports
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.collective_reasoning.collective_admissibility import (
    CollectiveStepContext, collective_admissibility
)
from radial_membrane_ai.collective_reasoning.collective_sao import collective_sao_promote
from radial_membrane_ai.collective_reasoning.mesh_correctness import MeshCorrectness
from radial_membrane_ai.collective_reasoning.coherence import (
    coherence_score,
    cluster_coherence_score,
    cosine_similarity,
    jaccard_similarity,
    value_similarity
)
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine


def test_cosine_and_jaccard_edge_cases() -> None:
    # Zero norm cosine similarity fallback
    assert cosine_similarity(np.zeros(5), np.ones(5)) == 1.0
    assert cosine_similarity(np.ones(5), np.zeros(5)) == 1.0

    # Empty set Jaccard fallback
    assert jaccard_similarity(set(), set()) == 1.0


def test_value_similarity_types() -> None:
    # Diff types
    assert value_similarity("hello", 123) == 0.0
    # Numpy arrays
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([3, 2, 1])
    assert value_similarity(arr1, arr2) == pytest.approx(1.0)
    assert value_similarity(arr1, arr3) < 1.0
    # List of different shape
    assert value_similarity(arr1, np.array([1, 2])) == 0.0

    # Lists (use completely orthogonal vectors to get 0.0)
    assert value_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert value_similarity([1, 0], [0, 1]) == 0.0
    # Non-convertible lists
    assert value_similarity(["a", "b"], ["a", "b"]) == 1.0

    # Strings
    assert value_similarity("a b", "a b") == 1.0
    assert value_similarity("a b", "a c") == pytest.approx(1.0 / 3.0)

    # Numbers
    assert value_similarity(10, 10) == 1.0
    assert value_similarity(10, 8) == 0.8

    # Fallback equality
    class CustomObj:
        def __init__(self, x: int) -> None:
            self.x = x

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, CustomObj) and self.x == other.x

    assert value_similarity(CustomObj(5), CustomObj(5)) == 1.0
    assert value_similarity(CustomObj(5), CustomObj(10)) == 0.0


def test_policy_envelope_intersection_and_compliance() -> None:
    env1 = PolicyEnvelope(
        allowed_agents={"agent_1", "agent_2"},
        allowed_clusters={"cluster_1"},
        min_trust=0.5,
        allowed_roles={"planner"},
        promotable_memory_fields={"field_1"},
        allowed_tags={"tag_1"},
        max_cost_band=1,
        stability_required_band="yellow",
        temporal_window_min=10,
        temporal_window_max=100
    )

    env2 = PolicyEnvelope(
        allowed_agents={"agent_2", "agent_3"},
        allowed_clusters={"cluster_1", "cluster_2"},
        min_trust=0.6,
        allowed_roles={"planner", "critic"},
        promotable_memory_fields={"field_1", "field_2"},
        allowed_tags={"tag_1", "tag_2"},
        max_cost_band=2,
        stability_required_band="green",
        temporal_window_min=20,
        temporal_window_max=80
    )

    # Intersection
    intersected = env1.intersect(env2)
    assert intersected.allowed_agents == {"agent_2"}
    assert intersected.allowed_clusters == {"cluster_1"}
    assert intersected.min_trust == 0.6
    assert intersected.allowed_roles == {"planner"}
    assert intersected.promotable_memory_fields == {"field_1"}
    assert intersected.allowed_tags == {"tag_1"}
    assert intersected.max_cost_band == 1
    assert intersected.stability_required_band == "green"
    assert intersected.temporal_window_min == 20
    assert intersected.temporal_window_max == 80

    # Compliance
    assert intersected.validate_compliance(
        current_trust=0.7,
        current_cost_band=0,
        current_stability_band="green",
        current_ticks=30,
        agent_ids={"agent_2"},
        agent_roles={"planner"},
        cluster_ids={"cluster_1"},
        used_tags={"tag_1"},
        used_fields={"field_1"}
    ) is True

    # Violation check: trust too low
    assert intersected.validate_compliance(0.5, 0, "green", 30) is False
    # Violation check: cost band too high
    assert intersected.validate_compliance(0.7, 2, "green", 30) is False
    # Violation check: stability band too low
    assert intersected.validate_compliance(0.7, 0, "yellow", 30) is False
    # Violation: ticks < temporal_window_min
    assert intersected.validate_compliance(0.7, 0, "green", 10) is False
    # Violation: ticks > temporal_window_max
    assert intersected.validate_compliance(0.7, 0, "green", 90) is False
    # Violation: non-allowed agent
    assert intersected.validate_compliance(0.7, 0, "green", 30, agent_ids={"agent_1"}) is False
    # Violation: non-allowed role
    assert intersected.validate_compliance(0.7, 0, "green", 30, agent_roles={"critic"}) is False
    # Violation: non-allowed cluster
    assert intersected.validate_compliance(0.7, 0, "green", 30, cluster_ids={"cluster_2"}) is False
    # Violation: non-allowed tags
    assert intersected.validate_compliance(0.7, 0, "green", 30, used_tags={"tag_2"}) is False
    # Violation: non-allowed field
    assert intersected.validate_compliance(0.7, 0, "green", 30, used_fields={"field_2"}) is False


def test_cross_agent_admissibility() -> None:
    agent = UFOAgent("agent_1")
    bind_to_agent(agent)
    cluster = UFOCluster("cluster_1")
    cluster.add_agent(agent)

    # Initialize temporal state histories
    from radial_membrane_ai.temporal import TemporalMembraneState
    t_state = TemporalMembraneState()
    # Populate tick histories to simulate running state
    for _ in range(10):
        t_state.update_tick_history(max_curvature=0.1, max_tension=0.2, max_closure_ratio=0.5)
    setattr(agent.membrane, "temporal_state", t_state)

    # Setup cluster tension history
    cluster.tension_metric = 0.3
    cluster.stability_band = "green"

    global_env = PolicyEnvelope(
        allowed_agents={"agent_1"},
        allowed_clusters={"cluster_1"},
        stability_required_band="yellow"
    )

    step = CollectiveStepContext(
        participating_agent_ids={"agent_1"},
        involved_cluster_ids={"cluster_1"},
        cost_band=0,
        stability_band="green",
        trust_score=0.8,
        ticks=10
    )

    # Valid step
    assert collective_admissibility([agent], [cluster], step, global_env) is True

    # Invalid: no matching agent
    invalid_step = CollectiveStepContext(participating_agent_ids={"agent_invalid"})
    assert collective_admissibility([agent], [cluster], invalid_step, global_env) is False

    # Invalid: instantaneous closure ratio > 1.0 (elevate ONLY first string activations AND set tiny boundary)
    agent.membrane.strings[0].activation = 1.0
    agent.membrane.strings[0].radius = 1.0
    agent.boundary.radius_deviation = {i: -0.99 for i in range(1, 13)}
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    # Reset
    agent.membrane.strings[0].activation = 0.0
    agent.membrane.strings[0].radius = 0.0
    agent.boundary.radius_deviation = {i: 0.0 for i in range(1, 13)}

    # Invalid: time-weighted closure ratio > 1.1
    t_state.admissibility_history.clear()
    t_state.admissibility_history.append(2.0)
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    # Restore
    t_state.admissibility_history.clear()
    for _ in range(10):
        t_state.admissibility_history.append(0.5)

    # Invalid: ticks < N_temporal_min
    t_state.consecutive_admissible_ticks = 2
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    t_state.consecutive_admissible_ticks = 10

    # Invalid: average tension too high
    t_state.tension_history.clear()
    t_state.tension_history.append(5.0)
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    # Restore
    t_state.tension_history.clear()
    for _ in range(10):
        t_state.tension_history.append(0.2)

    # Invalid: cluster tension too high
    cluster.tension_metric = 5.0
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    cluster.tension_metric = 0.3

    # Invalid: cluster stability band 'red'
    cluster.stability_band = "red"
    assert collective_admissibility([agent], [cluster], step, global_env) is False
    cluster.stability_band = "green"


def test_collective_sao_promotions() -> None:
    agent = UFOAgent("agent_1")
    bind_to_agent(agent)
    cluster = UFOCluster("cluster_1")
    mesh_memory = MeshSemanticMemory()
    ledger = ResidualLedger()

    # Initialize temporal state
    from radial_membrane_ai.temporal import TemporalMembraneState
    t_state = TemporalMembraneState()
    for _ in range(10):
        t_state.update_tick_history(0.1, 0.2, 0.5)
    t_state.accumulated_tension = 0.5
    setattr(agent.membrane, "temporal_state", t_state)

    global_env = PolicyEnvelope(min_trust=0.4, stability_required_band="yellow")

    # Local SAO (agent -> cluster)
    success, p_sao, rec_local = collective_sao_promote(
        level="local",
        source_entity=agent,
        target_entity=cluster,
        key="test_key",
        global_envelope=global_env,
        coherence_score_val=0.8,
        stability_bands={"global": "green", "cluster_1": "green"},
        current_ticks=10,
        ledger=ledger
    )
    assert success is True
    assert p_sao == pytest.approx(0.2)
    assert rec_local is not None
    assert rec_local.key == "test_key"
    assert "local" in rec_local.tags

    # Local SAO fails because ticks < 5
    t_state.consecutive_admissible_ticks = 2
    success, _, _ = collective_sao_promote("local", agent, cluster, "test_key", global_env, 0.8, {"global": "green"}, 2)
    assert success is False
    t_state.consecutive_admissible_ticks = 10

    # Cluster SAO (cluster -> global mesh)
    # Give cluster a default temporal state as well
    c_t_state = TemporalMembraneState()
    for _ in range(25):
        c_t_state.update_tick_history(0.1, 0.2, 0.5)
    setattr(cluster.membrane, "temporal_state", c_t_state)
    cluster.tension_metric = 0.2
    cluster.stability_band = "green"

    # Add memory key to cluster global store to promote
    rec_to_promote = MemoryRecord("test_key", "value", set(), time.time(), time.time(), "agent_1", global_env)
    cluster.semantic_memory.global_store["test_key"] = rec_to_promote

    success, p_sao, rec_cluster = collective_sao_promote(
        level="cluster",
        source_entity=cluster,
        target_entity=mesh_memory,
        key="test_key",
        global_envelope=global_env,
        coherence_score_val=0.8,
        stability_bands={"global": "green", "cluster_1": "green"},
        current_ticks=25,
        ledger=ledger
    )
    assert success is True
    assert rec_cluster is not None
    assert "cluster" in rec_cluster.tags

    # Cluster SAO fails because ticks < 20
    c_t_state.consecutive_admissible_ticks = 10
    success, _, _ = collective_sao_promote(
        "cluster", cluster, mesh_memory, "test_key", global_env, 0.8, {"global": "green"}, 10
    )
    assert success is False
    c_t_state.consecutive_admissible_ticks = 25

    # Global SAO (mesh -> global semantic memory)
    mesh_memory.global_store["test_key"] = rec_cluster
    success, p_sao, rec_global = collective_sao_promote(
        level="global",
        source_entity=mesh_memory,
        target_entity=mesh_memory,
        key="test_key",
        global_envelope=global_env,
        coherence_score_val=0.9,
        stability_bands={"global": "green"},
        current_ticks=60,
        ledger=ledger
    )
    assert success is True
    assert rec_global is not None
    assert "global" in rec_global.tags


def test_mesh_correctness_and_rollbacks() -> None:
    agent = UFOAgent("agent_1")
    bind_to_agent(agent)
    cluster = UFOCluster("cluster_1")
    cluster.add_agent(agent)
    mesh_memory = MeshSemanticMemory()
    checker = MeshCorrectness()

    # Pre-populate state
    agent.membrane.strings[0].activation = 0.5
    agent.membrane.strings[0].radius = 0.5
    agent.boundary.radius_deviation[1] = 0.2
    agent.residual_history.append(0.1)

    # Put a key in global store to ensure it is rolled back
    mesh_memory.global_store["some_key"] = MemoryRecord(
        "some_key", "original_val", set(), time.time(), time.time(), "agent_1", PolicyEnvelope()
    )

    # Backup
    backup = checker.backup_state([agent], [cluster], mesh_memory)

    # Mutate state
    agent.membrane.strings[0].activation = 1.0
    agent.membrane.strings[0].radius = 1.0
    agent.boundary.radius_deviation[1] = 0.9
    agent.residual_history.append(10.0)
    mesh_memory.global_store["some_key"] = MemoryRecord(
        "some_key", "mutated_val", set(), time.time(), time.time(), "agent_1", PolicyEnvelope()
    )

    # Rollback
    checker.rollback(backup, [agent], [cluster], mesh_memory)

    # Verify state was restored
    assert agent.membrane.strings[0].activation == 0.5
    assert agent.membrane.strings[0].radius == 0.5
    assert agent.boundary.radius_deviation[1] == 0.2
    assert agent.residual_history == [0.1]
    assert mesh_memory.global_store["some_key"].value == "original_val"


def test_mesh_correctness_audit() -> None:
    agent = UFOAgent("agent_1")
    bind_to_agent(agent)
    cluster = UFOCluster("cluster_1")
    cluster.add_agent(agent)
    checker = MeshCorrectness()

    global_env = PolicyEnvelope(max_cost_band=1, stability_required_band="green")
    step = CollectiveStepContext(
        participating_agent_ids={"agent_1"},
        involved_cluster_ids={"cluster_1"},
        cost_band=0,
        stability_band="green"
    )

    # Clear valid audit
    is_correct, violations = checker.audit_correctness([agent], [cluster], step, global_env)
    assert is_correct is True
    assert not violations

    # Overwrite check violation: memory record has newer update time but no residual state
    record = MemoryRecord(
        key="key1",
        value="val",
        tags=set(),
        created_at=time.time() - 10.0,
        updated_at=time.time(),
        origin_agent_id="agent_1",
        policy_envelope=global_env,
        residual_state=None
    )
    agent.semantic_memory.local_store["key1"] = record
    is_correct, violations = checker.audit_correctness([agent], [cluster], step, global_env)
    assert is_correct is False
    assert any("Destructive overwrite" in v for v in violations)


def test_emergent_coherence_scores() -> None:
    agent_1 = UFOAgent("agent_1")
    bind_to_agent(agent_1)
    agent_2 = UFOAgent("agent_2")
    bind_to_agent(agent_2)
    cluster = UFOCluster("cluster_1", agents=[agent_1, agent_2])

    from radial_membrane_ai.temporal import TemporalMembraneState
    t1 = TemporalMembraneState()
    t2 = TemporalMembraneState()
    for _ in range(5):
        t1.update_tick_history(0.1, 0.2, 0.3)
        t2.update_tick_history(0.1, 0.2, 0.3)
    setattr(agent_1.membrane, "temporal_state", t1)
    setattr(agent_2.membrane, "temporal_state", t2)

    # Initial perfect coherence (both stores empty)
    c_score = cluster_coherence_score(cluster)
    assert c_score == pytest.approx(1.0)

    # Write identical values
    policy = MemoryPolicy(min_trust=0.4)
    ctx = AdmissibilityContext(1.0, False, 1.0, 0.0)
    write_memory(agent_1.semantic_memory, "k1", "val", {"tag1"}, policy, ctx)
    write_memory(agent_2.semantic_memory, "k1", "val", {"tag1"}, policy, ctx)

    c_score = cluster_coherence_score(cluster)
    assert c_score == pytest.approx(1.0)


def test_engine_integration_ticks() -> None:
    # 1. Test MultiAgentEngine with collective reasoning enabled
    engine = MultiAgentEngine(n_agents=2)
    engine.collective_enabled = True

    # Setup default states to allow admissibility to pass
    for agent in engine.agents:
        t_state = getattr(agent.membrane, "temporal_state", None)
        if t_state is not None:
            # Set tick counts >= 5 to satisfy short-range gate
            t_state.consecutive_admissible_ticks = 6
            t_state.admissibility_history.append(0.5)
            t_state.tension_history.append(0.1)

    # Tick
    band = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert band in ("green", "yellow", "red")

    # 2. Test MultiClusterEngine with collective reasoning enabled
    c_engine = MultiClusterEngine()
    c_engine.collective_enabled = True
    c_engine.create_cluster("cluster_1")

    agent_c1 = UFOAgent("agent_1_c1")
    c_engine.assign_agent_to_cluster(agent_c1, "cluster_1")

    # Set temporal gates
    t_state_c = getattr(agent_c1.membrane, "temporal_state", None)
    if t_state_c is not None:
        t_state_c.consecutive_admissible_ticks = 55
        t_state_c.admissibility_history.append(0.5)
        t_state_c.tension_history.append(0.1)
    c_engine.temporal_state.consecutive_admissible_ticks = 55

    # Create dummy cluster record to promote
    cluster_rec = MemoryRecord(
        "k1", "val", {"tag1"}, time.time(), time.time(), "cluster_1", c_engine.global_envelope
    )
    c_engine.clusters["cluster_1"].semantic_memory.global_store["k1"] = cluster_rec

    # Tick
    band_c = c_engine.tick(task_value=0.5, default_excitation=np.ones(12) * 0.1)
    assert band_c in ("green", "yellow", "red")


def test_extra_covered_lines() -> None:
    # 1. coherence.py extra coverage
    # Empty semantic memories List
    assert coherence_score([]) == 1.0

    agent1 = UFOAgent("agent_1")
    agent2 = UFOAgent("agent_2")
    # Empty memories coherence fallback
    from radial_membrane_ai.collective_reasoning.coherence import compute_semantic_memory_coherence
    assert compute_semantic_memory_coherence([agent1, agent2]) == 1.0

    # Compare non-disjoint list elements value_similarity fallback try block except
    assert value_similarity(("a", "b"), ("b", "c")) == pytest.approx(1.0 / 3.0)

    # 2. collective_sao.py extra coverage
    # Invalid level check
    with pytest.raises(ValueError, match="Unknown level"):
        collective_sao_promote("invalid_level", agent1, None, "k", PolicyEnvelope(), 1.0, {}, 1)

    # SAO local ticks < 5
    success, _, _ = collective_sao_promote("local", agent1, None, "k", PolicyEnvelope(), 1.0, {}, 1)
    assert success is False

    # SAO local tension >= 2.0
    from radial_membrane_ai.temporal import TemporalMembraneState
    t = TemporalMembraneState()
    t.consecutive_admissible_ticks = 6
    t.accumulated_tension = 3.0
    setattr(agent1.membrane, "temporal_state", t)
    success, _, _ = collective_sao_promote("local", agent1, None, "k", PolicyEnvelope(), 1.0, {}, 6)
    assert success is False

    # SAO local coherence < 0.7
    t.accumulated_tension = 0.5
    success, _, _ = collective_sao_promote("local", agent1, None, "k", PolicyEnvelope(), 0.5, {}, 6)
    assert success is False

    # 3. policy_envelope.py extra coverage
    # stability band other
    e1 = PolicyEnvelope(stability_required_band="yellow")
    e2 = PolicyEnvelope(stability_required_band="green")
    assert e1.intersect(e2).stability_required_band == "green"
    assert e2.intersect(e1).stability_required_band == "green"

    # temporal_window_min other
    e3 = PolicyEnvelope(temporal_window_min=10)
    e4 = PolicyEnvelope(temporal_window_min=20)
    assert e3.intersect(e4).temporal_window_min == 20

    # 4. mesh_correctness.py extra coverage
    # extreme closure ratio violation > 1.2 (elevate first string activation AND shrink boundary)
    agent1.membrane.strings[0].activation = 5.0
    agent1.membrane.strings[0].radius = 5.0
    agent1.boundary.radius_deviation = {i: -0.99 for i in range(1, 13)}
    step = CollectiveStepContext(participating_agent_ids={"agent_1"})
    checker = MeshCorrectness()
    is_correct, violations = checker.audit_correctness([agent1], [], step, PolicyEnvelope())
    assert is_correct is False
    assert any("Extreme closure ratio violation" in v for v in violations)

    # Reset
    agent1.membrane.strings[0].activation = 0.0
    agent1.membrane.strings[0].radius = 0.0
    agent1.boundary.radius_deviation = {i: 0.0 for i in range(1, 13)}

    # avg_t > 1.5 critical average tension violation
    t_state_critical = TemporalMembraneState()
    t_state_critical.tension_history.append(2.0)
    setattr(agent1.membrane, "temporal_state", t_state_critical)
    is_correct, violations = checker.audit_correctness([agent1], [], step, PolicyEnvelope())
    assert is_correct is False
    assert any("Critical average temporal tension" in v for v in violations)
    setattr(agent1.membrane, "temporal_state", None)

    # Cost factor > 3.0
    agent1.shard.cost_factor = 5.0
    is_correct, violations = checker.audit_correctness([agent1], [], step, PolicyEnvelope())
    assert is_correct is False
    assert any("Cost factor" in v for v in violations)
    agent1.shard.cost_factor = 1.0

    # Cluster tension > 1.5
    cluster = UFOCluster("cluster_1")
    cluster.tension_metric = 2.0
    step_c = CollectiveStepContext(involved_cluster_ids={"cluster_1"})
    is_correct, violations = checker.audit_correctness([], [cluster], step_c, PolicyEnvelope())
    assert is_correct is False
    assert any("Critical cluster tension" in v for v in violations)

    # Conflicting promotions
    cluster.semantic_memory.global_store["clash_key"] = MemoryRecord(
        key="clash_key",
        value="val",
        tags=set(),
        created_at=time.time(),
        updated_at=time.time(),
        origin_agent_id="some_other_id",  # origin agent mismatch
        policy_envelope=PolicyEnvelope()
    )
    is_correct, violations = checker.audit_correctness([], [cluster], step_c, PolicyEnvelope())
    assert is_correct is False

    # Global stability band violation
    step_red = CollectiveStepContext(stability_band="red")
    global_env_green = PolicyEnvelope(stability_required_band="green")
    is_correct, violations = checker.audit_correctness([], [], step_red, global_env_green)
    assert is_correct is False
    assert any("stability band is red" in v for v in violations)

    # Rollback quarantine fail_ids
    ledger = ResidualLedger()
    checker_l = MeshCorrectness(ledger)
    backup = checker_l.backup_state([agent1], [cluster])
    checker_l.rollback(backup, [agent1], [cluster], quarantine_ticks=2, failing_ids={"agent_1"})
    assert agent1.shard.state == ShardState.QUARANTINED
    assert getattr(agent1, "quarantine_timer", 0) == 2
