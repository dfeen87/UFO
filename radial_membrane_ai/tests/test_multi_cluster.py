"""
Comprehensive unit and integration tests for multi-cluster distributed mesh architecture.
"""

from __future__ import annotations
import numpy as np
import pytest

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.multi_agent.cross_channels import CrossClusterVChannel, ClusterPolicyEnvelope
from radial_membrane_ai.multi_agent.hierarchical_governance import (
    HierarchicalHolisticGovernor,
    HierarchicalSAOPromotion
)
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.semantic_memory.core import (
    AgentSemanticMemory,
    MeshSemanticMemory,
    MemoryRecord,
    write_memory
)
from radial_membrane_ai.semantic_memory.policy import MemoryPolicy, AdmissibilityContext, PolicyContext
from radial_membrane_ai.shard import ShardState


def test_cluster_creation_and_agent_assignment() -> None:
    # 1. Create cluster
    cluster = UFOCluster(cluster_id="cluster_alpha", role="analytical")
    assert cluster.cluster_id == "cluster_alpha"
    assert cluster.role == "analytical"
    assert len(cluster.agents) == 0

    # 2. Create agent and assign to cluster
    agent_1 = UFOAgent(agent_id="agent_1", cost_sensitivity=1.2)
    cluster.add_agent(agent_1)
    assert len(cluster.agents) == 1
    assert cluster.agents[0].agent_id == "agent_1"

    # Test removing agent
    removed = cluster.remove_agent("agent_1")
    assert removed is not None
    assert removed.agent_id == "agent_1"
    assert len(cluster.agents) == 0

    # Try removing non-existing
    assert cluster.remove_agent("agent_nonexistent") is None


def test_cluster_membrane_update() -> None:
    cluster = UFOCluster(cluster_id="cluster_beta", role="creative")
    agent_1 = UFOAgent(agent_id="agent_1")
    agent_2 = UFOAgent(agent_id="agent_2")

    # Set some activations
    agent_1.membrane.strings[0].activation = 0.8
    agent_2.membrane.strings[0].activation = 0.4
    agent_1.membrane.strings[0].tension = 0.6
    agent_2.membrane.strings[0].tension = 0.2

    cluster.add_agent(agent_1)
    cluster.add_agent(agent_2)

    # Update cluster-level membrane
    cluster.update_cluster_membrane()

    # The cluster string should be the average
    assert cluster.membrane.strings[0].activation == pytest.approx(0.6)
    assert cluster.membrane.strings[0].tension == pytest.approx(0.4)
    assert cluster.tension_metric > 0.0

    # Test empty cluster fallback
    cluster.remove_agent("agent_1")
    cluster.remove_agent("agent_2")
    cluster.update_cluster_membrane()
    assert cluster.membrane.strings[0].activation == 0.0
    assert cluster.tension_metric == 0.0


def test_cross_cluster_v_channel_routing() -> None:
    cluster_src = UFOCluster(cluster_id="src_cluster", role="analytical")
    cluster_tgt = UFOCluster(cluster_id="tgt_cluster", role="analytical")

    agent_src = UFOAgent("agent_src")
    agent_tgt = UFOAgent("agent_tgt")
    cluster_src.add_agent(agent_src)
    cluster_tgt.add_agent(agent_tgt)

    # Verify default policy envelope allows propagation
    chan = CrossClusterVChannel(
        source_cluster=cluster_src,
        target_cluster=cluster_tgt,
        channel_type="Type-W",
        coupling_strength=0.2,
        dampening=0.9
    )

    # Set source activation
    cluster_src.membrane.strings[0].activation = 0.8
    agent_tgt.membrane.strings[0].activation = 0.1

    # Propagate Type-W (workload)
    success = chan.propagate()
    assert success is True

    # Check target cluster and target agent activations increased
    # Target cluster string 0 activation should increase by 0.2 * 0.9 * 0.8 = 0.144
    assert cluster_tgt.membrane.strings[0].activation == pytest.approx(0.144)
    # Target agent activation should increase downwards
    assert agent_tgt.membrane.strings[0].activation > 0.1

    # Test Type-T (tension) propagation
    chan_t = CrossClusterVChannel(
        source_cluster=cluster_src,
        target_cluster=cluster_tgt,
        channel_type="Type-T",
        coupling_strength=0.2,
        dampening=0.9
    )
    cluster_src.membrane.strings[1].tension = 0.5
    agent_tgt.membrane.strings[1].tension = 0.1
    success_t = chan_t.propagate()
    assert success_t is True
    assert cluster_tgt.membrane.strings[1].tension == pytest.approx(0.09)
    assert agent_tgt.membrane.strings[1].tension > 0.1

    # Test Type-R (residuals) propagation
    chan_r = CrossClusterVChannel(
        source_cluster=cluster_src,
        target_cluster=cluster_tgt,
        channel_type="Type-R",
        coupling_strength=0.2,
        dampening=0.9
    )
    cluster_src.membrane.strings[2].cost = 0.5
    agent_tgt.membrane.strings[2].cost = 0.1
    success_r = chan_r.propagate()
    assert success_r is True
    assert cluster_tgt.membrane.strings[2].cost == pytest.approx(0.19)  # 0.1 baseline + 0.09
    assert agent_tgt.membrane.strings[2].cost > 0.1


def test_cross_cluster_policy_enforcement() -> None:
    cluster_src = UFOCluster(cluster_id="src_cluster", role="creative")
    cluster_tgt = UFOCluster(cluster_id="tgt_cluster", role="analytical")

    agent_src = UFOAgent("agent_src")
    agent_tgt = UFOAgent("agent_tgt")
    cluster_src.add_agent(agent_src)
    cluster_tgt.add_agent(agent_tgt)

    # 1. Enforce minimum compliance policy
    policy = ClusterPolicyEnvelope(min_compliance=0.9)
    chan = CrossClusterVChannel(cluster_src, cluster_tgt, policy_envelope=policy)

    # Make source non-compliant
    agent_src.shard.policy_compliance = 0.2
    assert chan.propagate() is False
    assert chan.rejection_count == 1
    # Check that rejection increases conflict tension in target stability curvature
    assert cluster_tgt.stability_curvature.tension > 0.0

    # 2. Restore compliance but fail trust
    agent_src.shard.policy_compliance = 0.95
    agent_tgt.shard.trust_score = 0.2
    policy_trust = ClusterPolicyEnvelope(min_trust=0.8)
    chan_trust = CrossClusterVChannel(cluster_src, cluster_tgt, policy_envelope=policy_trust)
    assert chan_trust.propagate() is False

    # 3. Role validation
    agent_tgt.shard.trust_score = 0.95
    policy_role = ClusterPolicyEnvelope(allowed_roles={"executive"})
    chan_role = CrossClusterVChannel(cluster_src, cluster_tgt, policy_envelope=policy_role)
    # target cluster role is analytical, which is not in allowed_roles
    assert chan_role.propagate() is False

    # 4. Check key/tag admissibility
    policy_semantic = ClusterPolicyEnvelope(blocked_prefixes=["classified_"], blocked_tags={"restricted"})
    assert policy_semantic.check_admissibility(cluster_src, cluster_tgt, key="public_key") is True
    assert policy_semantic.check_admissibility(cluster_src, cluster_tgt, key="classified_secret") is False
    assert policy_semantic.check_admissibility(cluster_src, cluster_tgt, tags={"restricted", "data"}) is False


def test_hierarchical_governor_field_computation() -> None:
    gov = HierarchicalHolisticGovernor()
    assert gov.compute_global_holistic_field([]) == 1.0

    cluster_1 = UFOCluster("c1", "analytical")
    cluster_2 = UFOCluster("c2", "creative")

    agent_1 = UFOAgent("a1")
    agent_2 = UFOAgent("a2")
    cluster_1.add_agent(agent_1)
    cluster_2.add_agent(agent_2)

    # Compute global holistic governor field
    h_global = gov.compute_global_holistic_field([cluster_1, cluster_2])
    assert isinstance(h_global, float)
    assert len(gov.h_global_history) == 1


def test_hierarchical_sao_promotion() -> None:
    # 1. Tier 1: Agent to Cluster Promotion
    cluster = UFOCluster("cluster_alpha")
    agent = UFOAgent("agent_1")
    cluster.add_agent(agent)

    # Setup local semantic memory record on agent
    from radial_membrane_ai.semantic_memory.integration import bind_to_agent
    bind_to_agent(agent)

    policy = MemoryPolicy()
    context = AdmissibilityContext(coherence=0.9, quarantined=False, cost_factor=1.0, stability_energy=0.0)
    write_result = write_memory(
        memory=agent.semantic_memory,
        key="local_key",
        value="secret_data",
        tags={"analytical"},
        policy=policy,
        context=context
    )
    assert write_result.success is True

    # Promote to Cluster
    sao = HierarchicalSAOPromotion()
    promo_res = sao.execute_agent_to_cluster_promotion(agent, cluster, "local_key")
    assert promo_res.success is True
    assert "local_key" in cluster.semantic_memory.global_store
    assert cluster.semantic_memory.global_store["local_key"].value == "secret_data"

    # Test promotion without semantic memory on agent
    agent_no_mem = UFOAgent("agent_no_mem")
    # Manually remove semantic memory attribute
    if hasattr(agent_no_mem, "semantic_memory"):
        delattr(agent_no_mem, "semantic_memory")
    promo_fail = sao.execute_agent_to_cluster_promotion(agent_no_mem, cluster, "local_key")
    assert promo_fail.success is False

    # 2. Tier 2: Cluster to Global Promotion
    global_mem = MeshSemanticMemory()
    promo_global_res = sao.execute_cluster_to_global_promotion(cluster, global_mem, "local_key")
    assert promo_global_res.success is True
    assert "local_key" in global_mem.global_store
    assert global_mem.global_store["local_key"].value == "secret_data"
    assert promo_global_res.p_sao >= 0.0

    # Test cluster metrics block
    agent.shard.policy_compliance = 0.1
    promo_block_res = sao.execute_cluster_to_global_promotion(cluster, global_mem, "local_key")
    assert promo_block_res.success is False

    # Test key not found in cluster memory
    promo_not_found = sao.execute_cluster_to_global_promotion(cluster, global_mem, "unknown_key")
    assert promo_not_found.success is False


def test_multi_cluster_engine_simulation_and_stability_bands() -> None:
    engine = MultiClusterEngine()

    # Create clusters
    c_analytical = engine.create_cluster(cluster_id="c_analytical", role="analytical")
    c_creative = engine.create_cluster(cluster_id="c_creative", role="creative")

    # Create agents and assign them
    a1 = UFOAgent("a1")
    a2 = UFOAgent("a2")
    engine.assign_agent_to_cluster(a1, "c_analytical")
    engine.assign_agent_to_cluster(a2, "c_creative")

    # Set up cross-cluster channel
    engine.add_cross_channel(
        source_cluster_id="c_analytical",
        target_cluster_id="c_creative",
        channel_type="Type-W",
        coupling_strength=0.1
    )

    # Run tick
    excite = np.ones(12, dtype=np.float64) * 0.5
    band = engine.tick(task_value=0.8, default_excitation=excite)

    # Check stability band and histories
    assert band in ("green", "yellow", "red")
    assert len(engine.h_global_history) == 1
    assert len(engine.c_global_history) == 1
    assert len(engine.global_band_history) == 1
    assert len(engine.interventions) > 0

    # Test querying stability
    metrics = engine.query_cluster_stability("c_analytical")
    assert metrics["cluster_id"] == "c_analytical"
    assert metrics["stability_band"] in ("green", "yellow", "red")
    assert "tension_metric" in metrics
    assert "coherence" in metrics


def test_multi_cluster_engine_yellow_and_red_band_interventions() -> None:
    engine = MultiClusterEngine()
    c_alpha = engine.create_cluster("c_alpha", "analytical")
    a1 = UFOAgent("a1")
    a2 = UFOAgent("a2")
    engine.assign_agent_to_cluster(a1, "c_alpha")
    engine.assign_agent_to_cluster(a2, "c_alpha")

    # 1. Trigger Yellow Band
    # Make coherence lower, and tension moderate
    a1.shard.policy_compliance = 0.5
    a2.shard.policy_compliance = 0.5
    a1.membrane.strings[0].tension = 0.8
    a2.membrane.strings[0].tension = 0.8

    excite = np.ones(12, dtype=np.float64) * 0.5
    band_yellow = engine.tick(task_value=0.8, default_excitation=excite)
    assert band_yellow == "yellow"

    # 2. Trigger Red Band
    # High tension and poor compliance
    a1.shard.policy_compliance = 0.1
    a2.shard.policy_compliance = 0.1
    c_alpha.stability_curvature.tension = 4.0

    band_red = engine.tick(task_value=0.8, default_excitation=excite)
    assert band_red == "red"


def test_multi_cluster_invalid_operations() -> None:
    engine = MultiClusterEngine()

    # Assign to non-existing cluster
    agent = UFOAgent("test_agent")
    with pytest.raises(ValueError, match="Cluster 'invalid_id' does not exist"):
        engine.assign_agent_to_cluster(agent, "invalid_id")

    # Add cross channel between non-existing clusters
    with pytest.raises(ValueError, match="Source cluster 'invalid_src' does not exist"):
        engine.add_cross_channel("invalid_src", "invalid_tgt")

    engine.create_cluster("valid_src")
    with pytest.raises(ValueError, match="Target cluster 'invalid_tgt' does not exist"):
        engine.add_cross_channel("valid_src", "invalid_tgt")

    # Query stability of non-existing cluster
    with pytest.raises(ValueError, match="Cluster 'invalid_id' does not exist"):
        engine.query_cluster_stability("invalid_id")


def test_coverage_gap_filler() -> None:
    # 1. Initialize cluster with agents directly
    agent = UFOAgent("a_init")
    cluster = UFOCluster(cluster_id="c_init", agents=[agent])
    assert len(cluster.agents) == 1

    # 2. compute_cluster_coherence on empty cluster
    cluster_empty = UFOCluster("c_empty")
    assert cluster_empty.compute_cluster_coherence() == 1.0

    # 3. Yellow band on cluster membrane update
    cluster_yellow = UFOCluster("c_yellow")
    a_yel = UFOAgent("a_yel")
    cluster_yellow.add_agent(a_yel)
    # Configure so coherence is ~0.5 and tension is ~1.5
    a_yel.shard.policy_compliance = 0.5
    cluster_yellow.stability_curvature.tension = 1.5
    cluster_yellow.update_cluster_membrane()
    assert cluster_yellow.stability_band == "yellow"

    # 3b. Green band on cluster membrane update
    from unittest.mock import patch
    with patch.object(cluster_yellow, "compute_cluster_coherence", return_value=0.9):
        cluster_yellow.stability_curvature.tension = 0.0
        cluster_yellow.update_cluster_membrane()
    assert cluster_yellow.stability_band == "green"

    # 4. Empty source and target clusters check_admissibility
    policy = ClusterPolicyEnvelope()
    assert policy.check_admissibility(cluster_empty, cluster_yellow) is False
    assert policy.check_admissibility(cluster_yellow, cluster_empty) is False

    # 5. HierarchicalHolisticGovernor empty cluster active agents check
    gov = HierarchicalHolisticGovernor()
    assert len(gov.h_global_history) == 0
    gov.compute_global_holistic_field([cluster_empty])

    # 6. SAO promotions with empty cluster active agents
    sao = HierarchicalSAOPromotion()
    assert len(cluster_empty.agents) == 0
    # Add dummy record to cluster_empty.semantic_memory.global_store
    from radial_membrane_ai.semantic_memory.core import MemoryRecord
    from radial_membrane_ai.semantic_memory.policy import MemoryPolicy
    import time
    rec = MemoryRecord(
        key="key", value="val", tags=set(), created_at=time.time(),
        updated_at=time.time(), origin_agent_id="c_empty", policy_envelope=MemoryPolicy()
    )
    cluster_empty.semantic_memory.global_store["key"] = rec
    global_mem = MeshSemanticMemory()
    res = sao.execute_cluster_to_global_promotion(cluster_empty, global_mem, "key")
    assert res.success is True

    # 7. Overwrite prevention check in execute_cluster_to_global_promotion (promote same key twice)
    res_overwrite = sao.execute_cluster_to_global_promotion(cluster_empty, global_mem, "key")
    assert res_overwrite.success is True

    # 8. Engine tick with empty clusters and quarantined agents and high global tension
    engine = MultiClusterEngine()
    excite = np.ones(12, dtype=np.float64) * 0.5
    # No clusters
    band_no_cluster = engine.tick(0.8, excite)
    assert band_no_cluster == "green"

    # Engine tick with quarantined agent
    engine_q = MultiClusterEngine()
    c_q = engine_q.create_cluster("c_q")
    a_q = UFOAgent("a_q")
    a_q.shard.state = ShardState.QUARANTINED
    engine_q.assign_agent_to_cluster(a_q, "c_q")
    engine_q.global_mesh_memory.curvature_state.tension = 2.0
    band_q = engine_q.tick(0.8, excite)
    # Re-verification that high global tension warning is logged
    tension_logs = [log for log in engine_q.interventions if "Global Shared Memory Tension Warning" in log]
    assert len(tension_logs) > 0
