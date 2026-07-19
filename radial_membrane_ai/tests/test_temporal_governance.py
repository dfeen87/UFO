"""
Unit and integration tests for the UFO Temporal Governance Layer.
Validates hysteresis, tension accumulation, temporal SAO promotion gates,
time-weighted admissibility, drift/decay/recovery, and multi-cluster temporal coherence.
"""

from __future__ import annotations
import math
import numpy as np

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.admissibility import dynamic_capacity_boundary_temporal
from radial_membrane_ai.temporal import TemporalMembraneState
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.ufo_engine.config import CostWeights, StabilityBandConfig


def test_temporal_membrane_state_basic_operations() -> None:
    """Test the basic queue, exponential decaying averages, and decay of TemporalMembraneState."""
    t_state = TemporalMembraneState(short_horizon=3, long_horizon=5)

    assert t_state.consecutive_admissible_ticks == 0
    assert len(t_state.curvature_history) == 0

    # Test short properties when length < short_horizon
    assert len(t_state.short_tension_history) == 0
    assert len(t_state.short_admissibility_history) == 0

    # Feed admissible tick
    t_state.update_tick_history(max_curvature=0.2, max_tension=0.5, max_closure_ratio=0.8)
    assert t_state.consecutive_admissible_ticks == 1
    assert t_state.prior_curvature == 0.2
    assert t_state.prior_tension == 0.5

    assert len(t_state.short_tension_history) == 1
    assert len(t_state.short_admissibility_history) == 1

    # Feed non-admissible tick
    t_state.update_tick_history(max_curvature=0.4, max_tension=1.2, max_closure_ratio=1.5)
    assert t_state.consecutive_admissible_ticks == 0

    # Fill beyond long_horizon
    for _ in range(10):
        t_state.update_tick_history(max_curvature=0.1, max_tension=0.2, max_closure_ratio=0.5)

    assert len(t_state.curvature_history) == 5
    assert t_state.consecutive_admissible_ticks == 10

    # Test short properties when length >= short_horizon
    assert len(t_state.short_tension_history) == 3
    assert len(t_state.short_admissibility_history) == 3

    # Compute averages
    avg_i, avg_t = t_state.compute_exponential_decay_averages(eta=0.7)
    assert 0.0 < avg_i < 1.0
    assert 0.0 < avg_t < 1.0

    # Test tension accumulation
    accumulated = t_state.update_tension_accumulation(
        v_channel_load=1.5,
        cost_taxonomy_contrib=2.0,
        sao_promotions_intensity=0.5,
        mem_writes_norm=0.1,
        gamma=0.8
    )
    assert accumulated > 0.0

    # Test tension decay
    prior_acc = t_state.accumulated_tension
    t_state.decay_tension(low_load=True, rho=0.7)
    assert t_state.accumulated_tension == prior_acc * 0.7
    assert t_state.prior_tension == 0.2 * 0.7

    # Test recovery dynamics
    mu_rec = t_state.update_recovery_dynamics(is_green=True, mu=0.1)
    assert mu_rec == 0.1  # Not 10 ticks green yet

    # Force 10 ticks green
    for _ in range(10):
        t_state.update_recovery_dynamics(is_green=True, mu=0.1)
    mu_rec_green = t_state.update_recovery_dynamics(is_green=True, mu=0.1)
    assert mu_rec_green == 0.15


def test_geometric_hysteresis_and_capacity_scaling() -> None:
    """Test dynamic capacity boundary calculations under temporal hysteresis and tension."""
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()

    # Default state (no history/tension)
    c_base = boundary.get_radius(0.0)
    c_temp = dynamic_capacity_boundary_temporal(boundary, 0.0, membrane)
    assert math.isclose(c_base, c_temp)

    # Populate high tension history to trigger capacity contraction
    t_state = membrane.temporal_state
    for _ in range(10):
        t_state.update_tick_history(max_curvature=0.5, max_tension=2.0, max_closure_ratio=1.2)

    c_contracted = dynamic_capacity_boundary_temporal(boundary, 0.0, membrane)
    # The capacity should be significantly contracted under high tension history
    assert c_contracted < c_base


def test_temporal_sao_promotion_gates() -> None:
    """Test short, mid, and long range SAO promotion gates under temporal admissibility rules."""
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()
    promotor = SAOPromotor(promotion_threshold=0.3)

    # Clean alignment to make base SAO successful
    for s in membrane.strings:
        s.activation = 0.8

    # Case 1: Less than 5 ticks history -> Defaults to original promotion (no temporal restrictions enforced)
    verdict_HG, _, _ = promotor.promote(membrane, boundary, "Holistic Governor")
    assert verdict_HG == "ascend"

    # Case 2: 5 ticks of high tension history -> Fails temporal gate criteria
    t_state = membrane.temporal_state
    for _ in range(5):
        t_state.update_tick_history(max_curvature=0.5, max_tension=2.5, max_closure_ratio=0.8)

    # Holistic Governor target (Short-range gate: requires consec >= 5 and avg_t < 2.0)
    # Since avg_t is 2.5 (>= 2.0), it should restrict/admit
    verdict_HG_restricted, _, _ = promotor.promote(membrane, boundary, "Holistic Governor")
    assert verdict_HG_restricted == "admit"

    # Case 3: 5 ticks of low tension history (consec = 5, avg_t = 0.2)
    t_state.consecutive_admissible_ticks = 5
    t_state.tension_history.clear()
    for _ in range(5):
        t_state.tension_history.append(0.2)

    verdict_HG_ascend, _, _ = promotor.promote(membrane, boundary, "Holistic Governor")
    assert verdict_HG_ascend == "ascend"

    # Case 4: Mid-range target: "semantic memory" (Requires consec >= 20, avg_t < 1.0)
    # With only 5 ticks history, it should fail and constrain
    for _ in range(20):
        t_state.update_tick_history(max_curvature=0.2, max_tension=0.1, max_closure_ratio=0.5)
    t_state.consecutive_admissible_ticks = 10  # Less than 20
    verdict_SM_fail, _, _ = promotor.promote(membrane, boundary, "semantic memory")
    assert verdict_SM_fail == "constrain"

    t_state.consecutive_admissible_ticks = 20
    verdict_SM_pass, _, _ = promotor.promote(membrane, boundary, "semantic memory")
    assert verdict_SM_pass == "ascend"

    # Case 5: Long-range target: "identity core" (Requires consec >= 50, avg_t < 1.0)
    t_state.consecutive_admissible_ticks = 40  # Less than 50
    verdict_IC_fail, _, _ = promotor.promote(membrane, boundary, "identity core")
    assert verdict_IC_fail == "block"

    t_state.consecutive_admissible_ticks = 50
    verdict_IC_pass, _, _ = promotor.promote(membrane, boundary, "identity core")
    assert verdict_IC_pass == "ascend"


def test_curvature_drift_dynamics() -> None:
    """Test physical boundary curvature drift back to neutral state."""
    boundary = BoundaryGeometry()
    t_state = TemporalMembraneState()

    # Deform boundary
    boundary.radius_deviation[1] = 1.5
    boundary.radius_deviation[2] = -0.8

    # Apply drift
    t_state.apply_curvature_drift(boundary, mu=0.1)
    assert math.isclose(boundary.radius_deviation[1], 1.5 * 0.9)
    assert math.isclose(boundary.radius_deviation[2], -0.8 * 0.9)


def test_single_agent_engine_temporal_integration() -> None:
    """Test SingleAgentEngine integration, effective energy modulation, and decay."""
    engine = SingleAgentEngine(
        cost_weights=CostWeights.get_preset("balanced"),
        band_config=StabilityBandConfig(v_green=0.01, v_red=0.1)
    )

    # Let's run 6 ticks to build a temporal history past the startup phase (>= 5 ticks)
    for _ in range(6):
        engine.tick(task_value=0.5, excitation=np.zeros(12))

    # Verify temporal state is updated and contains history
    t_state = engine.membrane.temporal_state
    assert len(t_state.tension_history) >= 5
    assert t_state.consecutive_admissible_ticks >= 5

    # Run tick with high excitation to build tension
    engine.tick(task_value=0.9, excitation=np.ones(12) * 5.0)
    assert t_state.accumulated_tension > 0.0


def test_multi_agent_engine_temporal_coherence() -> None:
    """Test MultiAgentEngine temporal updates, aggregation, and effective coherence scaling."""
    engine = MultiAgentEngine(n_agents=3)

    # Ensure all agents have a temporal state
    for agent in engine.agents:
        assert getattr(agent.membrane, "temporal_state", None) is not None

    # Tick engine 6 times
    for _ in range(6):
        engine.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # Verify global engine-level temporal state is aggregated and updated
    assert len(engine.temporal_state.tension_history) >= 5
    assert engine.temporal_state.accumulated_tension >= 0.0


def test_multi_cluster_engine_coherence_aggregation() -> None:
    """Test MultiClusterEngine cluster-level and global-level temporal aggregations."""
    engine = MultiClusterEngine()

    c1 = engine.create_cluster("cluster_1", role="analytical")
    engine.create_cluster("cluster_2", role="creative")

    from radial_membrane_ai.multi_agent.agent import UFOAgent
    a1 = UFOAgent("agent_1_1")
    a2 = UFOAgent("agent_2_1")

    engine.assign_agent_to_cluster(a1, "cluster_1")
    engine.assign_agent_to_cluster(a2, "cluster_2")

    # Verify cluster and agents have temporal state ownership
    assert getattr(c1.membrane, "temporal_state", None) is not None
    assert getattr(a1.membrane, "temporal_state", None) is not None

    # Tick the multi-cluster engine 6 times
    for _ in range(6):
        engine.tick(task_value=0.6, default_excitation=np.ones(12) * 0.4)

    # Verify engine-level global temporal state aggregates cluster metrics
    assert len(engine.temporal_state.tension_history) >= 5
    assert getattr(c1.membrane.temporal_state, "accumulated_tension") is not None
