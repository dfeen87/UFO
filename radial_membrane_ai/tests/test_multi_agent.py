"""
Comprehensive tests for multi-agent membrane coupling and mesh governance.
"""

from __future__ import annotations
import numpy as np
import pytest

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.coupling import InterAgentVChannel, GlobalHolisticGovernor
from radial_membrane_ai.multi_agent.governance import MultiAgentMeshGovernance
from radial_membrane_ai.multi_agent.simulation import MultiAgentSimulation
from radial_membrane_ai.shard import ShardState


def test_ufo_agent_initialization_and_step() -> None:
    # Initialize agent
    agent = UFOAgent("test_agent", cost_sensitivity=1.2, kernel_regime="analytical")
    assert agent.agent_id == "test_agent"
    assert agent.cost_sensitivity == 1.2
    assert agent.kernel_regime == "analytical"
    assert agent.shard.state == ShardState.IDLE

    # Run step on analytical regime
    excitation = np.ones(12, dtype=np.float64) * 0.5
    agent.step(task_value=0.8, excitation=excitation)

    # Coherences and history should be populated
    assert len(agent.coherence_history) == 1
    assert len(agent.activation_history) == 1
    assert len(agent.residual_history) == 1
    assert 0.0 <= agent.coherence_history[0] <= 1.0


def test_ufo_agent_creative_regime() -> None:
    agent = UFOAgent("creative_agent", cost_sensitivity=1.0, kernel_regime="creative")
    excitation = np.ones(12, dtype=np.float64) * 0.5
    agent.step(task_value=0.8, excitation=excitation)
    assert len(agent.activation_history) == 1


def test_ufo_agent_boundary_collapse() -> None:
    # Test block verdict (energy_threshold is extremely low, non-flat field)
    agent = UFOAgent("collapsing_agent_block", cost_sensitivity=5.0)
    agent.envelope.energy_threshold = 0.00001
    for s in agent.membrane.strings:
        s.activation = 0.0
    # Set peak on string 0 to create strong Fourier components (a, b)
    agent.membrane.strings[0].activation = 1.0

    for i in range(1, 13):
        agent.boundary.radius_deviation[i] = -0.9

    excitation = np.zeros(12, dtype=np.float64)
    excitation[0] = 1.0
    agent.step(task_value=0.8, excitation=excitation)
    assert agent.shard.state == ShardState.SATURATED

    # Test constrain verdict with cost_sensitivity=0.0 to prevent governor suppression
    agent_constrain = UFOAgent("collapsing_agent_constrain", cost_sensitivity=0.0)
    agent_constrain.envelope.energy_threshold = 20.0
    agent_constrain.boundary.base_radius = 0.3
    for s in agent_constrain.membrane.strings:
        s.activation = 0.0
    agent_constrain.membrane.strings[0].activation = 1.0

    agent_constrain.step(task_value=0.8, excitation=excitation)


def test_inter_agent_v_channel() -> None:
    agent_a = UFOAgent("agent_a")
    agent_b = UFOAgent("agent_b")

    # Set some activations
    agent_a.membrane.strings[0].activation = 0.8
    agent_b.membrane.strings[0].activation = 0.2

    # Type-W channel
    chan_w = InterAgentVChannel(agent_a, agent_b, channel_type="Type-W", coupling_strength=0.1)
    chan_w.propagate()
    assert agent_b.membrane.strings[0].activation > 0.2

    # Type-T channel
    agent_a.membrane.strings[1].tension = 0.5
    chan_t = InterAgentVChannel(agent_a, agent_b, channel_type="Type-T", coupling_strength=0.2)
    chan_t.propagate()
    assert agent_b.membrane.strings[1].tension > 0.0

    # Type-R channel
    agent_a.membrane.strings[2].cost = 0.6
    chan_r = InterAgentVChannel(agent_a, agent_b, channel_type="Type-R", coupling_strength=0.3)
    chan_r.propagate()
    assert agent_b.membrane.strings[2].cost > 0.1

    # Quarantined agent should not propagate
    agent_a.shard.state = ShardState.QUARANTINED
    old_tgt_act = agent_b.membrane.strings[0].activation
    chan_w.propagate()
    assert agent_b.membrane.strings[0].activation == old_tgt_act


def test_global_holistic_governor() -> None:
    agent_a = UFOAgent("agent_a")
    agent_b = UFOAgent("agent_b")
    gov = GlobalHolisticGovernor()

    # Empty list
    assert gov.compute_global_holistic_field([]) == 0.0

    # Non-empty
    h_field = gov.compute_global_holistic_field([agent_a, agent_b])
    assert isinstance(h_field, float)
    assert len(gov.h_history) == 1


def test_multi_agent_mesh_governance() -> None:
    # Test empty agents list
    gov_empty = MultiAgentMeshGovernance([])
    assert gov_empty.compute_mesh_coherence() == 1.0

    agent_a = UFOAgent("agent_a")
    agent_b = UFOAgent("agent_b")
    gov = MultiAgentMeshGovernance([agent_a, agent_b])

    # Compute mesh coherence
    coh = gov.compute_mesh_coherence()
    assert 0.0 <= coh <= 1.0

    # Test empty active agents in coherence
    agent_a.shard.state = ShardState.QUARANTINED
    agent_b.shard.state = ShardState.QUARANTINED
    coh_empty = gov.compute_mesh_coherence()
    assert 0.0 <= coh_empty <= 1.0

    # Restore state
    agent_a.shard.state = ShardState.ACTIVE
    agent_b.shard.state = ShardState.ACTIVE

    # SAO Promotion - Ascend case
    agent_a.membrane.strings[0].activation = 0.3
    agent_b.membrane.strings[0].activation = 0.3
    verdict, p_sao, proj = gov.execute_sao_promotion(agent_a, agent_b, shared_capacity_limit=0.8)
    assert verdict == "ascend"
    assert p_sao == 0.0

    # SAO Promotion - Constrain case
    agent_a.membrane.strings[0].activation = 0.9
    agent_b.membrane.strings[0].activation = 0.9
    verdict, p_sao, proj = gov.execute_sao_promotion(agent_a, agent_b, shared_capacity_limit=0.5)
    assert verdict == "constrain"
    assert p_sao > 0.15

    # SAO Promotion - Block case
    agent_a.membrane.strings[0].activation = 1.0
    agent_b.membrane.strings[0].activation = 1.0
    # Let's ensure high norm difference by using low limit
    verdict, p_sao, proj = gov.execute_sao_promotion(agent_a, agent_b, shared_capacity_limit=0.1)
    assert verdict == "block"
    assert p_sao > 0.4

    # Run mesh audit - normal
    agent_a.shard.state = ShardState.ACTIVE
    gov.run_mesh_audit()
    assert agent_a.shard.state != ShardState.QUARANTINED

    # Run mesh audit - trigger quarantine
    agent_a.residual_history.append(2.0)  # > 1.5 threshold
    gov.run_mesh_audit()
    assert agent_a.shard.state == ShardState.QUARANTINED
    assert len(gov.ledger.records) > 0


def test_multi_agent_simulation_scenarios() -> None:
    sim = MultiAgentSimulation(n_agents=3)
    assert len(sim.agents) == 3
    assert len(sim.channels) > 0

    # Configure agents to run in green band
    for a in sim.agents:
        a.shard.latency = 5.0
        a.shard.trust_score = 1.0
        a.shard.policy_compliance = 1.0

    # Scenario: cooperative
    sim.run(n_steps=2, scenario_type="cooperative")
    assert len(sim.h_hol_history) == 2
    assert len(sim.c_mesh_history) == 2

    # Verify green band log was hit
    green_logs = [log for log in sim.intervention_log if "Green Band" in log]
    assert len(green_logs) > 0

    # Scenario: competitive
    sim.run(n_steps=2, scenario_type="competitive")

    # Scenario: policy_tension
    sim.run(n_steps=2, scenario_type="policy_tension")


def test_simulation_no_coupling_channels() -> None:
    # Test initialization with 1 agent (should not create channels)
    sim = MultiAgentSimulation(n_agents=1)
    assert len(sim.channels) == 0


def test_simulation_interventions_yellow_band() -> None:
    sim = MultiAgentSimulation(n_agents=2)
    # Artificially drive down mesh coherence to yellow band
    sim.agents[0].shard.policy_compliance = 0.1
    sim.agents[1].shard.policy_compliance = 0.1
    sim.agents[0].shard.latency = 400.0
    sim.agents[1].shard.latency = 400.0

    sim.run_step("cooperative")
    # Interventions log should contain yellow band message
    yellow_interventions = [log for log in sim.intervention_log if "Yellow Band" in log]
    assert len(yellow_interventions) > 0


def test_simulation_interventions_red_band() -> None:
    sim = MultiAgentSimulation(n_agents=2)
    # Artificially drive down mesh coherence to red band
    # by making trust high enough to avoid quarantine but latency high
    sim.agents[0].shard.trust_score = 0.95
    sim.agents[0].shard.policy_compliance = 0.95
    sim.agents[0].shard.latency = 2000.0

    sim.agents[1].shard.trust_score = 0.95
    sim.agents[1].shard.policy_compliance = 0.95
    sim.agents[1].shard.latency = 2000.0

    sim.run_step("sao_block")
    # Interventions log should contain red band message
    red_interventions = [log for log in sim.intervention_log if "Red Band" in log]
    assert len(red_interventions) > 0

    # Check for SAO block
    block_logs = [log for log in sim.intervention_log if "SAO Promotion Blocked" in log]
    assert len(block_logs) > 0


def test_simulation_fallback() -> None:
    # Test fallback check (specifically having exactly one active agent)
    sim = MultiAgentSimulation(n_agents=2)
    sim.agents[0].shard.state = ShardState.ACTIVE
    sim.agents[1].shard.state = ShardState.QUARANTINED

    # Force red band coherence
    sim.agents[0].shard.latency = 5000.0

    sim.run_step("cooperative")
    # Fallback log should be populated
    fallback_logs = [log for log in sim.intervention_log if "Fallback triggered" in log]
    assert len(fallback_logs) > 0
