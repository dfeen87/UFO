"""
Comprehensive unit tests for the U.F.O. Governed Simulation Engine (ufo_engine).
Achieving 100% line coverage.
"""

from __future__ import annotations
import pytest
import numpy as np

from radial_membrane_ai.ufo_engine import (
    CostWeights,
    StabilityBandConfig,
    SingleAgentEngine,
    MultiAgentEngine
)
from radial_membrane_ai.ufo_engine.serialization import (
    ledger_to_json,
    export_simulation_results_to_json
)
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.multi_agent.agent import UFOAgent


def test_cost_weights_presets() -> None:
    # Test default/balanced
    balanced = CostWeights.get_preset("balanced")
    assert balanced.tokens == 0.05
    assert balanced.latency == 0.5

    # Test strict
    strict = CostWeights.get_preset("strict")
    assert strict.tokens == 0.2
    assert strict.latency == 0.8

    # Test exploratory
    exploratory = CostWeights.get_preset("exploratory")
    assert exploratory.tokens == 0.01
    assert exploratory.latency == 0.2

    # Test to_dict
    d = strict.to_dict()
    assert d["tokens"] == 0.2
    assert d["corrections"] == 0.9


def test_single_agent_engine_basic_tick() -> None:
    engine = SingleAgentEngine(
        cost_weights=CostWeights.get_preset("balanced"),
        band_config=StabilityBandConfig(v_green=0.01, v_red=0.1)
    )

    # Tick nominal
    band = engine.tick(task_value=0.5, excitation=np.zeros(12))
    assert band == "green"
    assert len(engine.v_history) == 1
    assert len(engine.band_history) == 1
    assert len(engine.cost_history) == 1
    assert len(engine.observable_cost_history) == 1
    assert len(engine.coherence_history) == 1
    assert len(engine.activation_history) == 2  # initial + 1 step

    # Excite strings heavily to drive up Lyapunov energy and trigger Yellow/Red bands
    band_escalated = engine.tick(task_value=0.9, excitation=np.ones(12) * 5.0)
    assert band_escalated in ("yellow", "red")


def test_single_agent_engine_run() -> None:
    engine = SingleAgentEngine()
    results = engine.run(
        n_steps=3,
        task_value_sequence=[0.8, 0.5],
        excitation_sequence=[np.ones(12) * 0.1, np.ones(12) * 0.5]
    )
    assert len(results.v_history) == 3
    assert len(results.band_history) == 3
    assert len(results.cost_history) == 3
    assert len(results.observable_cost_history) == 3


def test_single_agent_interventions_and_brim() -> None:
    engine = SingleAgentEngine(
        band_config=StabilityBandConfig(v_green=1.0, v_red=5.0)
    )

    # 1. Test Red stability band and Brim "block"
    # We mock energy to return 10.0 (above v_red=5.0) => Red band
    engine.governor.compute_lyapunov_energy = lambda m: 10.0  # type: ignore
    engine.envelope.evaluate_envelope = lambda m, b: ("block", {})  # type: ignore
    engine.sao_promotor.promote = lambda m, b, l: ("block", 0.5, {})  # type: ignore

    engine.tick(task_value=0.5, excitation=np.ones(12) * 1.0)

    assert any("Red Band" in m for m in engine.interventions)
    assert any("Brim Envelope Block" in m for m in engine.interventions)

    # 2. Test Yellow stability band and Brim "constrain"
    engine_yellow = SingleAgentEngine(
        band_config=StabilityBandConfig(v_green=1.0, v_red=5.0)
    )
    # Mock energy to return 3.0 (between v_green=1.0 and v_red=5.0) => Yellow band
    engine_yellow.governor.compute_lyapunov_energy = lambda m: 3.0  # type: ignore
    engine_yellow.envelope.evaluate_envelope = lambda m, b: ("constrain", {})  # type: ignore
    engine_yellow.sao_promotor.promote = lambda m, b, l: ("constrain", 0.2, {})  # type: ignore

    # Ensure activations are set to 0.9 before update so that newly calculated string costs exceed task_value
    for s in engine_yellow.membrane.strings:
        s.activation = 0.9

    # Set extremely low task_value to guarantee calculated string cost > task_value
    engine_yellow.tick(task_value=0.01, excitation=np.ones(12) * 1.0)

    assert any("Yellow Band" in m for m in engine_yellow.interventions)
    assert any("Brim Envelope Constrain" in m for m in engine_yellow.interventions)


def test_multi_agent_engine_basic_tick() -> None:
    engine = MultiAgentEngine(
        n_agents=2,
        cost_weights=CostWeights.get_preset("exploratory"),
        band_config=StabilityBandConfig(c_green=0.8, c_red=0.3)
    )

    # Mock high coherence to guarantee Green band branch is executed
    engine.mesh_governance.compute_mesh_coherence = lambda **kwargs: 0.95  # type: ignore

    # Tick normal
    band = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert band == "green"
    assert len(engine.h_hol_history) == 1
    assert len(engine.c_mesh_history) == 1
    assert len(engine.band_history) == 1
    assert len(engine.interventions) == 1
    assert len(engine.residual_history) == 1


def test_multi_agent_custom_agents_and_under_two_agents() -> None:
    custom_agents = [UFOAgent("custom_1")]
    # Initialize with less than 2 agents to hit edge cases
    engine = MultiAgentEngine(custom_agents=custom_agents)
    assert len(engine.channels) == 0

    # Running tick
    engine.tick(task_value=0.5, excitation=np.ones(12) * 0.2)
    assert len(engine.sao_events) == 0


def test_multi_agent_quarantine_and_fallback() -> None:
    # 2 agents
    agent_1 = UFOAgent("agent_1")
    agent_2 = UFOAgent("agent_2")
    engine = MultiAgentEngine(
        custom_agents=[agent_1, agent_2],
        band_config=StabilityBandConfig(c_green=0.9, c_red=0.4)
    )

    # 1. Force red band and quarantine agent_2
    agent_2.shard.trust_score = 0.1  # trigger quarantine in run_mesh_audit
    agent_2.residual_history.append(5.0)

    # Mock red band coherence
    engine.mesh_governance.compute_mesh_coherence = lambda **kwargs: 0.1  # type: ignore

    # Tick 1: triggers quarantine
    engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert agent_2.shard.state == ShardState.QUARANTINED

    # Tick 2: quarantined agent_2 is skipped (covers agent.shard.state == QUARANTINED branch)
    engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)

    fallback_msgs = [m for m in engine.interventions if "Fallback triggered" in m]
    assert len(fallback_msgs) > 0

    # 2. Test Yellow band in Multi-Agent tick
    engine_yellow = MultiAgentEngine(
        custom_agents=[UFOAgent("a1"), UFOAgent("a2")],
        band_config=StabilityBandConfig(c_green=0.8, c_red=0.4)
    )
    engine_yellow.mesh_governance.compute_mesh_coherence = lambda **kwargs: 0.6  # type: ignore
    engine_yellow.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert any("Yellow Band" in m for m in engine_yellow.interventions)


def test_multi_agent_engine_run() -> None:
    engine = MultiAgentEngine(n_agents=3)
    results = engine.run(
        n_steps=2,
        task_value_sequence=[0.7],
        excitation_sequence=[np.ones(12) * 0.5]
    )
    assert len(results.h_hol_history) == 2
    assert len(results.c_mesh_history) == 2
    assert len(results.band_history) == 2


def test_serialization_and_exports(tmp_path) -> None:
    # Test ledger exporting
    engine = SingleAgentEngine()
    # Manually log a failure to test serialization logic
    engine.ledger.log_failure(
        record_id="test_record_1",
        error_type="sao_test_failure",
        shard_id="single_agent",
        severity="high",
        details={"p_sao": 0.55}
    )

    # Confirm ledger records exist
    assert len(engine.ledger.records) > 0

    ledger_file = tmp_path / "subdir" / "ledger.json"  # Test directory creation
    ledger_to_json(engine.ledger, str(ledger_file))
    assert ledger_file.exists()

    # Test arbitrary results exporting
    results_file = tmp_path / "subdir_2" / "results.json"
    dummy_results = {
        "v_history": [0.1, 0.2],
        "activations": np.array([0.5, 0.5]),
        "some_object": object()  # force default serializer to convert object to string
    }
    export_simulation_results_to_json(dummy_results, str(results_file))
    assert results_file.exists()


def test_stability_metrics_post_tick_violations() -> None:
    """
    Tests that post-tick verification on stability metrics (Lyapunov energy,
    tension, stiffness, curvature, deviation) correctly raises GovernanceError on violations.
    """
    from radial_membrane_ai.exceptions import GovernanceError

    # 1. Test Lyapunov energy violation
    engine_sa = SingleAgentEngine()
    engine_sa.membrane.strings[0].tension = 20.0  # Exceeds hard_energy_limit of 15.0 through computed energy
    with pytest.raises(GovernanceError, match="Lyapunov energy.*exceeded hard limit"):
        engine_sa.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # 2. Test Tension violation
    engine_sa2 = SingleAgentEngine()
    if hasattr(engine_sa2.membrane, "temporal_state") and engine_sa2.membrane.temporal_state:
        engine_sa2.membrane.temporal_state.accumulated_tension = 100.0  # Exceeds hard_tension_limit of 10.0
    with pytest.raises(GovernanceError, match="Temporal tension.*exceeded hard limit"):
        engine_sa2.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # 3. Test Stiffness violation
    engine_sa3 = SingleAgentEngine()
    # Artificially modify a string stiffness to exceed 2.0
    engine_sa3.membrane.strings[0].stiffness = 5.0
    with pytest.raises(GovernanceError, match="Stiffness.*exceeded hard limit"):
        engine_sa3.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # 4. Test Curvature violation
    engine_sa4 = SingleAgentEngine()
    setattr(engine_sa4.boundary, "curvature", lambda theta, **kwargs: 100.0)
    with pytest.raises(GovernanceError, match="Boundary curvature.*exceeded hard limit"):
        engine_sa4.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # 5. Test Radius Deviation violation
    engine_sa5 = SingleAgentEngine()
    setattr(engine_sa5.boundary, "update_boundary", lambda *args, **kwargs: None)
    engine_sa5.boundary.radius_deviation[1] = 100.0
    with pytest.raises(GovernanceError, match="Radius deviation.*exceeded hard limit"):
        engine_sa5.tick(task_value=0.5, excitation=np.ones(12) * 0.5)
