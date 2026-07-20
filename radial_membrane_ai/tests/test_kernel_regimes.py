"""
Comprehensive unit tests for the Kernel Regime Expansion Layer.
"""

from __future__ import annotations
import numpy as np
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.ufo_engine import SingleAgentEngine, MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.kernel_regimes.manager import RegimeManager
from radial_membrane_ai.kernel_regimes.regime import (
    KernelRegimeType,
    BALANCED_REGIME,
    DETERMINISTIC_REGIME,
    STOCHASTIC_REGIME,
    HIGH_CURVATURE_REGIME,
    ADVERSARIAL_REGIME,
    MULTI_PHASE_REGIME,
    balanced_closure_ratio_rule,
    balanced_tension_rule,
    balanced_capacity_rule,
    balanced_sao_rule,
    balanced_stability_rule,
    balanced_temporal_rule,
    deterministic_capacity_rule,
    deterministic_temporal_rule,
    stochastic_capacity_rule,
    high_curvature_closure_ratio_rule,
    high_curvature_tension_rule,
    adversarial_tension_rule,
    adversarial_capacity_rule,
    multiphase_closure_ratio_rule,
    multiphase_tension_rule,
    multiphase_capacity_rule,
    multiphase_sao_rule,
    multiphase_stability_rule,
    multiphase_temporal_rule,
    deterministic_closure_ratio_rule,
    stochastic_sao_rule,
    high_curvature_capacity_rule,
    high_curvature_sao_rule,
    adversarial_sao_rule,
    adversarial_stability_rule
)


def test_regime_manager_basic_get_set() -> None:
    manager = RegimeManager(KernelRegimeType.DETERMINISTIC)
    assert manager.get_global_regime().regime_type == KernelRegimeType.DETERMINISTIC

    # Set global regime
    manager.set_global_regime(KernelRegimeType.STOCHASTIC)
    assert manager.get_global_regime().regime_type == KernelRegimeType.STOCHASTIC

    # Get/set agent-specific regime
    assert manager.get_regime_for_agent("agent_1").regime_type == KernelRegimeType.STOCHASTIC
    manager.set_regime_for_agent("agent_1", KernelRegimeType.HIGH_CURVATURE)
    assert manager.get_regime_for_agent("agent_1").regime_type == KernelRegimeType.HIGH_CURVATURE
    assert manager.get_regime_for_agent("agent_2").regime_type == KernelRegimeType.STOCHASTIC

    # Get/set cluster-specific regime
    assert manager.get_regime_for_cluster("cluster_1").regime_type == KernelRegimeType.STOCHASTIC
    manager.set_regime_for_cluster("cluster_1", KernelRegimeType.ADVERSARIAL)
    assert manager.get_regime_for_cluster("cluster_1").regime_type == KernelRegimeType.ADVERSARIAL


def test_regime_switching_triggers() -> None:
    manager = RegimeManager()
    manager.automatic_switching_enabled = True

    # 1. Switch to MULTI_PHASE
    # T_bar > 0.8 AND coherence >= 0.6
    regime = manager.evaluate_switching_triggers(
        entity="agent_1",
        t_bar=0.9,
        coherence=0.65,
        stability_band="green",
        curvature=0.0
    )
    assert regime.regime_type == KernelRegimeType.MULTI_PHASE

    # 2. Switch to HIGH_CURVATURE
    # T_bar > 0.8 OR curvature > 0.7
    manager.set_regime_for_agent("agent_1", KernelRegimeType.DETERMINISTIC)
    regime = manager.evaluate_switching_triggers(
        entity="agent_1",
        t_bar=0.9,
        coherence=0.1,  # low coherence to avoid multi-phase
        stability_band="green",
        curvature=0.2
    )
    assert regime.regime_type == KernelRegimeType.HIGH_CURVATURE

    # 3. Switch to ADVERSARIAL
    # coherence < 0.4 OR stability_band in ("yellow", "red")
    manager.set_regime_for_agent("agent_1", KernelRegimeType.DETERMINISTIC)
    regime = manager.evaluate_switching_triggers(
        entity="agent_1",
        t_bar=0.3,
        coherence=0.3,
        stability_band="green"
    )
    assert regime.regime_type == KernelRegimeType.ADVERSARIAL

    # 4. Switch to STOCHASTIC
    # coherence < 0.5 OR cost_band == 1 OR cluster_tension > 0.7
    manager.set_regime_for_agent("agent_1", KernelRegimeType.DETERMINISTIC)
    regime = manager.evaluate_switching_triggers(
        entity="agent_1",
        t_bar=0.3,
        coherence=0.45,
        stability_band="green"
    )
    assert regime.regime_type == KernelRegimeType.STOCHASTIC

    # 5. Return to DETERMINISTIC
    # T_bar < 0.5 AND coherence >= 0.7 AND stability_band is green
    manager.set_regime_for_agent("agent_1", KernelRegimeType.STOCHASTIC)
    regime = manager.evaluate_switching_triggers(
        entity="agent_1",
        t_bar=0.2,
        coherence=0.8,
        stability_band="green"
    )
    assert regime.regime_type == KernelRegimeType.DETERMINISTIC


def test_deterministic_rules() -> None:
    engine = SingleAgentEngine()
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.DETERMINISTIC)

    # Trigger some tick updates
    for s in engine.membrane.strings:
        s.activation = 0.5

    # Monkeypatch angular_decomposition to return huge values (guaranteeing closure_ratio > 1.0)
    import radial_membrane_ai.admissibility as adm_mod
    original_decomp = adm_mod.angular_decomposition
    adm_mod.angular_decomposition = lambda *args, **kwargs: (10.0, 10.0)

    try:
        # Running tick should trigger deterministic admissibility violation clamp and soft rollback
        _ = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.8)
        assert any("Deterministic Admissibility" in m for m in engine.interventions)
    finally:
        adm_mod.angular_decomposition = original_decomp


def test_stochastic_rules() -> None:
    engine = SingleAgentEngine()
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.STOCHASTIC)

    # Let's verify that tension rule returns noise
    noise = STOCHASTIC_REGIME.tension_rule(None, None, engine)
    assert isinstance(noise, float)

    # Verification of stochastic tick adding noise
    engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    # Ensure activations have varied from 0 due to stochastic noise
    assert any(s.activation > 0 for s in engine.membrane.strings)


def test_high_curvature_rules() -> None:
    engine = SingleAgentEngine()
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.HIGH_CURVATURE)

    # Curvature rule multiplier
    rule_res = HIGH_CURVATURE_REGIME.temporal_rule(None, None, engine)
    assert rule_res["curvature_amplifier"] == 1.5
    assert rule_res["tension_accumulation_multiplier"] == 2.0

    # Capacity rule shrinks capacity
    # Populate tension history
    engine.membrane.temporal_state.tension_history.append(1.5)
    scale = HIGH_CURVATURE_REGIME.capacity_rule(None, None, engine)
    assert scale < 1.0

    # Running tick
    band = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.3)
    assert band in ("green", "yellow", "red")


def test_adversarial_rules() -> None:
    engine = SingleAgentEngine()
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.ADVERSARIAL)

    # Stability rule limits yellow/red thresholds
    rule_res = ADVERSARIAL_REGIME.stability_rule(None, None, engine)
    assert rule_res["threshold_multiplier"] == 0.8

    # Run tick in adversarial regime
    band = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.5)
    assert band in ("green", "yellow", "red")


def test_multiphase_rules() -> None:
    engine = SingleAgentEngine()
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.MULTI_PHASE)
    engine.regime_manager.evaluate_switching_triggers = lambda *args, **kwargs: MULTI_PHASE_REGIME  # type: ignore

    # Starts in deterministic phase
    phase = engine.regime_manager.get_active_multiphase_phase("single_agent")
    assert phase == "deterministic"

    # Evaluate temporal rule returns disable_hysteresis = True
    rule_res = MULTI_PHASE_REGIME.temporal_rule(None, None, engine)
    assert rule_res["disable_hysteresis"] is True

    # Tick 10 times to transition phase
    for _ in range(10):
        engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)

    phase = engine.regime_manager.get_active_multiphase_phase("single_agent")
    assert phase == "stochastic"


def test_engine_complete_rollback_and_quarantine() -> None:
    engine = SingleAgentEngine()
    # Force high energy
    engine.governor.compute_lyapunov_energy = lambda m: 10.0  # type: ignore

    # Under Deterministic regime, Red band is a stability violation
    engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.DETERMINISTIC)

    # Running tick should trigger stability rule violation rollback + 5-tick quarantine
    _ = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    assert any("Regime Stability Violation" in m for m in engine.interventions)
    assert any("quarantine" in m.lower() for m in engine.interventions)
    assert engine.quarantine_timer == 5

    # Ticking again while quarantined should skip and return "quarantined"
    band_q = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.1)
    assert band_q == "quarantined"


def test_multi_agent_regime_integrations() -> None:
    engine = MultiAgentEngine(n_agents=2)

    # Set different regimes for different agents
    engine.regime_manager.set_regime_for_agent("agent_1", KernelRegimeType.DETERMINISTIC)
    engine.regime_manager.set_regime_for_agent("agent_2", KernelRegimeType.STOCHASTIC)

    # Let's perform a tick
    band = engine.tick(task_value=0.5, excitation=np.ones(12) * 0.2)
    assert band in ("green", "yellow", "red")


def test_multi_cluster_regime_integrations() -> None:
    engine = MultiClusterEngine()
    _ = engine.create_cluster("cluster_1", "analytical")
    agent_1 = UFOAgent("agent_1")
    engine.assign_agent_to_cluster(agent_1, "cluster_1")

    # Set regime on cluster and agent
    engine.regime_manager.set_regime_for_cluster("cluster_1", KernelRegimeType.MULTI_PHASE)
    engine.regime_manager.set_regime_for_agent("agent_1", KernelRegimeType.STOCHASTIC)

    band = engine.tick(task_value=0.5, default_excitation=np.ones(12) * 0.2)
    assert band in ("green", "yellow", "red")


def test_coverage_gaps_in_kernel_regimes() -> None:
    # 1. regime.py balanced rules
    assert balanced_closure_ratio_rule(None, None, None)["admissible"] is True
    assert balanced_tension_rule(None, None, None) == 0.0
    balanced_capacity_rule(None, None, None)
    assert "short" in balanced_sao_rule(None, None, None)["allowed_ranges"]
    assert balanced_stability_rule(None, None, None)["valid"] is True
    assert balanced_temporal_rule(None, None, None)["disable_hysteresis"] is False

    # 2. deterministic / stochastic / curvature helpers
    deterministic_capacity_rule(None, None, None)
    assert deterministic_temporal_rule(None, None, None)["disable_hysteresis"] is True
    stochastic_capacity_rule(None, None, None)
    assert high_curvature_closure_ratio_rule(None, None, None)["admissible"] is True
    assert high_curvature_tension_rule(None, None, None) == 0.0
    assert adversarial_tension_rule(None, None, None) == 0.0
    adversarial_capacity_rule(None, None, None)

    # 3. multiphase deterministic / stochastic rules manually executed with mocks
    manager = RegimeManager()
    engine = SingleAgentEngine()
    engine.regime_manager = manager

    # Check phase deterministic
    manager.multiphase_states["global"] = ("deterministic", 3)
    assert multiphase_closure_ratio_rule(None, None, engine)["clamped_activations"] is None
    assert multiphase_tension_rule(None, None, engine) == 0.0
    multiphase_capacity_rule(None, None, engine)
    assert multiphase_stability_rule(None, None, engine)["valid"] is True
    assert multiphase_temporal_rule(None, None, engine)["disable_hysteresis"] is True

    # Check phase stochastic
    manager.multiphase_states["global"] = ("stochastic", 3)
    assert multiphase_closure_ratio_rule(None, None, engine)["admissible"] is True
    assert isinstance(multiphase_tension_rule(None, None, engine), float)
    multiphase_capacity_rule(None, None, engine)
    assert multiphase_stability_rule(None, None, engine)["valid"] is True
    assert multiphase_temporal_rule(None, None, engine)["activation_noise"] is not None

    # 4. multiphase_sao_rule
    res_sao_1 = multiphase_sao_rule(None, None, engine)
    assert len(res_sao_1["allowed_ranges"]) == 0  # not boundary

    # Mock boundary tick
    engine.tick_count = 10
    res_sao_2 = multiphase_sao_rule(None, None, engine)
    assert "short" in res_sao_2["allowed_ranges"]

    # 5. Manager coverage lines
    # set_regime_for_agent using KernelRegime directly instead of enum
    manager.set_regime_for_agent("agent_x", DETERMINISTIC_REGIME)
    assert manager.get_regime_for_agent("agent_x") == DETERMINISTIC_REGIME

    # set_regime_for_cluster using KernelRegime directly
    manager.set_regime_for_cluster("cluster_y", STOCHASTIC_REGIME)
    assert manager.get_regime_for_cluster("cluster_y") == STOCHASTIC_REGIME

    # _get_entity_key and getters for other objects
    assert manager._get_entity_key("cluster_abc") == "global"

    class DummyAgent:

        def __init__(self):
            self.agent_id = "dum"

    class DummyCluster:

        def __init__(self):
            self.cluster_id = "clu"

    dum_agent = DummyAgent()
    dum_cluster = DummyCluster()
    assert manager._get_entity_key(dum_agent) == "agent_dum"
    assert manager._get_entity_key(dum_cluster) == "cluster_clu"

    # evaluate_switching_triggers with automatic_switching_enabled = False (should immediately return)
    manager.automatic_switching_enabled = False
    assert manager.evaluate_switching_triggers(dum_agent, 0.0, 1.0, "green") == BALANCED_REGIME

    # _get_active_regime_for_entity / _set_active_regime_for_entity coverages
    manager.set_regime_for_cluster("cluster_clu", STOCHASTIC_REGIME)
    assert manager._get_active_regime_for_entity("cluster_clu") == STOCHASTIC_REGIME
    assert manager._get_active_regime_for_entity(dum_agent) == BALANCED_REGIME
    assert manager._get_active_regime_for_entity(dum_cluster) == BALANCED_REGIME

    manager._set_active_regime_for_entity(dum_agent, HIGH_CURVATURE_REGIME)
    assert manager.get_regime_for_agent("dum") == HIGH_CURVATURE_REGIME

    manager._set_active_regime_for_entity(dum_cluster, ADVERSARIAL_REGIME)
    assert manager.get_regime_for_cluster("clu") == ADVERSARIAL_REGIME

    manager._set_active_regime_for_entity("cluster_abc", MULTI_PHASE_REGIME)
    assert manager.get_regime_for_cluster("cluster_abc") == MULTI_PHASE_REGIME

    manager._set_active_regime_for_entity(None, BALANCED_REGIME)
    assert manager.get_global_regime() == BALANCED_REGIME

    # Tick multiphase state trigger transition T_bar > 0.8
    manager.multiphase_states["global"] = ("deterministic", 5)
    manager.tick_multiphase_state(None, 0.9)
    assert manager.multiphase_states["global"] == ("stochastic", 1)


def test_additional_coverage_gaps() -> None:
    # deterministic_closure_ratio_rule with agent=None
    res = deterministic_closure_ratio_rule(None, None, None)
    assert res["admissible"] is True

    # stochastic_sao_rule with cluster mocked
    class MockCluster:

        def compute_cluster_coherence(self):
            return 0.8

    assert "mid" in stochastic_sao_rule(None, MockCluster(), None)["allowed_ranges"]

    # high_curvature_capacity_rule with active_agent temporal state hasattr check
    class MockAgent:

        class MockMembrane:
            pass

        def __init__(self):
            self.membrane = MockAgent.MockMembrane()

    res_cap = high_curvature_capacity_rule(MockAgent(), None, None)
    assert res_cap == 1.0

    # high_curvature_sao_rule / adversarial_sao_rule / adversarial_stability_rule
    assert "long" in high_curvature_sao_rule(None, None, None)["allowed_ranges"]
    assert adversarial_sao_rule(None, None, None)["restrict_to_cluster"] is True
    assert adversarial_stability_rule(None, None, None)["threshold_multiplier"] == 0.8

    # multiphase_closure_ratio_rule and multiphase_tension_rule with cluster or none manager fallback
    assert multiphase_closure_ratio_rule(None, None, None)["admissible"] is True
    assert multiphase_tension_rule(None, None, None) == 0.0

    class MockClusterWithManager:

        def __init__(self):
            self.regime_manager = RegimeManager()

    assert multiphase_closure_ratio_rule(None, MockClusterWithManager(), None)["admissible"] is True
    assert multiphase_tension_rule(None, MockClusterWithManager(), None) == 0.0

    # multiphase_sao_rule with temporal_state ticks branch
    class MockEngineWithTemporal:

        class MockTemporal:

            def __init__(self):
                self.consecutive_admissible_ticks = 10

        def __init__(self):
            self.temporal_state = MockEngineWithTemporal.MockTemporal()

    assert "short" in multiphase_sao_rule(None, None, MockEngineWithTemporal())["allowed_ranges"]

    # manager.py line 100-101 fallback in _get_entity_key
    manager = RegimeManager()
    assert manager._get_entity_key(object()) == "global"

    # manager.py lines 148, 173 fallback in _get_active_regime_for_entity / _set_active_regime_for_entity
    assert manager._get_active_regime_for_entity(object()) == BALANCED_REGIME
    manager._set_active_regime_for_entity(object(), DETERMINISTIC_REGIME)
    assert manager.get_global_regime() == DETERMINISTIC_REGIME
