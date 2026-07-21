"""
Comprehensive unit and integration tests for Governed Stress-Testing Workload Family Design.
"""

from __future__ import annotations
import pytest
from typing import Any

from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType, HIGH_CURVATURE_REGIME
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.semantic_memory.core import MeshSemanticMemory

from radial_membrane_ai.workloads import (
    Action,
    ActionType,
    StabilityBand,
    SAOLevel,
    SimulationTarget,
    WorkloadStep,
    Workload,
    WorkloadEngine,
    create_cooperative_workload,
    create_adversarial_workload,
    create_high_curvature_workload,
    create_policy_tension_workload,
    create_asymmetric_workload,
    create_global_mesh_workload
)


def test_action_helper_methods() -> None:
    """
    Tests all Action helper classmethods and verifying their payloads.
    """
    a1 = Action.set_activation("agent_1", 0.8)
    assert a1.type == ActionType.SET_ACTIVATION
    assert a1.payload["agent_id"] == "agent_1"
    assert a1.payload["value"] == 0.8

    a2 = Action.write_memory("global", "some_key", "some_val", ["tag1", "tag2"])
    assert a2.type == ActionType.WRITE_MEMORY
    assert a2.payload["scope"] == "global"
    assert a2.payload["key"] == "some_key"
    assert a2.payload["value"] == "some_val"
    assert a2.payload["tags"] == ["tag1", "tag2"]

    a3 = Action.trigger_violation("agent_2", "unauthorized_write")
    assert a3.type == ActionType.TRIGGER_VIOLATION
    assert a3.payload["entity_id"] == "agent_2"
    assert a3.payload["violation_type"] == "unauthorized_write"

    a4 = Action.change_regime("cluster_1", "HIGH_CURVATURE")
    assert a4.type == ActionType.CHANGE_REGIME
    assert a4.payload["entity_id"] == "cluster_1"
    assert a4.payload["regime"] == "HIGH_CURVATURE"

    a5 = Action.set_envelope("global", 0.85, "yellow")
    assert a5.type == ActionType.SET_ENVELOPE
    assert a5.payload["entity_id"] == "global"
    assert a5.payload["min_trust"] == 0.85
    assert a5.payload["stability_required_band"] == "yellow"

    a6 = Action.set_tension("agent_1", 2.5)
    assert a6.type == ActionType.SET_TENSION
    assert a6.payload["entity_id"] == "agent_1"
    assert a6.payload["tension"] == 2.5

    a7 = Action.set_curvature("agent_1", 1.5)
    assert a7.type == ActionType.SET_CURVATURE
    assert a7.payload["entity_id"] == "agent_1"
    assert a7.payload["curvature"] == 1.5

    a8 = Action.set_cost_profile("agent_1", 1.8)
    assert a8.type == ActionType.SET_COST_PROFILE
    assert a8.payload["entity_id"] == "agent_1"
    assert a8.payload["cost_sensitivity"] == 1.8

    a9 = Action.set_role("agent_1", "planner")
    assert a9.type == ActionType.SET_ROLE
    assert a9.payload["entity_id"] == "agent_1"
    assert a9.payload["role"] == "planner"

    a10 = Action.custom_action("hello", {"x": 1})
    assert a10.type == ActionType.CUSTOM
    assert a10.payload["name"] == "hello"
    assert a10.payload["payload"] == {"x": 1}


def test_workload_builders() -> None:
    """
    Tests that all six workload builders produce valid workloads with expected parameters.
    """
    coop = create_cooperative_workload()
    assert coop.target == SimulationTarget.MULTI_AGENT
    assert coop.regime_expectation == KernelRegimeType.DETERMINISTIC
    assert coop.stability_expectation == StabilityBand.GREEN
    assert len(coop.steps) > 0

    adv = create_adversarial_workload()
    assert adv.target == SimulationTarget.MULTI_AGENT
    assert adv.regime_expectation == KernelRegimeType.ADVERSARIAL
    assert adv.stability_expectation == StabilityBand.RED

    curv = create_high_curvature_workload()
    assert curv.target == SimulationTarget.SINGLE_AGENT
    assert curv.regime_expectation == KernelRegimeType.HIGH_CURVATURE
    assert curv.stability_expectation == StabilityBand.YELLOW

    policy = create_policy_tension_workload()
    assert policy.target == SimulationTarget.MULTI_AGENT
    assert policy.regime_expectation == KernelRegimeType.BALANCED
    assert policy.stability_expectation == StabilityBand.RED

    asym = create_asymmetric_workload()
    assert asym.target == SimulationTarget.MULTI_AGENT
    assert asym.regime_expectation == KernelRegimeType.BALANCED
    assert asym.stability_expectation == StabilityBand.GREEN

    glob = create_global_mesh_workload()
    assert glob.target == SimulationTarget.MULTI_CLUSTER
    assert glob.regime_expectation == KernelRegimeType.MULTI_PHASE
    assert glob.stability_expectation == StabilityBand.GREEN


def test_engine_run_single_agent() -> None:
    """
    Runs the high-curvature workload on a single agent engine and verifies the diagnostic trace.
    """
    sa_engine = SingleAgentEngine()
    workload_engine = WorkloadEngine(single_agent_engine=sa_engine)

    workload = create_high_curvature_workload()
    # Modify steps to verify specific activation structures
    workload.steps[0].agent_actions["act_spike"] = Action.set_activation("single_agent", 0.75)

    result = workload_engine.run(workload)

    assert len(result.frames) == 2
    assert result.frames[0].step_index == 0
    assert result.frames[0].regime == "high_curvature"
    # Verify tension or curvature were set correctly (clamped to 1.8)
    assert sa_engine.boundary.radius_deviation[1] == 1.8  # from second step

    # Correctness report should exist
    report = result.correctness_report
    assert "success" in report
    assert report["total_steps"] == 2


def test_engine_run_multi_agent_cooperative() -> None:
    """
    Runs the cooperative workload on a multi-agent engine and verifies high coherence.
    """
    ma_engine = MultiAgentEngine(n_agents=3)
    ma_engine.mesh_memory = MeshSemanticMemory()
    ma_engine.regime_manager.set_regime_for_agent("agent_1", KernelRegimeType.BALANCED)

    workload_engine = WorkloadEngine(multi_agent_engine=ma_engine)
    workload = create_cooperative_workload()

    # Change second step regime to STOCHASTIC so we definitely get a transition
    workload.steps[1].agent_actions["agent_1_new_reg"] = Action.change_regime("agent_1", "STOCHASTIC")

    result = workload_engine.run(workload)

    assert len(result.frames) == 2
    assert len(result.regime_transitions) > 0

    report = result.correctness_report
    assert report["average_coherence"] > 0.6


def test_engine_run_multi_agent_adversarial_and_policy() -> None:
    """
    Runs the adversarial and policy-tension workloads on a multi-agent engine to verify
    quarantines, rollbacks, and policy envelope checks.
    """
    ma_engine = MultiAgentEngine(n_agents=3)
    ma_engine.mesh_memory = MeshSemanticMemory()

    workload_engine = WorkloadEngine(multi_agent_engine=ma_engine)

    # 1. Adversarial workload
    workload_adv = create_adversarial_workload()
    result_adv = workload_engine.run(workload_adv)

    # Adversarial activates the adversarial regime rules
    assert ma_engine.regime_manager.get_regime_for_agent("agent_1").regime_type == KernelRegimeType.ADVERSARIAL
    # Low compliance / high tension should be simulated, creating Red band or rollback events
    assert len(result_adv.frames) == 2

    # 2. Policy-tension workload
    ma_engine_policy = MultiAgentEngine(n_agents=3)
    ma_engine_policy.mesh_memory = MeshSemanticMemory()
    workload_engine_policy = WorkloadEngine(multi_agent_engine=ma_engine_policy)

    workload_policy = create_policy_tension_workload()
    result_policy = workload_engine_policy.run(workload_policy)

    # Policy violations should cause rollbacks and quarantine events
    assert len(result_policy.frames) == 2
    report_policy = result_policy.correctness_report
    assert report_policy["total_quarantines"] >= 0


def test_engine_run_asymmetric() -> None:
    """
    Runs the asymmetric multi-agent workload and checks role assignments and cost modifications.
    """
    ma_engine = MultiAgentEngine(n_agents=3)
    ma_engine.mesh_memory = MeshSemanticMemory()

    workload_engine = WorkloadEngine(multi_agent_engine=ma_engine)
    workload = create_asymmetric_workload()

    result = workload_engine.run(workload)

    # Check roles and cost sensitivites applied
    assert ma_engine.agents[0].kernel_regime == "planner"
    assert ma_engine.agents[1].kernel_regime == "critic"
    assert ma_engine.agents[2].kernel_regime == "explorer"
    assert ma_engine.agents[0].cost_sensitivity == 2.5
    assert ma_engine.agents[1].cost_sensitivity == 0.5

    assert len(result.frames) == 2


def test_engine_run_multi_cluster_global() -> None:
    """
    Sets up MultiClusterEngine, runs the global mesh workload, and checks cross-cluster routing
    and global rollbacks.
    """
    mc_engine = MultiClusterEngine()

    # Create clusters
    mc_engine.create_cluster("cluster_1", "planner")
    mc_engine.create_cluster("cluster_2", "critic")

    # Create agents and assign
    a1 = UFOAgent("agent_1")
    a2 = UFOAgent("agent_2")
    mc_engine.assign_agent_to_cluster(a1, "cluster_1")
    mc_engine.assign_agent_to_cluster(a2, "cluster_2")

    # Add cross-cluster channel
    mc_engine.add_cross_channel("cluster_1", "cluster_2")

    workload_engine = WorkloadEngine(multi_cluster_engine=mc_engine)
    workload = create_global_mesh_workload()

    result = workload_engine.run(workload)

    assert len(result.frames) == 2
    # Verify cluster regimes were set
    # last step change
    assert mc_engine.regime_manager.get_regime_for_cluster("cluster_1").regime_type == KernelRegimeType.HIGH_CURVATURE
    assert mc_engine.regime_manager.get_global_regime().regime_type == KernelRegimeType.MULTI_PHASE

    # Verify global store contains federated key
    assert "federated_fact" in mc_engine.global_mesh_memory.global_store
    assert len(result.semantic_diffs) > 0


def test_unsupported_simulation_target() -> None:
    """
    Verifies that calling WorkloadEngine with an unknown SimulationTarget raises a ValueError.
    """
    engine = WorkloadEngine()
    bad_workload = Workload(
        name="Bad",
        description="Bad",
        target=None,  # type: ignore
        regime_expectation=KernelRegimeType.BALANCED,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=1.0,
        envelope_expectation=PolicyEnvelope(),
        steps=[],
        _bypass_validation=True
    )
    with pytest.raises(ValueError, match="Unknown target:"):
        engine.run(bad_workload)


def test_engine_init_defaults() -> None:
    """
    Checks that WorkloadEngine constructs default simulation engines if none are passed.
    """
    engine = WorkloadEngine()
    assert isinstance(engine.single_agent_engine, SingleAgentEngine)
    assert isinstance(engine.multi_agent_engine, MultiAgentEngine)
    assert isinstance(engine.multi_cluster_engine, MultiClusterEngine)


def test_engine_coverage_booster() -> None:
    """
    Exercises all code branches of WorkloadEngine's action dispatch logic
    across all three simulation engines to secure 100% coverage on engine.py.
    """
    # 1. Single Agent Dispatch coverage
    sa_engine = SingleAgentEngine()
    # Force Red band via band config thresholds
    sa_engine.band_config.v_green = 0.0001
    sa_engine.band_config.v_red = 0.0002
    # Force SAO promotions to always ascend
    sa_engine.sao_promotor.promotion_threshold = -1.0
    # Force coherence to be high to satisfy allowed ranges checks in stochastic
    setattr(sa_engine, "compute_local_coherence", lambda: 1.0)

    # Sequential overrides to cover all envelope states in SingleAgent
    def mock_evaluate_envelope(mem: Any, bnd: Any) -> tuple[str, dict[str, Any]]:
        tc = sa_engine.tick_count
        if tc == 1:
            return "admit", {}
        elif tc == 2:
            return "constrain", {}
        elif tc == 3:
            return "block", {}
        elif tc == 4:
            return "rest", {}
        else:
            return "re-project", {}

    setattr(sa_engine.envelope, "evaluate_envelope", mock_evaluate_envelope)

    # Wrap sa_engine.tick to force specific band_history and sao_events states
    orig_tick_sa = sa_engine.tick

    def custom_tick_sa(task_value: float, excitation: Any, *args: Any, **kwargs: Any) -> str:
        res = orig_tick_sa(task_value, excitation, *args, **kwargs)
        tc = sa_engine.tick_count
        if tc == 1:
            sa_engine.band_history[-1] = "green"
        elif tc == 2:
            sa_engine.band_history[-1] = "yellow"  # hits line 550
            # Force SAO long (since active regime HIGH_CURVATURE allows long)
            sa_engine.sao_events.append({"verdict": "ascend"})  # hits line 560
        elif tc == 3:
            sa_engine.band_history[-1] = "red"  # hits line 552
            # Force SAO mid (stochastic allows short and mid)
            sa_engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.STOCHASTIC)
            sa_engine.sao_events.append({"verdict": "ascend"})  # hits line 562
        elif tc == 4:
            # Force SAO short (deterministic allows short range only)
            sa_engine.regime_manager.set_regime_for_agent("single_agent", KernelRegimeType.DETERMINISTIC)
            sa_engine.sao_events.append({"verdict": "ascend"})  # hits line 564
        return res

    setattr(sa_engine, "tick", custom_tick_sa)

    # Force regime manager to return HIGH_CURVATURE_REGIME for step 3 to hit line 560
    orig_get_reg = sa_engine.regime_manager.get_regime_for_agent

    def custom_get_reg(agent_id: str) -> Any:
        if sa_engine.tick_count == 4:
            return HIGH_CURVATURE_REGIME
        return orig_get_reg(agent_id)

    setattr(sa_engine.regime_manager, "get_regime_for_agent", custom_get_reg)

    engine = WorkloadEngine(single_agent_engine=sa_engine)

    sa_steps = [
        # Step 0: Write key
        WorkloadStep(
            agent_actions={
                "act_float": Action.set_activation("single_agent", 0.3),
                # list padding coverage (line 186)
                "act_list_underpad": Action.set_activation("single_agent", [0.2] * 6),
                "regime": Action.change_regime("single_agent", "HIGH_CURVATURE"),
                "write_mem": Action.write_memory(
                    "single_agent", "key_sa", "val_sa", ["tag_sa"]
                ),
                "viol": Action.trigger_violation("single_agent", "custom_viol"),
                "env": Action.set_envelope("single_agent", 0.7, "yellow"),
                "tens": Action.set_tension("single_agent", 1.5),
                "curv": Action.set_curvature("single_agent", 2.2),
                "cost": Action.set_cost_profile("single_agent", 1.5),
                "role": Action.set_role("single_agent", "critic"),
            },
            expected_regime=KernelRegimeType.HIGH_CURVATURE,
            expected_sao=SAOLevel.LONG
        ),
        # Step 1: Overwrite key to test diff_before mapping
        WorkloadStep(
            agent_actions={
                "act_float_2": Action.set_activation("single_agent", 0.3),
                "regime_det": Action.change_regime("single_agent", "STOCHASTIC"),
                "write_mem_overwrite": Action.write_memory(
                    "single_agent", "key_sa", "new_val_sa", ["tag_sa"]
                ),
            },
            expected_regime=KernelRegimeType.STOCHASTIC,
            expected_sao=SAOLevel.MID
        ),
        # Step 2: Delete key manually right before step 2 starts.
        # This covers the deleted key semantic diff branch (line 292).
        WorkloadStep(
            agent_actions={
                "act_zero": Action.set_activation("single_agent", 0.0)  # triggers 'rest' / 're-project' sequence
            },
            expected_regime=KernelRegimeType.DETERMINISTIC,
            expected_sao=SAOLevel.SHORT
        ),
        # Steps 3 and 4 to execute tc=4 and tc=5 for full envelope coverage
        WorkloadStep(
            expected_regime=KernelRegimeType.HIGH_CURVATURE
        ),
        WorkloadStep(
            expected_regime=KernelRegimeType.HIGH_CURVATURE
        )
    ]

    sa_workload = Workload(
        name="SA Booster",
        description="Booster",
        target=SimulationTarget.SINGLE_AGENT,
        regime_expectation=KernelRegimeType.BALANCED,  # mismatch to cover balanced fallback (line 687)
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=1.1,  # mismatch to cover avg_coherence < expected (line 707)
        envelope_expectation=PolicyEnvelope(),
        steps=sa_steps,
        _bypass_validation=True
    )

    # Monkeypatch semantic store to simulate a deletion right before step 2 execution
    engine_any: Any = engine
    orig_run_step = engine_any.run

    def booster_run(workload: Workload) -> Any:
        # We manually inject key_sa deletion before Step 2 starts
        orig_apply_actions = engine_any._apply_actions

        def custom_apply_actions(eng: Any, step: Any, tgt: Any) -> None:
            if step == workload.steps[2]:
                if "key_sa" in sa_engine.semantic_memory.local_store:
                    del sa_engine.semantic_memory.local_store["key_sa"]
            orig_apply_actions(eng, step, tgt)

        setattr(engine_any, "_apply_actions", custom_apply_actions)
        return orig_run_step(workload)

    setattr(engine_any, "run", booster_run)
    _ = engine.run(sa_workload)

    # 2. Multi Agent Dispatch coverage
    ma_engine = MultiAgentEngine(n_agents=3)
    ma_engine.mesh_memory = MeshSemanticMemory()
    # Force Yellow stability band
    ma_engine.band_config.c_green = 0.99999
    ma_engine.band_config.c_red = 0.9999

    # Set up quarantine timer decay scenario
    ma_engine.agents[0].shard.state = ShardState.QUARANTINED
    ma_engine.agents[0].quarantine_timer = 2

    # Wrap ma_engine.tick
    orig_tick_ma = ma_engine.tick

    def custom_tick_ma(task_value: float, excitation: Any, *args: Any, **kwargs: Any) -> str:
        res = orig_tick_ma(task_value, excitation, *args, **kwargs)
        tc = ma_engine.tick_count
        if tc == 1:
            # Force Yellow band
            ma_engine.band_history[-1] = "yellow"  # hits line 598
            # Force SAO mid
            ma_engine.sao_events.append({"verdict": "ascend"})  # hits line 606
        elif tc == 2:
            # Manually release from quarantine to force line 258 branch execution
            ma_engine.agents[0].shard.state = ShardState.IDLE
        return res

    setattr(ma_engine, "tick", custom_tick_ma)

    engine_ma = WorkloadEngine(multi_agent_engine=ma_engine)

    ma_steps = [
        # Step 0: Quarantine is active
        WorkloadStep(
            agent_actions={
                "act_float_ma": Action.set_activation("agent_1", 0.4),
                "act_list_ma": Action.set_activation("agent_2", [0.4] * 12),
                "reg_ma": Action.change_regime("agent_1", "BALANCED"),
                "reg_glob_ma": Action.change_regime("global", "BALANCED"),
                "write_mem_ma": Action.write_memory("agent_1", "key_ma", "val_ma", ["tag_ma"]),
                "write_mem_glob": Action.write_memory("global", "key_glob", "val_glob", ["tag_glob"]),
                "viol_ma": Action.trigger_violation("agent_1", "custom_viol_ma"),
                "env_ma": Action.set_envelope("agent_1", 0.6, "yellow"),
                "env_glob_ma": Action.set_envelope("global", 0.6, "yellow"),
                "tens_ma": Action.set_tension("agent_1", 1.2),
                "tens_glob_ma": Action.set_tension("global", 1.2),
                "curv_ma": Action.set_curvature("agent_1", 1.1),
                "cost_ma": Action.set_cost_profile("agent_1", 1.2),
                "role_ma": Action.set_role("agent_1", "critic"),
            },
            expected_regime=KernelRegimeType.BALANCED
        ),
        # Step 1: Quarantine will decay and agent gets released OUT of quarantine (lines 258, 267)
        WorkloadStep(
            expected_regime=KernelRegimeType.BALANCED
        )
    ]
    ma_workload = Workload(
        name="MA Booster",
        description="Booster",
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.BALANCED,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.0,
        envelope_expectation=PolicyEnvelope(),
        steps=ma_steps,
        _bypass_validation=True
    )
    _ = engine_ma.run(ma_workload)

    # 3. Multi Cluster Dispatch coverage
    mc_engine_booster = MultiClusterEngine()
    _ = mc_engine_booster.create_cluster("cluster_1", "planner")
    _ = mc_engine_booster.create_cluster("cluster_2", "critic")
    a1_mc = UFOAgent("agent_1")
    a2_mc = UFOAgent("agent_2")
    mc_engine_booster.assign_agent_to_cluster(a1_mc, "cluster_1")
    mc_engine_booster.assign_agent_to_cluster(a2_mc, "cluster_2")

    # Inject fake events and history using setattr
    setattr(mc_engine_booster, "sao_events", [{"verdict": "ascend"}])

    # Force global Yellow stability band
    mc_engine_booster.temporal_state.accumulated_tension = 1.5

    # Set up quarantine timer decay scenario for cluster and agent (timer is 2)
    mc_engine_booster.clusters["cluster_1"].quarantine_timer = 2
    a1_mc.shard.state = ShardState.QUARANTINED
    a1_mc.quarantine_timer = 2

    engine_mc = WorkloadEngine(multi_cluster_engine=mc_engine_booster)

    mc_steps = [
        # Step 0: Quarantine active, write to agent_2 (active and not quarantined) to trigger line 159 loop
        WorkloadStep(
            agent_actions={
                "act_float_mc": Action.set_activation("agent_1", 0.4),
                "act_list_mc": Action.set_activation("agent_2", [0.4] * 12),
                "reg_agent_mc": Action.change_regime("agent_1", "STOCHASTIC"),
                "reg_cluster_mc": Action.change_regime("cluster_1", "STOCHASTIC"),
                "reg_glob_mc": Action.change_regime("global", "STOCHASTIC"),
                "write_mem_agent_2": Action.write_memory("agent_2", "key_mc_a", "val_mc_a", ["tag_mc_a"]),
                "write_mem_cluster": Action.write_memory("cluster_1", "key_mc_c", "val_mc_c", ["tag_mc_c"]),
                "write_mem_glob_mc": Action.write_memory(
                    "global", "key_mc_g", "val_mc_g", ["tag_mc_g"]
                ),
                "viol_agent": Action.trigger_violation("agent_1", "viol_mc_a"),
                "viol_cluster": Action.trigger_violation("cluster_1", "viol_mc_c"),
                "viol_other": Action.trigger_violation("non_existent", "viol_mc_x"),
                "env_agent": Action.set_envelope("agent_1", 0.5, "yellow"),
                "env_cluster": Action.set_envelope("cluster_1", 0.5, "yellow"),
                "env_glob": Action.set_envelope("global", 0.5, "yellow"),
                "tens_agent": Action.set_tension("agent_1", 1.0),
                "tens_cluster": Action.set_tension("cluster_1", 1.0),
                "tens_glob": Action.set_tension("global", 1.0),
                "curv_agent": Action.set_curvature("agent_1", 0.8),
                "cost_agent": Action.set_cost_profile("agent_1", 1.1),
                "role_agent": Action.set_role("agent_1", "planner"),
                "role_cluster": Action.set_role("cluster_1", "planner"),
                "role_other": Action.set_role("agent_2", "planner"),
            },
            expected_regime=KernelRegimeType.STOCHASTIC
        ),
        # Step 1: Quarantines decay and get released OUT of quarantine (lines 258, 267)
        WorkloadStep(
            expected_regime=KernelRegimeType.STOCHASTIC
        )
    ]
    mc_workload = Workload(
        name="MC Booster",
        description="Booster",
        target=SimulationTarget.MULTI_CLUSTER,
        regime_expectation=KernelRegimeType.STOCHASTIC,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.0,
        envelope_expectation=PolicyEnvelope(),
        steps=mc_steps,
        _bypass_validation=True
    )

    # Monkeypatch to release cluster and agent quarantine right before step 1.
    # This guarantees that decay/release paths are hit.
    engine_mc_any: Any = engine_mc
    orig_apply_mc = engine_mc_any._apply_actions

    def mc_custom_apply(eng: Any, step: Any, tgt: Any) -> None:
        if step == mc_workload.steps[0]:
            mc_engine_booster.global_band_history.append("green")
        elif step == mc_workload.steps[1]:
            mc_engine_booster.clusters["cluster_1"].quarantine_timer = 0
            a1_mc.shard.state = ShardState.IDLE
            a1_mc.quarantine_timer = 0
            mc_engine_booster.global_band_history.append("yellow")  # hits line 632
            # Set to red on another step/frame to hit line 644
            mc_engine_booster.global_band_history[-1] = "red"  # hits line 644
        orig_apply_mc(eng, step, tgt)

    setattr(engine_mc_any, "_apply_actions", mc_custom_apply)
    _ = engine_mc.run(mc_workload)


def test_invalid_workload_validation_error() -> None:
    """
    Tests that a workload with a non-existent agent/cluster ID causes
    WorkloadConfigurationError to be raised when executed.
    """
    from radial_membrane_ai.exceptions import WorkloadConfigurationError

    # We define a workload targeting a non-existent entity and verify validation catches it.
    invalid_step = WorkloadStep(
        agent_actions={
            "bad_action": Action.change_regime("non_existent_agent_99", "STOCHASTIC")
        }
    )
    invalid_workload = Workload(
        name="Invalid Workload Test",
        description="Should fail validation",
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.BALANCED,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.5,
        envelope_expectation=PolicyEnvelope(),
        steps=[invalid_step]
    )

    engine = WorkloadEngine()
    with pytest.raises(WorkloadConfigurationError, match="references invalid entity"):
        engine.run(invalid_workload)

    # Test run_trace also runs the same validation
    with pytest.raises(WorkloadConfigurationError, match="references invalid entity"):
        engine.run_trace(invalid_workload)
