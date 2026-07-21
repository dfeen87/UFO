"""
Integration and Unit Tests for the Pre-Launch Sweep Release Readiness.
Ensures CI/CD automation of deterministic seeding, model invariants,
reproducible visualization rendering, L.D.E. stress-testing, and post-tick assertions.
"""

from __future__ import annotations
import random
import pytest
import numpy as np
from PIL import Image

# Core UFO frameworks
from radial_membrane_ai.utils import set_deterministic_env
from radial_membrane_ai.exceptions import (
    GovernanceError,
    GeometryValidationError,
    ValidationError,
    WorkloadValidationError,
    WorkloadConfigurationError,
    ReconstructionError
)
from radial_membrane_ai.lde.models import LDEConfig, LDEBoundaryGeometry, LDEState
from radial_membrane_ai.lde.pipeline import lde_encode
from radial_membrane_ai.governor import GovernorConfig
from radial_membrane_ai.workloads.workload import (
    Action,
    StabilityBand,
    SimulationTarget,
    WorkloadStep,
    Workload,
    create_high_curvature_workload
)
from radial_membrane_ai.workloads.engine import WorkloadEngine
from radial_membrane_ai.visualization.visualizer import MeshVisualizer
from radial_membrane_ai.ufo_engine import SingleAgentEngine, MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.multi_agent.agent import UFOAgent


def test_deterministic_seeding_and_reproducibility() -> None:
    """Confirms that the centralized seeding behaves deterministically."""
    set_deterministic_env(seed=0)
    v_rand_1 = random.random()
    v_np_1 = np.random.rand()

    set_deterministic_env(seed=0)
    v_rand_2 = random.random()
    v_np_2 = np.random.rand()

    assert v_rand_1 == v_rand_2
    assert v_np_1 == v_np_2


def test_model_invariants() -> None:
    """Verifies that improper state model inputs raise structured errors."""
    with pytest.raises(GeometryValidationError):
        LDEBoundaryGeometry(radius_map=[], curvature_map=[], tangent_map=[], asymmetry=0.0)

    with pytest.raises(ValidationError):
        LDEConfig(sigma=[])

    with pytest.raises(ValidationError):
        GovernorConfig(w_d=-0.5)

    with pytest.raises(WorkloadValidationError):
        WorkloadStep(expected_coherence_range=(1.2, 0.4))


def test_visualization_reproducible_rendering(tmp_path) -> None:
    """Verifies diagnostic rendering results in consistent, valid visual output."""
    set_deterministic_env(seed=0)
    workload_engine = WorkloadEngine()
    curv_workload = create_high_curvature_workload()
    trace = workload_engine.run_trace(curv_workload)

    visualizer = MeshVisualizer(samples_resolution=100)
    report = visualizer.render_workload_trace(trace)

    assert report.rendered is not None
    out_img_path = tmp_path / "test_timeline.png"
    report.rendered.save(out_img_path, dpi=(120, 120))

    img = Image.open(out_img_path)
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_lde_roundtrip_stress_test() -> None:
    """Stress tests L.D.E. with non-ASCII, casing, extreme spacing, and ReconstructionError."""
    stress_text = "✨ UFO [2026]! ä, ö, ü, ß.   Extreme spacing   and punc...???"
    state = lde_encode(stress_text, LDEConfig(rho="full-reconstruction"))
    assert isinstance(state, LDEState)

    kelvin_symbol = "\u212a"
    with pytest.raises(ReconstructionError):
        lde_encode(kelvin_symbol, LDEConfig(rho="full-reconstruction"))


def test_workload_structural_validation() -> None:
    """Confirms invalid steps/targets are rejected before run."""
    invalid_step = WorkloadStep(
        agent_actions={
            "invalid": Action.change_regime("non_existent_99", "STOCHASTIC")
        }
    )
    workload = Workload(
        name="Invalid Name Test",
        description="Testing workload structural validation.",
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=create_high_curvature_workload().regime_expectation,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.8,
        envelope_expectation=create_high_curvature_workload().envelope_expectation,
        steps=[invalid_step]
    )

    workload_engine = WorkloadEngine()
    with pytest.raises(WorkloadConfigurationError):
        workload_engine.run(workload)


def test_stability_metrics_post_tick_assertions() -> None:
    """Forces each hard stability limit to fail and verifies the deterministic halt raise."""
    # Lyapunov Energy limit
    sa_e = SingleAgentEngine()
    sa_e.membrane.strings[0].tension = 25.0
    with pytest.raises(GovernanceError, match="Lyapunov energy"):
        sa_e.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # Tension limit
    sa_t = SingleAgentEngine()
    sa_t.membrane.temporal_state.accumulated_tension = 100.0
    with pytest.raises(GovernanceError, match="Temporal tension"):
        sa_t.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # Stiffness limit
    sa_s = SingleAgentEngine()
    sa_s.membrane.strings[0].stiffness = 5.0
    with pytest.raises(GovernanceError, match="Stiffness"):
        sa_s.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # Boundary Curvature limit
    sa_c = SingleAgentEngine()
    setattr(sa_c.boundary, "curvature", lambda theta, **kwargs: 60.0)
    with pytest.raises(GovernanceError, match="Boundary curvature"):
        sa_c.tick(task_value=0.5, excitation=np.ones(12) * 0.5)

    # Radius Deviation limit
    sa_r = SingleAgentEngine()
    setattr(sa_r.boundary, "update_boundary", lambda *args, **kwargs: None)
    sa_r.boundary.radius_deviation[1] = 7.0
    with pytest.raises(GovernanceError, match="Radius deviation"):
        sa_r.tick(task_value=0.5, excitation=np.ones(12) * 0.5)


def test_dry_run_scenarios() -> None:
    """Executes small deterministic nominal runs across single, multi-agent, and multi-cluster."""
    set_deterministic_env(seed=0)

    # Single agent
    sa = SingleAgentEngine()
    b_sa = sa.tick(task_value=0.8, excitation=np.ones(12) * 0.05)
    assert b_sa in ("green", "yellow", "red")

    # Multi-agent
    ma = MultiAgentEngine(n_agents=2)
    b_ma = ma.tick(task_value=0.8, excitation=np.ones(12) * 0.05)
    assert b_ma in ("green", "yellow", "red")

    # Multi-cluster
    mc = MultiClusterEngine()
    mc.create_cluster("cluster_alpha", "planner")
    mc.assign_agent_to_cluster(UFOAgent("agent_1"), "cluster_alpha")
    b_mc = mc.tick(task_value=0.8, default_excitation=np.ones(12) * 0.05)
    assert b_mc in ("green", "yellow", "red")
