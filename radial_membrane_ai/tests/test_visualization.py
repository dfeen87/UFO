"""
Comprehensive unit and integration tests for Mesh Visualization Layer (Governed Diagnostic Dashboard).
"""

from __future__ import annotations
import math
import numpy as np
import pytest
import matplotlib.pyplot as plt
from PIL import Image

from radial_membrane_ai.visualization.snapshots import (
    MembraneGeometrySnapshot,
    VChannelSnapshot,
    VisualizationSummary,
    VisualizationFrame,
    VisualizationReport
)
from radial_membrane_ai.visualization.dashboard import (
    render_membrane_geometry_panel,
    render_vchannel_panel,
    render_curvature_tension_timeline,
    render_sao_timeline,
    render_stability_band_panel,
    render_coherence_panel,
    render_regime_timeline,
    draw_unified_dashboard
)
from radial_membrane_ai.visualization.visualizer import MeshVisualizer, extract_geom_and_vchannel
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.workloads import (
    create_high_curvature_workload,
    create_cooperative_workload,
    create_adversarial_workload,
    WorkloadEngine,
    StabilityBand,
    SAOLevel
)
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine


def test_snapshots_dataclass_initialization() -> None:
    """Verifies that all visualization snapshots can be constructed with expected fields."""
    geom = MembraneGeometrySnapshot(
        boundary_coords=[(1.0, 0.0), (0.0, 1.0)],
        curvature_map=[0.5] * 12,
        tension_map=[0.2] * 12,
        admissibility_zones=[True] * 12,
        capacity_scalar=1.0
    )
    vch = VChannelSnapshot(
        pressures=[0.3] * 12,
        routing_dirs=[0.0] * 12,
        admissible=[True] * 12,
        sao_eligible=[False] * 12
    )
    summary = VisualizationSummary(
        max_curvature=1.5,
        max_tension=2.0,
        sao_event_count=3,
        rollback_count=1,
        quarantine_count=0,
        regime_transition_count=2,
        coherence_stability=0.85
    )
    frame = VisualizationFrame(
        membrane_geometry=geom,
        vchannels=vch,
        curvature=0.6,
        tension=0.4,
        sao_events=[],
        stability_band=StabilityBand.GREEN,
        coherence=0.9,
        regime=KernelRegimeType.BALANCED
    )
    report = VisualizationReport(
        frames=[frame],
        summary=summary,
        rendered=None
    )
    assert report.summary.sao_event_count == 3
    assert report.frames[0].stability_band == StabilityBand.GREEN


def test_individual_panel_rendering() -> None:
    """Tests rendering of each individual panel in isolation."""
    geom = MembraneGeometrySnapshot(
        boundary_coords=[(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 100)],
        curvature_map=[0.2] * 12,
        tension_map=[0.1] * 12,
        admissibility_zones=[True] * 12,
        capacity_scalar=1.05
    )
    vch = VChannelSnapshot(
        pressures=[0.4] * 12,
        routing_dirs=[(2 * math.pi * i) / 12 for i in range(12)],
        admissible=[True] * 12,
        sao_eligible=[True] * 12
    )
    frame = VisualizationFrame(
        membrane_geometry=geom,
        vchannels=vch,
        curvature=0.5,
        tension=0.3,
        sao_events=[],
        stability_band=StabilityBand.GREEN,
        coherence=0.8,
        regime=KernelRegimeType.DETERMINISTIC
    )

    # Panel A: Geometry
    fig, ax = plt.subplots()
    render_membrane_geometry_panel(geom, ax)
    plt.close(fig)

    # Panel B: V-Channel
    fig, ax = plt.subplots()
    render_vchannel_panel(vch, ax)
    plt.close(fig)

    # Panel C: Curvature Tension Timeline
    fig, ax = plt.subplots()
    render_curvature_tension_timeline([frame], ax)
    plt.close(fig)

    # Panel D: SAO Timeline
    from radial_membrane_ai.workloads.engine import SAOEvent
    frame_sao = VisualizationFrame(
        membrane_geometry=geom,
        vchannels=vch,
        curvature=0.5,
        tension=0.3,
        sao_events=[
            SAOEvent(step_index=0, level=SAOLevel.SHORT),
            SAOEvent(step_index=0, level=SAOLevel.MID),
            SAOEvent(step_index=0, level=SAOLevel.LONG)
        ],
        stability_band=StabilityBand.YELLOW,
        coherence=0.8,
        regime=KernelRegimeType.STOCHASTIC
    )
    fig, ax = plt.subplots()
    render_sao_timeline([frame_sao], ax)
    plt.close(fig)

    # Panel E: Stability Band Timeline
    fig, ax = plt.subplots()
    render_stability_band_panel([frame, frame_sao], None, ax)
    plt.close(fig)

    # Panel F: Coherence
    fig, ax = plt.subplots()
    render_coherence_panel([frame], ax)
    plt.close(fig)

    # Panel G: Regime Transition
    fig, ax = plt.subplots()
    render_regime_timeline([frame, frame_sao], ax)
    plt.close(fig)


def test_empty_and_edge_case_timelines() -> None:
    """Verifies that rendering timelines with no frames or incomplete data does not crash."""
    fig, ax = plt.subplots()
    render_curvature_tension_timeline([], ax)
    render_sao_timeline([], ax)
    render_stability_band_panel([], None, ax)
    render_coherence_panel([], ax)
    render_regime_timeline([], ax)
    plt.close(fig)


def test_visualizer_single_agent_integration() -> None:
    """Verifies MeshVisualizer integration with SingleAgentEngine state."""
    sa_engine = SingleAgentEngine()
    visualizer = MeshVisualizer()

    frame = visualizer.render_agent_state(sa_engine)
    assert frame is not None
    assert isinstance(frame, VisualizationFrame)
    assert frame.rendered is not None
    plt.close(frame.rendered)

    # Trigger custom regime/band settings to test mapping branches
    sa_engine.band_history.append("yellow")
    sa_engine.sao_events.append({"verdict": "ascend"})
    frame_yellow = visualizer.render_agent_state(sa_engine)
    assert frame_yellow.stability_band == StabilityBand.YELLOW
    plt.close(frame_yellow.rendered)

    sa_engine.band_history.append("red")
    frame_red = visualizer.render_agent_state(sa_engine)
    assert frame_red.stability_band == StabilityBand.RED
    plt.close(frame_red.rendered)


def test_visualizer_invalid_state_exception() -> None:
    """Ensures MeshVisualizer raises ValueError if passed state lacks membrane or boundary attributes."""
    visualizer = MeshVisualizer()
    with pytest.raises(ValueError, match="State does not have valid membrane or boundary attributes"):
        visualizer.render_agent_state(object())


def test_visualizer_cluster_state_rendering() -> None:
    """Verifies MeshVisualizer cluster state rendering."""
    from radial_membrane_ai.multi_agent.cluster import UFOCluster
    cluster = UFOCluster(cluster_id="cluster_alpha")
    visualizer = MeshVisualizer()

    frame = visualizer.render_cluster_state(cluster)
    assert frame is not None
    assert frame.rendered is not None
    plt.close(frame.rendered)


def test_visualizer_global_state_rendering() -> None:
    """Verifies MeshVisualizer global state rendering for MultiAgentEngine & MultiClusterEngine."""
    visualizer = MeshVisualizer()

    # 1. MultiAgentEngine global state
    ma_engine = MultiAgentEngine(n_agents=2)
    frame_ma = visualizer.render_global_state(ma_engine)
    assert frame_ma is not None
    assert frame_ma.rendered is not None
    plt.close(frame_ma.rendered)

    # 2. MultiClusterEngine global state
    mc_engine = MultiClusterEngine()
    mc_engine.create_cluster("c_1")
    frame_mc = visualizer.render_global_state(mc_engine)
    assert frame_mc is not None
    assert frame_mc.rendered is not None
    plt.close(frame_mc.rendered)


def test_workload_trace_execution_and_rendering() -> None:
    """Executes a workload, produces a WorkloadTrace, and renders a complete VisualizationReport."""
    workload_engine = WorkloadEngine()
    visualizer = MeshVisualizer()

    # 1. High-Curvature Single Agent workload
    curv_workload = create_high_curvature_workload()
    trace_curv = workload_engine.run_trace(curv_workload)

    assert len(trace_curv.frames) == 2
    report_curv = visualizer.render_workload_trace(trace_curv)
    assert report_curv is not None
    assert isinstance(report_curv, VisualizationReport)
    assert report_curv.rendered is not None
    assert isinstance(report_curv.rendered, Image.Image)

    # 2. Cooperative Multi-Agent workload
    coop_workload = create_cooperative_workload()
    trace_coop = workload_engine.run_trace(coop_workload)
    report_coop = visualizer.render_workload_trace(trace_coop)
    assert report_coop is not None

    # 3. Adversarial Multi-Agent workload
    adv_workload = create_adversarial_workload()
    trace_adv = workload_engine.run_trace(adv_workload)
    report_adv = visualizer.render_workload_trace(trace_adv)
    assert report_adv is not None


def test_deterministic_and_reproducible_rendering() -> None:
    """Ensures that two consecutive renderings of the same workload trace yield deterministic summaries."""
    workload_engine = WorkloadEngine()
    visualizer = MeshVisualizer()

    workload = create_high_curvature_workload()
    trace = workload_engine.run_trace(workload)

    # Render twice
    report_1 = visualizer.render_workload_trace(trace)
    report_2 = visualizer.render_workload_trace(trace)

    # Compare summaries
    assert report_1.summary.max_curvature == report_2.summary.max_curvature
    assert report_1.summary.max_tension == report_2.summary.max_tension
    assert report_1.summary.sao_event_count == report_2.summary.sao_event_count
    assert report_1.summary.rollback_count == report_2.summary.rollback_count
    assert report_1.summary.quarantine_count == report_2.summary.quarantine_count
    assert report_1.summary.regime_transition_count == report_2.summary.regime_transition_count
    assert report_1.summary.coherence_stability == report_2.summary.coherence_stability


def test_coverage_gaps_booster() -> None:
    """Covers rare branches to ensure 100% line coverage on the visualization package."""
    # 1. Unknown regime name rendering fallback
    geom = MembraneGeometrySnapshot(
        boundary_coords=[],
        curvature_map=[0.0] * 12,
        tension_map=[0.0] * 12,
        admissibility_zones=[True] * 12,
        capacity_scalar=1.0
    )
    vch = VChannelSnapshot(
        pressures=[0.0] * 12,
        routing_dirs=[0.0] * 12,
        admissible=[True] * 12,
        sao_eligible=[False] * 12
    )
    frame = VisualizationFrame(
        membrane_geometry=geom,
        vchannels=vch,
        curvature=0.0,
        tension=0.0,
        sao_events=[],
        stability_band=StabilityBand.GREEN,
        coherence=1.0,
        regime=None  # type: ignore
    )

    # Render with None regime
    fig, ax = plt.subplots()
    render_curvature_tension_timeline([frame], ax)
    plt.close(fig)

    # 2. Extract geometry/vchannels with missing elements to trigger fallbacks (lines 50-51)
    class BrokenString:
        def __init__(self) -> None:
            self.theta = 0.0
            self.tension = 0.0

    class BrokenMembrane:
        def __init__(self) -> None:
            self.strings = [BrokenString()]

        def field_value(self, angle: float, basis_type: str = "gaussian") -> float:
            raise RuntimeError("Field value error")

    geom_fb, vch_fb = extract_geom_and_vchannel(BrokenMembrane(), None)
    assert len(geom_fb.curvature_map) == 1
    assert geom_fb.admissibility_zones[0] is True  # fallback hit

    # 3. Extract with completely empty membrane (lines 55-57 of visualizer.py)
    class EmptyMembrane:
        def __init__(self) -> None:
            pass

    geom_empty, vch_empty = extract_geom_and_vchannel(EmptyMembrane(), None)
    assert len(geom_empty.curvature_map) == 12
    assert len(vch_empty.pressures) == 12

    # 4. MeshVisualizer.render_agent_state with string-based kernel regime
    class DummyState:
        def __init__(self) -> None:
            self.membrane = SingleAgentEngine().membrane
            self.boundary = SingleAgentEngine().boundary
            self.kernel_regime = "invalid_regime_name"
            self.band_history = ["green"]
            self.sao_events = [{"verdict": "block"}]

    dummy = DummyState()
    visualizer = MeshVisualizer()
    frame_dummy = visualizer.render_agent_state(dummy)
    assert frame_dummy.regime == KernelRegimeType.BALANCED
    plt.close(frame_dummy.rendered)

    # 5. render_global_state with missing agents and clusters fallback
    # (lines 196-198) and missing reg_mgr (lines 212-215)
    class DummyGlobalState:
        def __init__(self) -> None:
            self.temporal_state = object()
            self.agents = None
            from radial_membrane_ai.multi_agent.cluster import UFOCluster
            self.clusters = {"c_alpha": UFOCluster(cluster_id="c_alpha")}
            self.global_band_history = ["yellow"]

    dummy_global = DummyGlobalState()
    frame_global = visualizer.render_global_state(dummy_global)
    assert frame_global.regime == KernelRegimeType.BALANCED
    assert frame_global.stability_band == StabilityBand.YELLOW
    plt.close(frame_global.rendered)

    # 6. render_global_state without temporal_state (line 230)
    class DummySimpleState:
        def __init__(self) -> None:
            self.membrane = SingleAgentEngine().membrane
            self.boundary = SingleAgentEngine().boundary

    dummy_simple = DummySimpleState()
    frame_simple = visualizer.render_global_state(dummy_simple)
    assert frame_simple is not None
    plt.close(frame_simple.rendered)

    # 7. render_workload_trace with missing snapshots on frame (to trigger fallbacks on lines 253-264)
    from radial_membrane_ai.workloads.engine import WorkloadFrameMetrics, WorkloadFrame, WorkloadTrace
    metrics_dummy = WorkloadFrameMetrics(
        step_index=0,
        curvature=0.4,
        tension=0.2,
        coherence=0.9,
        regime="DETERMINISTIC",
        stability_band=StabilityBand.GREEN,
        sao_level=SAOLevel.NONE,
        envelope_state=None  # type: ignore
    )
    # We must explicitly set membrane_geometry and vchannels to None here!
    wf_dummy = WorkloadFrame(metrics=metrics_dummy)
    wf_dummy.membrane_geometry = None
    wf_dummy.vchannels = None
    trace_dummy = WorkloadTrace(frames=[wf_dummy], result=None)
    report_dummy = visualizer.render_workload_trace(trace_dummy)
    assert report_dummy is not None
    assert report_dummy.frames[0].membrane_geometry.capacity_scalar == 1.0

    # 8. render_stability_band_panel with rollbacks and draw_unified_dashboard keywords
    summary = VisualizationSummary(
        max_curvature=1.0,
        max_tension=1.0,
        sao_event_count=0,
        rollback_count=1,
        quarantine_count=0,
        regime_transition_count=0,
        coherence_stability=1.0
    )
    report = VisualizationReport(frames=[frame], summary=summary)
    fig, ax = plt.subplots()
    render_stability_band_panel([frame], report, ax)
    plt.close(fig)

    fig_dash = draw_unified_dashboard([frame], report, dpi=120, figsize=(12, 8))
    plt.close(fig_dash)

    # 9. Trigger non-short-circuit branch of line 92: membrane None but boundary is not None, and vice-versa
    class BoundaryOnlyState:
        def __init__(self) -> None:
            self.membrane = None
            self.boundary = SingleAgentEngine().boundary

    class MembraneOnlyState:
        def __init__(self) -> None:
            self.membrane = SingleAgentEngine().membrane
            self.boundary = None

    with pytest.raises(ValueError, match="State does not have valid membrane or boundary attributes"):
        visualizer.render_agent_state(BoundaryOnlyState())
    with pytest.raises(ValueError, match="State does not have valid membrane or boundary attributes"):
        visualizer.render_agent_state(MembraneOnlyState())

    # 10. Default fallback to BALANCED when kernel_regime is missing entirely (line 122)
    class MissingRegimeState:
        def __init__(self) -> None:
            self.membrane = SingleAgentEngine().membrane
            self.boundary = SingleAgentEngine().boundary

    frame_mr = visualizer.render_agent_state(MissingRegimeState())
    assert frame_mr.regime == KernelRegimeType.BALANCED
    plt.close(frame_mr.rendered)
