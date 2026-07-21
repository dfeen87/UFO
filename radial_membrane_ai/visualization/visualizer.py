"""
MeshVisualizer coordinates snapshot conversion and rendering across all architectural levels.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Tuple

from radial_membrane_ai.visualization.snapshots import (
    VisualizationFrame,
    VisualizationReport,
    VisualizationSummary,
    MembraneGeometrySnapshot,
    VChannelSnapshot
)
from radial_membrane_ai.visualization.dashboard import draw_unified_dashboard, convert_figure_to_image
from radial_membrane_ai.workloads.workload import StabilityBand
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.workloads.engine import WorkloadTrace


def extract_geom_and_vchannel(
    membrane: Any,
    boundary: Any,
    samples: int = 100
) -> Tuple[MembraneGeometrySnapshot, VChannelSnapshot]:
    """Helper to extract geometry and V-channel snapshots from any membrane & boundary."""
    # Sample configured number of angles for boundary coordinates
    boundary_coords = []
    for theta in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        r = boundary.get_radius(theta) if hasattr(boundary, 'get_radius') else 1.0
        boundary_coords.append((r * math.cos(theta), r * math.sin(theta)))

    curvature_map = []
    tension_map = []
    admissibility_zones = []

    from radial_membrane_ai.admissibility import angular_decomposition
    from radial_membrane_ai.projection import closure_ratio

    strings = getattr(membrane, 'strings', [])
    for s in strings:
        curv = boundary.curvature(s.theta) if hasattr(boundary, 'curvature') else 0.0
        curvature_map.append(float(curv))
        tension_map.append(float(getattr(s, 'tension', 0.0)))

        # Sector-level admissibility
        try:
            a_theta, b_theta = angular_decomposition(membrane, s.theta, samples=32)
            c_theta = boundary.get_radius(s.theta) if hasattr(boundary, 'get_radius') else 1.0
            i_ratio = closure_ratio(a_theta, b_theta, c_theta)
            admissibility_zones.append(i_ratio <= 1.0)
        except Exception:
            admissibility_zones.append(True)

    # Fill defaults if strings is empty
    if not strings:
        curvature_map = [0.0] * 12
        tension_map = [0.0] * 12
        admissibility_zones = [True] * 12

    capacity_scalar = getattr(boundary, '_custom_radius_scale', 1.0)

    geom = MembraneGeometrySnapshot(
        boundary_coords=boundary_coords,
        curvature_map=curvature_map,
        tension_map=tension_map,
        admissibility_zones=admissibility_zones,
        capacity_scalar=capacity_scalar
    )

    pressures = [float(getattr(s, 'radius', 0.0)) for s in strings] if strings else [0.0] * 12
    routing_dirs = [float(getattr(s, 'theta', 0.0)) for s in strings] if strings else [0.0] * 12
    admissible = [float(getattr(s, 'activation', 0.0)) < 0.95 for s in strings] if strings else [True] * 12
    sao_eligible = [float(getattr(s, 'activation', 0.0)) > 0.5 for s in strings] if strings else [False] * 12

    vch = VChannelSnapshot(
        pressures=pressures,
        routing_dirs=routing_dirs,
        admissible=admissible,
        sao_eligible=sao_eligible
    )
    return geom, vch


class MeshVisualizer:
    """
    Renders diagnostic dashboard frames and timelines across agents, clusters,
    and the global mesh.
    """
    def __init__(self, samples_resolution: int = 100) -> None:
        self.samples_resolution = samples_resolution

    def render_agent_state(self, agent_state: Any) -> VisualizationFrame:
        """Renders single agent state snapshot at a single timestep."""
        if isinstance(agent_state, VisualizationFrame):
            return agent_state

        # Can be SingleAgentEngine, UFOAgent, or similar
        membrane = getattr(agent_state, 'membrane', None)
        boundary = getattr(agent_state, 'boundary', None)

        # Avoid logical branching issues in Python 3.12 coverage
        if not (membrane and boundary):
            raise ValueError("State does not have valid membrane or boundary attributes.")

        geom, vch = extract_geom_and_vchannel(membrane, boundary, samples=self.samples_resolution)

        # Base metrics
        curvature = float(max(geom.curvature_map)) if geom.curvature_map else 0.0
        tension = float(max(geom.tension_map)) if geom.tension_map else 0.0

        # Check active regime
        reg_mgr = getattr(agent_state, 'regime_manager', None)
        if reg_mgr:
            regime = reg_mgr.get_regime_for_agent("single_agent").regime_type
        else:
            regime_str = getattr(agent_state, 'kernel_regime', "BALANCED")
            regime_str = regime_str.upper() if regime_str else "BALANCED"
            regime = (
                KernelRegimeType[regime_str]
                if regime_str in KernelRegimeType.__members__
                else KernelRegimeType.BALANCED
            )

        # Coherence
        if hasattr(agent_state, 'compute_local_coherence'):
            coherence = agent_state.compute_local_coherence()
        elif hasattr(agent_state, 'shard') and hasattr(agent_state.shard, 'quality_score'):
            coherence = agent_state.shard.quality_score
        else:
            coherence = 1.0

        # Stability Band
        band_history = getattr(agent_state, 'band_history', [])
        if band_history:
            band_str = band_history[-1]
            if "yellow" in band_str.lower():
                band = StabilityBand.YELLOW
            elif "red" in band_str.lower():
                band = StabilityBand.RED
            else:
                band = StabilityBand.GREEN
        else:
            band = StabilityBand.GREEN

        # SAO Events
        sao_events = []
        raw_events = getattr(agent_state, 'sao_events', [])
        for i, ev_item in enumerate(raw_events):
            from radial_membrane_ai.workloads.workload import SAOLevel
            from radial_membrane_ai.workloads.engine import SAOEvent
            sao_events.append(SAOEvent(step_index=i, level=SAOLevel.MID))

        frame = VisualizationFrame(
            membrane_geometry=geom,
            vchannels=vch,
            curvature=curvature,
            tension=tension,
            sao_events=sao_events,
            stability_band=band,
            coherence=coherence,
            regime=regime
        )

        # Render the unified dashboard for this single frame
        fig = draw_unified_dashboard([frame])
        frame.rendered = fig
        return frame

    def render_cluster_state(self, cluster_state: Any) -> VisualizationFrame:
        """Renders single cluster state snapshot at a single timestep."""
        return self.render_agent_state(cluster_state)

    def render_global_state(self, global_state: Any) -> VisualizationFrame:
        """Renders global mesh state snapshot at a single timestep."""
        if hasattr(global_state, 'temporal_state'):
            from radial_membrane_ai.multi_agent.cluster import UFOCluster
            temp_c = UFOCluster(cluster_id="global_mesh")

            # Populate with aggregated agents
            agents = getattr(global_state, 'agents', []) or []
            agents = list(agents)

            if not agents:
                clusters = getattr(global_state, 'clusters', {})
                for c in clusters.values():
                    agents.extend(c.agents)

            temp_c.agents = agents
            temp_c.update_cluster_membrane()

            # Map history and global regime
            reg_mgr = getattr(global_state, 'regime_manager', None)
            regime = reg_mgr.get_global_regime().regime_type if reg_mgr else KernelRegimeType.BALANCED

            frame = self.render_agent_state(temp_c)
            frame.regime = regime

            # Synchronize history
            band_hist = getattr(global_state, 'global_band_history', getattr(global_state, 'band_history', []))
            if band_hist:
                band_str = band_hist[-1]
                if "yellow" in band_str.lower():
                    frame.stability_band = StabilityBand.YELLOW
                elif "red" in band_str.lower():
                    frame.stability_band = StabilityBand.RED
                else:
                    frame.stability_band = StabilityBand.GREEN

            return frame
        else:
            return self.render_agent_state(global_state)

    def render_workload_trace(self, trace: WorkloadTrace) -> VisualizationReport:
        """Converts a WorkloadTrace into a full VisualizationReport and renders its timeline."""
        viz_frames = []
        for i, wf in enumerate(trace.frames):
            metrics = wf.metrics

            geom_snapshot = getattr(wf, 'membrane_geometry', None)
            vch_snapshot = getattr(wf, 'vchannels', None)

            if geom_snapshot is None:
                geom_snapshot = MembraneGeometrySnapshot(
                    boundary_coords=[(math.cos(theta), math.sin(theta)) for theta in np.linspace(0, 2*math.pi, 100)],
                    curvature_map=[metrics.curvature] * 12,
                    tension_map=[metrics.tension] * 12,
                    admissibility_zones=[True] * 12,
                    capacity_scalar=1.0
                )
            if vch_snapshot is None:
                vch_snapshot = VChannelSnapshot(
                    pressures=[metrics.tension] * 12,
                    routing_dirs=[(2.0 * math.pi * idx) / 12.0 for idx in range(12)],
                    admissible=[True] * 12,
                    sao_eligible=[metrics.sao_level.value != 'none'] * 12
                )

            # Parse StabilityBand
            b_str = (
                metrics.stability_band.value
                if hasattr(metrics.stability_band, 'value')
                else str(metrics.stability_band)
            )
            if "yellow" in b_str.lower():
                band = StabilityBand.YELLOW
            elif "red" in b_str.lower():
                band = StabilityBand.RED
            else:
                band = StabilityBand.GREEN

            # Parse KernelRegimeType
            reg_str = metrics.regime.upper()
            try:
                regime = KernelRegimeType[reg_str]
            except KeyError:
                regime = KernelRegimeType.BALANCED

            lde_state = getattr(wf, 'lde_state', None)
            frame = VisualizationFrame(
                membrane_geometry=geom_snapshot,
                vchannels=vch_snapshot,
                curvature=metrics.curvature,
                tension=metrics.tension,
                sao_events=wf.sao_events,
                stability_band=band,
                coherence=metrics.coherence,
                regime=regime,
                lde_boundary=lde_state.boundary if lde_state else None,
                lde_channels=lde_state.channels if lde_state else None,
                lde_strings=lde_state.strings if lde_state else None,
                lde_state=lde_state
            )
            viz_frames.append(frame)

        # Compute summary
        curvatures = [f.curvature for f in viz_frames]
        tensions = [f.tension for f in viz_frames]
        coherences = [f.coherence for f in viz_frames]

        max_curvature = float(max(curvatures)) if curvatures else 0.0
        max_tension = float(max(tensions)) if tensions else 0.0

        # Calculate event counts
        sao_event_count = 0
        rollback_count = 0
        quarantine_count = 0
        regime_transition_count = 0

        result = getattr(trace, 'result', None)
        if result:
            sao_event_count = len(getattr(result, 'sao_events', []))
            rollback_count = len(getattr(result, 'rollback_events', []))
            quarantine_count = len(getattr(result, 'quarantine_events', []))
            regime_transition_count = len(getattr(result, 'regime_transitions', []))

        coherence_stability = float(np.mean(coherences)) if coherences else 1.0

        summary = VisualizationSummary(
            max_curvature=max_curvature,
            max_tension=max_tension,
            sao_event_count=sao_event_count,
            rollback_count=rollback_count,
            quarantine_count=quarantine_count,
            regime_transition_count=regime_transition_count,
            coherence_stability=coherence_stability
        )

        report = VisualizationReport(
            frames=viz_frames,
            summary=summary
        )

        # Render unified timeline dashboard
        if viz_frames and viz_frames[0].lde_state is not None:
            from radial_membrane_ai.lde.visualizer import LDEVisualizer
            lde_vis = LDEVisualizer()
            fig = lde_vis.render_lde_dashboard(viz_frames[0].lde_state)
            report.rendered = convert_figure_to_image(fig)
        else:
            fig = draw_unified_dashboard(viz_frames, report)
            report.rendered = convert_figure_to_image(fig)
        return report
