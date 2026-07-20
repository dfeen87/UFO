"""
Data models and snapshots for the Mesh Visualization Layer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from PIL.Image import Image

from radial_membrane_ai.workloads.workload import StabilityBand
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.workloads.engine import SAOEvent


@dataclass
class MembraneGeometrySnapshot:
    boundary_coords: List[Tuple[float, float]]        # radial boundary polygon
    curvature_map: List[float]                       # curvature per sector (12 sectors)
    tension_map: List[float]                         # tension per sector (12 sectors)
    admissibility_zones: List[bool]                  # sector-level admissibility flags
    capacity_scalar: float                           # global shrink/expand scalar


@dataclass
class VChannelSnapshot:
    pressures: List[float]                           # pressure per channel
    routing_dirs: List[float]                        # angle or direction per channel
    admissible: List[bool]                           # channel-level admissibility
    sao_eligible: List[bool]                         # SAO promotion eligibility flags


@dataclass
class VisualizationSummary:
    max_curvature: float
    max_tension: float
    sao_event_count: int
    rollback_count: int
    quarantine_count: int
    regime_transition_count: int
    coherence_stability: float


@dataclass
class VisualizationFrame:
    membrane_geometry: MembraneGeometrySnapshot
    vchannels: VChannelSnapshot
    curvature: float
    tension: float
    sao_events: List[SAOEvent]
    stability_band: StabilityBand
    coherence: float
    regime: KernelRegimeType
    rendered: Optional[Figure] = None


@dataclass
class VisualizationReport:
    frames: List[VisualizationFrame]
    summary: VisualizationSummary
    rendered: Optional[Image] = None
