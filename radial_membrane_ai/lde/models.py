"""
Data models and configuration for the Letter‑Depth Encoding (L.D.E.) subsystem.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any


@dataclass
class LDEConfig:
    """Configuration parameters for the L.D.E. pipeline."""
    sigma: List[str] = field(default_factory=lambda: [chr(c) for c in range(ord('a'), ord('z') + 1)])
    w_f: float = 0.20
    w_s: float = 0.20
    w_b: float = 0.20
    w_r: float = 0.20
    w_c: float = 0.20
    rho: str = "full-reconstruction"  # 'full-reconstruction' or 'compressed'
    r_0: float = 1.0
    epsilon: float = 1e-6


@dataclass
class LDEString:
    """Letter-level governed string representing a single character's state."""
    activation: float
    phase: float
    positions: List[int]
    spread: float
    depth: float
    tension: float
    stiffness: float
    cost: float
    links: Dict[str, float] = field(default_factory=dict)


@dataclass
class LDEVChannel:
    """Letter-pair coherence corridor representing routing pathways between characters."""
    source: str
    target: str
    adjacency: float
    coherence: float
    pressure: float


@dataclass
class LDEBoundaryGeometry:
    """Polar boundary geometry representing paragraph/sentence identity."""
    radius_map: List[float]
    curvature_map: List[float]
    tangent_map: List[float]
    asymmetry: float


@dataclass
class LDEState:
    """Full encoding output of the L.D.E. pipeline."""
    strings: Dict[str, LDEString]
    coherence_matrix: Dict[Tuple[str, str], float]
    channels: List[LDEVChannel]
    boundary: LDEBoundaryGeometry
    reconstruction_map: Dict[str, Any]
