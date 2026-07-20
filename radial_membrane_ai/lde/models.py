"""
Data models and configuration for the Letter‑Depth Encoding (L.D.E.) subsystem.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

from radial_membrane_ai.exceptions import GeometryValidationError, ValidationError


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
    geometry_samples: int = 100

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces strict invariants on L.D.E. Configuration."""
        if not self.sigma:
            raise ValidationError("LDEConfig.sigma cannot be empty.")
        if len(self.sigma) != len(set(self.sigma)):
            raise ValidationError("LDEConfig.sigma elements must be unique.")
        weights = [
            ("w_f", self.w_f), ("w_s", self.w_s), ("w_b", self.w_b),
            ("w_r", self.w_r), ("w_c", self.w_c)
        ]
        for w_name, w_val in weights:
            if w_val < 0.0:
                raise ValidationError(f"Weight {w_name} must be non-negative, got {w_val}.")
        if self.r_0 <= 0.0:
            raise ValidationError(f"r_0 must be strictly positive, got {self.r_0}.")
        if self.epsilon <= 0.0:
            raise ValidationError(f"epsilon must be strictly positive, got {self.epsilon}.")
        if self.rho not in {"full-reconstruction", "compressed"}:
            raise ValidationError(f"rho must be 'full-reconstruction' or 'compressed', got {self.rho}.")
        if self.geometry_samples <= 0:
            raise ValidationError(f"geometry_samples must be strictly positive, got {self.geometry_samples}.")


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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces strict invariants on L.D.E. Boundary Geometry."""
        if not self.radius_map:
            raise GeometryValidationError("radius_map cannot be empty.")
        if not self.curvature_map:
            raise GeometryValidationError("curvature_map cannot be empty.")
        if not self.tangent_map:
            raise GeometryValidationError("tangent_map cannot be empty.")

        n_rad = len(self.radius_map)
        if len(self.curvature_map) != n_rad or len(self.tangent_map) != n_rad:
            raise GeometryValidationError(
                f"Geometry lists must be of equal length. "
                f"Got radius={n_rad}, curvature={len(self.curvature_map)}, tangent={len(self.tangent_map)}."
            )

        for idx, r in enumerate(self.radius_map):
            if r < 0.0:
                raise GeometryValidationError(f"Radius map capacity cannot be negative. Got {r} at index {idx}.")

        if self.asymmetry < 0.0:
            raise GeometryValidationError(f"Asymmetry must be non-negative, got {self.asymmetry}.")


@dataclass
class LDEState:
    """Full encoding output of the L.D.E. pipeline."""
    strings: Dict[str, LDEString]
    coherence_matrix: Dict[Tuple[str, str], float]
    channels: List[LDEVChannel]
    boundary: LDEBoundaryGeometry
    reconstruction_map: Dict[str, Any]
