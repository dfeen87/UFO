# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Custom Exceptions for the UFO Governed Deformable Radial Membrane framework.
"""

from __future__ import annotations


class GovernanceError(ValueError):
    """Base exception for all governance and verification errors in the UFO system."""
    pass


class GeometryValidationError(GovernanceError):
    """Raised when radial membrane boundary or geometry structures violate invariants."""
    pass


class ValidationError(GovernanceError):
    """Raised when generic dataclass validations fail."""
    pass


class WorkloadValidationError(ValidationError):
    """Raised when workload step definitions or parameters violate schema or validation invariants."""
    pass


class WorkloadConfigurationError(GovernanceError):
    """Raised when workload engine setup or pre-run configuration matches/validation fails."""
    pass


class InvalidSimulationTargetError(GovernanceError):
    """Raised when the simulation engine target does not match the workload requirements."""
    pass


class ReconstructionError(GovernanceError):
    """Raised when the L.D.E. round-trip text reconstruction fails or mismatches the original text."""
    pass
