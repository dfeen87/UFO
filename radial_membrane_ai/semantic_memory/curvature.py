"""
Curvature and tension modeling for Policy-Bound Semantic Memory.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MemoryCurvatureState:
    """
    Represents the geometric curvature and tension state of a semantic memory segment.
    - Curvature represents the access and data volume load on the memory space.
    - Tension represents conflicts, policy violations, or contradictory tags/writes in the memory space.
    """
    curvature: float = 0.0
    tension: float = 0.0
    access_count: int = 0
    conflict_count: int = 0

    def update_on_write(self, size_factor: float = 0.1) -> None:
        """
        Updates state metrics when a memory record is written.
        """
        self.access_count += 1
        # Increase curvature based on size factor/write load, clamped to [0.0, 5.0]
        self.curvature = min(5.0, self.curvature + size_factor)

    def update_on_read(self, read_intensity: float = 0.02) -> None:
        """
        Updates state metrics when a memory record is read.
        """
        self.access_count += 1
        self.curvature = min(5.0, self.curvature + read_intensity)

    def update_on_conflict(self, conflict_intensity: float = 0.25) -> None:
        """
        Updates state metrics when a write or policy check triggers a conflict.
        """
        self.conflict_count += 1
        self.tension = min(5.0, self.tension + conflict_intensity)

    def decay(self, rate: float = 0.05) -> None:
        """
        Allows curvature and tension to smoothly decay over time (e.g., during simulation steps).
        """
        self.curvature = max(0.0, self.curvature - rate)
        self.tension = max(0.0, self.tension - rate * 0.5)
