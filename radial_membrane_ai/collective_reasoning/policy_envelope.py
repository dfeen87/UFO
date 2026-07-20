"""
Shared Policy Envelopes for U.F.O. collective reasoning.
Defines global, cluster, and local policy envelopes, intersection rules,
and violation detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Optional, Any

STABILITY_ORDER = {"red": 0, "yellow": 1, "green": 2}


@dataclass(frozen=True)
class PolicyEnvelope:
    """
    A unified standalone policy envelope governing collective operations,
    reasoning steps, and semantic memory sharing.
    """
    allowed_agents: Set[str] = field(default_factory=set)
    allowed_clusters: Set[str] = field(default_factory=set)
    min_trust: float = 0.0
    allowed_roles: Set[str] = field(default_factory=set)
    promotable_memory_fields: Set[str] = field(default_factory=set)
    allowed_tags: Set[str] = field(default_factory=set)
    max_cost_band: int = 2  # 0=green, 1=yellow, 2=red (higher means more cost allowed/permissive)
    stability_required_band: str = "red"  # red, yellow, green (green is most strict)
    temporal_window_min: int = 0
    temporal_window_max: Optional[int] = None

    def check_write_admissible(self, record: Any, context: Any) -> bool:
        """
        Base/fallback admissibility check. Specializations (like MemoryPolicy) override this.
        """
        return True

    def check_read_allowed(self, record: Any, agent_id: str, context: Any) -> bool:
        """
        Base/fallback read check. Specializations (like MemoryPolicy) override this.
        """
        return True

    def intersect(self, other: PolicyEnvelope) -> PolicyEnvelope:
        """
        Computes the intersection of this envelope with another, producing
        a more restrictive combined PolicyEnvelope.
        """
        # Intersection of sets
        intersected_agents = self.allowed_agents.intersection(other.allowed_agents) if (
            self.allowed_agents and other.allowed_agents
        ) else (self.allowed_agents or other.allowed_agents)

        intersected_clusters = self.allowed_clusters.intersection(other.allowed_clusters) if (
            self.allowed_clusters and other.allowed_clusters
        ) else (self.allowed_clusters or other.allowed_clusters)

        intersected_roles = self.allowed_roles.intersection(other.allowed_roles) if (
            self.allowed_roles and other.allowed_roles
        ) else (self.allowed_roles or other.allowed_roles)

        intersected_fields = self.promotable_memory_fields.intersection(other.promotable_memory_fields) if (
            self.promotable_memory_fields and other.promotable_memory_fields
        ) else (self.promotable_memory_fields or other.promotable_memory_fields)

        intersected_tags = self.allowed_tags.intersection(other.allowed_tags) if (
            self.allowed_tags and other.allowed_tags
        ) else (self.allowed_tags or other.allowed_tags)

        # Trust = max of the two values (more strict)
        new_min_trust = max(self.min_trust, other.min_trust)

        # Cost band = min (stricter limit, e.g. min(yellow=1, red=2) = yellow=1)
        new_max_cost_band = min(self.max_cost_band, other.max_cost_band)

        # Stability band = max of the requirements (stricter requirement: max of green=2 and yellow=1 is green=2)
        v1 = STABILITY_ORDER.get(self.stability_required_band.lower(), 0)
        v2 = STABILITY_ORDER.get(other.stability_required_band.lower(), 0)
        new_stability_band = self.stability_required_band if v1 >= v2 else other.stability_required_band

        # Temporal window min = max (more strict minimum)
        new_temp_min = max(self.temporal_window_min, other.temporal_window_min)

        # Temporal window max = min (more strict maximum limit)
        if self.temporal_window_max is not None and other.temporal_window_max is not None:
            new_temp_max: Optional[int] = min(self.temporal_window_max, other.temporal_window_max)
        else:
            new_temp_max = (
                self.temporal_window_max if self.temporal_window_max is not None else other.temporal_window_max
            )

        return PolicyEnvelope(
            allowed_agents=intersected_agents,
            allowed_clusters=intersected_clusters,
            min_trust=new_min_trust,
            allowed_roles=intersected_roles,
            promotable_memory_fields=intersected_fields,
            allowed_tags=intersected_tags,
            max_cost_band=new_max_cost_band,
            stability_required_band=new_stability_band,
            temporal_window_min=new_temp_min,
            temporal_window_max=new_temp_max
        )

    def validate_compliance(
        self,
        current_trust: float,
        current_cost_band: int,
        current_stability_band: str,
        current_ticks: int,
        agent_ids: Optional[Set[str]] = None,
        agent_roles: Optional[Set[str]] = None,
        cluster_ids: Optional[Set[str]] = None,
        cluster_roles: Optional[Set[str]] = None,
        used_tags: Optional[Set[str]] = None,
        used_fields: Optional[Set[str]] = None
    ) -> bool:
        """
        Validates compliance of active state parameters against this PolicyEnvelope.
        Returns True if fully compliant, False if any constraint is violated.
        """
        # 1. Trust compliance
        if current_trust < self.min_trust:
            return False

        # 2. Cost compliance
        if current_cost_band > self.max_cost_band:
            return False

        # 3. Stability compliance
        curr_stab_val = STABILITY_ORDER.get(current_stability_band.lower(), 0)
        req_stab_val = STABILITY_ORDER.get(self.stability_required_band.lower(), 0)
        if curr_stab_val < req_stab_val:
            return False

        # 4. Temporal compliance
        if current_ticks < self.temporal_window_min:
            return False
        if self.temporal_window_max is not None and current_ticks > self.temporal_window_max:
            return False

        # 5. Set-based validations
        if agent_ids and self.allowed_agents:
            if not agent_ids.issubset(self.allowed_agents):
                return False

        if agent_roles and self.allowed_roles:
            if not agent_roles.issubset(self.allowed_roles):
                return False

        if cluster_ids and self.allowed_clusters:
            if not cluster_ids.issubset(self.allowed_clusters):
                return False

        if used_tags and self.allowed_tags:
            if not used_tags.issubset(self.allowed_tags):
                return False

        if used_fields and self.promotable_memory_fields:
            if not used_fields.issubset(self.promotable_memory_fields):
                return False

        return True
