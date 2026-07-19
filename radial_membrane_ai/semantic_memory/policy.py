"""
Policy definitions and admissibility/policy checking for Policy-Bound Semantic Memory.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.semantic_memory.core import MemoryRecord


@dataclass(frozen=True)
class AdmissibilityContext:
    """
    Context for checking if a memory write is admissible.
    """
    coherence: float
    quarantined: bool
    cost_factor: float
    stability_energy: float


@dataclass(frozen=True)
class PolicyContext:
    """
    Context for checking if a memory read is allowed.
    """
    coherence: float
    quarantined_agents: list[str]


@dataclass(frozen=True)
class MemoryPolicy:
    """
    Policy constraining the reading and writing of a semantic memory record.
    """
    read_scope: Literal["local", "shared", "global"] = "global"
    write_scope: Literal["local", "shared"] = "shared"
    sensitivity: Literal["low", "medium", "high"] = "low"
    admissibility_required: bool = True
    governor_override_allowed: bool = False

    def check_write_admissible(self, record: MemoryRecord, context: AdmissibilityContext) -> bool:
        """
        Validates if a write is admissible given the admissibility requirements and current context.
        """
        # If agent is quarantined, reject write
        if context.quarantined:
            return False

        # If admissibility is required, check coherence and cost limits
        if self.admissibility_required:
            if context.coherence < 0.4:
                return False
            # Sensitivity adjustments
            if self.sensitivity == "high" and context.coherence < 0.7:
                return False
            if self.sensitivity == "medium" and context.coherence < 0.5:
                return False

        # Governor override / stability limit
        if context.stability_energy > 4.5:
            if not self.governor_override_allowed:
                return False

        return True

    def check_read_allowed(self, record: MemoryRecord, agent_id: str, context: PolicyContext) -> bool:
        """
        Checks if an agent is permitted to read a record under the current policy.
        """
        # If reading agent is quarantined, reject
        if agent_id in context.quarantined_agents:
            return False

        # Read scope checks
        if self.read_scope == "local":
            # Only origin agent can read
            return agent_id == record.origin_agent_id

        elif self.read_scope == "shared":
            # Allowed unless global coherence is dangerously low
            if context.coherence < 0.3:
                return False
            # High sensitivity shared records require higher coherence
            if self.sensitivity == "high" and context.coherence < 0.6:
                return False
            return True

        # Global scope
        if self.sensitivity == "high" and context.coherence < 0.4:
            return False

        return True
