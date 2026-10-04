# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Federated Shard Architecture module.

Implements Shards with states: Idle, Resonant, Active, Saturated, Inhibited,
Quarantined, Revoked, Expired.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any
from radial_membrane_ai.numeric import finite_real


ADMISSIBILITY_EVIDENCE_FIELDS = frozenset({"capacity", "privacy", "latency", "cost"})
ROUTING_METADATA_FIELDS = frozenset({"routed"})


class ShardState(Enum):
    IDLE = auto()
    RESONANT = auto()
    ACTIVE = auto()
    SATURATED = auto()
    INHIBITED = auto()
    QUARANTINED = auto()
    REVOKED = auto()
    EXPIRED = auto()


@dataclass
class FederatedShard:
    """
    Represents a single shard acting as a governed compute facet in a federated mesh.
    """
    shard_id: str
    state: ShardState = ShardState.IDLE
    capacity: float = 1.0
    trust_score: float = 1.0
    consent_granted: bool = True
    privacy_level: float = 1.0
    policy_compliance: float = 1.0
    cost_factor: float = 1.0
    latency: float = 10.0  # ms
    quality_score: float = 1.0

    def evaluate_admissibility(self, workload: Dict[str, Any]) -> bool:
        """
        Evaluates dynamic workload admissibility against capacity, trust, consent,
        privacy, policy compliance, cost, latency, and quality.
        """
        if self.state in (ShardState.QUARANTINED, ShardState.REVOKED, ShardState.EXPIRED):
            return False

        # This method is a trust boundary. Unknown keys, booleans, non-numeric
        # values, and non-finite evidence are rejected rather than relying on
        # comparisons whose NaN semantics can silently fail open.
        allowed_keys = ADMISSIBILITY_EVIDENCE_FIELDS | ROUTING_METADATA_FIELDS
        if not isinstance(workload, dict) or not set(workload).issubset(allowed_keys):
            return False
        # ``routed`` is an informational projection marker, not attestation.  It
        # is deliberately excluded from every calculation below.  Restricting
        # its public shape lets routing output compose with admission without
        # allowing arbitrary metadata through this trust boundary.
        if "routed" in workload and workload["routed"] is not True:
            return False
        if not isinstance(self.state, ShardState) or not isinstance(self.consent_granted, bool):
            return False

        shard_evidence = (
            self.capacity, self.privacy_level, self.latency, self.cost_factor,
            self.trust_score, self.policy_compliance, self.quality_score,
        )
        if any(finite_real(value) is None for value in shard_evidence):
            return False
        if any(value < 0.0 for value in shard_evidence):
            return False
        normalized_evidence = (
            self.trust_score, self.policy_compliance, self.privacy_level, self.quality_score,
        )
        if any(value > 1.0 for value in normalized_evidence):
            return False

        required_capacity = workload.get("capacity", 0.1)
        required_privacy = workload.get("privacy", 0.5)
        max_latency = workload.get("latency", 100.0)
        max_cost = workload.get("cost", 5.0)

        requirements = (required_capacity, required_privacy, max_latency, max_cost)
        if any(finite_real(value) is None or value < 0.0 for value in requirements):
            return False

        if not self.consent_granted:
            return False
        if self.capacity < required_capacity:
            return False
        if self.privacy_level < required_privacy:
            return False
        if self.latency > max_latency:
            return False
        if self.cost_factor > max_cost:
            return False
        if self.trust_score < 0.5 or self.policy_compliance < 0.5:
            return False

        return True
