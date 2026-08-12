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

        required_capacity = workload.get("capacity", 0.1)
        required_privacy = workload.get("privacy", 0.5)
        max_latency = workload.get("latency", 100.0)
        max_cost = workload.get("cost", 5.0)

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
