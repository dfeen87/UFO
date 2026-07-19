"""
Federated Routing & V-Channels module.

Implements shard routing using projection into admissible shard geometry,
and data-plane V-Channels: Type-W, Type-R, Type-T, Type-A, Type-E.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from radial_membrane_ai.shard import FederatedShard, ShardState


class FederatedRoutingChannel:
    """
    V-Channel routing mechanics mapping workloads onto federated shard nodes.
    Supported data-plane V-Channels:
    - Type-W (workload)
    - Type-R (results)
    - Type-T (telemetry)
    - Type-A (attestation)
    - Type-E (exceptions/residuals)
    """

    def __init__(self, channel_id: str, channel_type: str = "Type-W") -> None:
        self.channel_id = channel_id
        valid_types = {"Type-W", "Type-R", "Type-T", "Type-A", "Type-E"}
        if channel_type not in valid_types:
            raise ValueError(f"Invalid V-Channel type: {channel_type}")
        self.channel_type = channel_type

    def project_into_shard_geometry(self, workload: Dict[str, Any], shard: FederatedShard) -> Dict[str, Any]:
        """
        Calculates II_{G_shard}(W(t)) - projecting workload weights onto the shard limits.
        """
        # A simple mathematical projection of workload parameters into shard capacity bounds
        workload_cap = workload.get("capacity", 0.5)
        projected_cap = min(workload_cap, shard.capacity)

        projected_workload = workload.copy()
        projected_workload["capacity"] = projected_cap
        projected_workload["routed"] = True
        return projected_workload

    def route_workload(self, workload: Dict[str, Any], shards: List[FederatedShard]) -> Optional[FederatedShard]:
        """
        Routes the workload to the most suitable admissible federated shard.
        """
        best_shard: Optional[FederatedShard] = None
        best_score = -1.0

        for shard in shards:
            if shard.evaluate_admissibility(workload):
                # Calculate routing compatibility score: higher trust and capacity, lower cost & latency
                score = (shard.trust_score * shard.capacity) / (shard.cost_factor * shard.latency + 1e-6)
                if score > best_score:
                    best_score = score
                    best_shard = shard

        if best_shard:
            best_shard.state = ShardState.ACTIVE
        return best_shard
