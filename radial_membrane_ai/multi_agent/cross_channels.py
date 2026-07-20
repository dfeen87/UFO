"""
Cross-Cluster V-Channels and Policy Envelopes for multi-cluster UFO communication.
"""

from __future__ import annotations
from typing import List, Set, Optional
from dataclasses import dataclass, field

from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope


@dataclass(frozen=True)
class ClusterPolicyEnvelope(PolicyEnvelope):
    """
    Enforces policies governing cross-cluster signal routing and memory sharing.
    ClusterPolicyEnvelope specializes PolicyEnvelope with cluster-specific defaults.
    """
    min_trust: float = 0.5
    min_compliance: float = 0.5
    allowed_roles: Set[str] = field(default_factory=set)
    blocked_tags: Set[str] = field(default_factory=set)
    blocked_prefixes: List[str] = field(default_factory=list)

    def check_admissibility(
        self,
        source: UFOCluster,
        target: UFOCluster,
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """
        Evaluates whether a cross-cluster transfer is admissible under policy constraints.
        """
        # 1. Compute aggregate metrics for source cluster
        src_active = [a for a in source.agents if a.shard.state != ShardState.QUARANTINED]
        if src_active:
            src_trust = sum(a.shard.trust_score for a in src_active) / len(src_active)
            src_compliance = sum(a.shard.policy_compliance for a in src_active) / len(src_active)
        else:
            src_trust = 0.0
            src_compliance = 0.0

        # 2. Compute aggregate metrics for target cluster
        tgt_active = [a for a in target.agents if a.shard.state != ShardState.QUARANTINED]
        if tgt_active:
            tgt_trust = sum(a.shard.trust_score for a in tgt_active) / len(tgt_active)
            tgt_compliance = sum(a.shard.policy_compliance for a in tgt_active) / len(tgt_active)
        else:
            tgt_trust = 0.0
            tgt_compliance = 0.0

        # 3. Check trust and compliance limits
        if src_trust < self.min_trust or tgt_trust < self.min_trust:
            return False
        if src_compliance < self.min_compliance or tgt_compliance < self.min_compliance:
            return False

        # 4. Check role constraints (if target has roles not matching allowed roles, and allowed_roles is specified)
        if self.allowed_roles and target.role not in self.allowed_roles:
            return False

        # 5. Check semantic constraints (blocked prefixes / tags)
        if key is not None:
            for prefix in self.blocked_prefixes:
                if key.startswith(prefix):
                    return False

        if tags is not None and self.blocked_tags:
            if not tags.isdisjoint(self.blocked_tags):
                return False

        return True


class CrossClusterVChannel:
    """
    Represents a governed connection between two clusters carrying signals
    between cluster-level membranes and active child agents.
    """

    def __init__(
        self,
        source_cluster: UFOCluster,
        target_cluster: UFOCluster,
        channel_type: str = "Type-W",
        coupling_strength: float = 0.1,
        dampening: float = 0.9,
        policy_envelope: Optional[ClusterPolicyEnvelope] = None
    ) -> None:
        """
        Initializes the CrossClusterVChannel.

        Args:
            source_cluster: Source cluster of the signal.
            target_cluster: Target cluster of the signal.
            channel_type: "Type-W" (workload), "Type-T" (tension), or "Type-R" (residuals).
            coupling_strength: Weight of coupling.
            dampening: Loss factor representing signal transmission decay across boundaries.
            policy_envelope: Enclosing policy envelope governing boundary traversal.
        """
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster
        self.channel_type = channel_type
        self.coupling_strength = coupling_strength
        self.dampening = dampening
        self.policy_envelope = policy_envelope if policy_envelope is not None else ClusterPolicyEnvelope()

        self.rejection_count: int = 0

    def propagate(self) -> bool:
        """
        Propagates signals from the source cluster membrane to the target cluster
        and downwards to target agents, subject to policy checks.

        Returns:
            True if propagation was successful, False if rejected by policy constraints.
        """
        # Validate admissibility via policy envelope
        if not self.policy_envelope.check_admissibility(self.source_cluster, self.target_cluster):
            self.rejection_count += 1
            # Increase conflict metric on target cluster stability curvature
            self.target_cluster.stability_curvature.update_on_conflict(conflict_intensity=0.2)
            return False

        # Target active agents list to receive downward signals
        tgt_active = [a for a in self.target_cluster.agents if a.shard.state != ShardState.QUARANTINED]

        for i in range(12):
            src_str = self.source_cluster.membrane.strings[i]
            tgt_str = self.target_cluster.membrane.strings[i]

            if self.channel_type == "Type-W":
                # Propagate workload activation
                delta_act = self.coupling_strength * self.dampening * src_str.activation
                tgt_str.activation = max(0.0, min(1.0, tgt_str.activation + delta_act))

                # Downward propagation to active agents
                if tgt_active:
                    agent_delta = delta_act / len(tgt_active)
                    for agent in tgt_active:
                        agent.membrane.strings[i].activation = max(
                            0.0,
                            min(1.0, agent.membrane.strings[i].activation + agent_delta)
                        )

            elif self.channel_type == "Type-T":
                # Propagate tension
                delta_tension = self.coupling_strength * self.dampening * src_str.tension
                tgt_str.tension = max(0.0, min(1.0, tgt_str.tension + delta_tension))

                # Downward propagation to active agents
                if tgt_active:
                    agent_delta = delta_tension / len(tgt_active)
                    for agent in tgt_active:
                        agent.membrane.strings[i].tension = max(
                            0.0,
                            min(1.0, agent.membrane.strings[i].tension + agent_delta)
                        )

            elif self.channel_type == "Type-R":
                # Propagate residuals / cost factors
                delta_cost = self.coupling_strength * self.dampening * src_str.cost
                tgt_str.cost += delta_cost

                # Downward propagation to active agents
                if tgt_active:
                    agent_delta = delta_cost / len(tgt_active)
                    for agent in tgt_active:
                        agent.membrane.strings[i].cost += agent_delta

        return True
