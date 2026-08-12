# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Cross-agent collective admissibility constraints for U.F.O. collective reasoning.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence, Set, Any, Optional, List
from dataclasses import dataclass, field

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.semantic_memory.core import MemoryRecord
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.admissibility import angular_decomposition, dynamic_capacity_boundary_temporal
from radial_membrane_ai.projection import closure_ratio


@dataclass
class CollectiveStepContext:
    """
    Encapsulates context for a proposed collective reasoning step/transition.
    """
    proposed_activation: Optional[np.ndarray] = None
    involved_records: List[MemoryRecord] = field(default_factory=list)
    participating_agent_ids: Set[str] = field(default_factory=set)
    involved_cluster_ids: Set[str] = field(default_factory=set)
    cost_band: int = 0  # 0=green, 1=yellow, 2=red
    stability_band: str = "green"
    trust_score: float = 1.0
    ticks: int = 1
    used_tags: Set[str] = field(default_factory=set)
    used_fields: Set[str] = field(default_factory=set)


def get_agent_temporal_closure_ratio(agent: UFOAgent, theta: float) -> float:
    """
    Computes the instantaneous temporal capacity closure ratio at theta for an agent.
    """
    a, b = angular_decomposition(agent.membrane, theta, samples=64)
    c = dynamic_capacity_boundary_temporal(agent.boundary, theta, agent.membrane)
    return closure_ratio(a, b, c)


def collective_admissibility(
    agents: Sequence[UFOAgent],
    clusters: Sequence[UFOCluster],
    step: CollectiveStepContext,
    global_envelope: PolicyEnvelope,
    N_temporal_min: int = 5,
    max_tension_threshold: float = 0.7
) -> bool:
    """
    Evaluates whether a multi-agent collective reasoning step is admissible.

    A step is admissible if and only if:
    1. Closure ratio constraints hold for all participating agents (instantaneous <= 1.0, mean <= 1.1).
    2. Temporal admissibility holds (consecutive ticks >= N_temporal_min, avg tension <= 0.7).
    3. Cluster-level admissibility holds (mean cluster tension <= 0.7, stability != "red").
    4. Global and local intersected envelope constraints are fully satisfied.
    """
    participating_agents = [
        a for a in agents
        if a.agent_id in step.participating_agent_ids
    ]
    involved_clusters = [
        c for c in clusters
        if c.cluster_id in step.involved_cluster_ids
    ]

    # If no agents or clusters specified, use those whose IDs match
    if not participating_agents and step.participating_agent_ids:
        return False
    if not involved_clusters and step.involved_cluster_ids:
        return False

    # 1. Check participating agents' admissibility constraints
    for agent in participating_agents:
        # Instantaneous temporal closure ratio check across all 12 strings
        max_instant = 0.0
        for s in agent.membrane.strings:
            cr = get_agent_temporal_closure_ratio(agent, s.theta)
            if cr > max_instant:
                max_instant = cr
        if max_instant > 1.0 + 1e-9:
            return False

        # Time-weighted closure ratio check
        t_state = getattr(agent.membrane, "temporal_state", None)
        if t_state is not None:
            avg_i = float(np.mean(t_state.admissibility_history)) if t_state.admissibility_history else 0.0
            if avg_i > 1.1 + 1e-9:
                return False

            # 2. Temporal admissibility check
            if t_state.consecutive_admissible_ticks < N_temporal_min:
                return False

            avg_t = float(np.mean(t_state.tension_history)) if t_state.tension_history else 0.0
            if avg_t > max_tension_threshold:
                return False

    # 3. Cluster-level admissibility check
    for cluster in involved_clusters:
        if cluster.tension_metric > max_tension_threshold:
            return False
        if cluster.stability_band == "red":
            return False

    # 4. Global policy envelope constraints check
    # Start with global envelope
    intersected = global_envelope

    # Helper to get PolicyEnvelope
    def get_policy_envelope(obj: Any) -> PolicyEnvelope:
        return getattr(obj, "policy_envelope", None) or PolicyEnvelope()

    # Intersect with all participating agents' envelopes
    for agent in participating_agents:
        intersected = intersected.intersect(get_policy_envelope(agent))

    # Intersect with all involved clusters' envelopes
    for cluster in involved_clusters:
        intersected = intersected.intersect(get_policy_envelope(cluster))

    # Map roles and properties
    agent_roles: Set[str] = set()
    for agent in participating_agents:
        # Deduce role from kernel_regime or similar if present
        role = getattr(agent, "role", agent.kernel_regime)
        if role:
            agent_roles.add(role)

    cluster_roles: Set[str] = set()
    for cluster in involved_clusters:
        if cluster.role:
            cluster_roles.add(cluster.role)

    # Validate compliance of the step against the intersected policy envelope
    compliant = intersected.validate_compliance(
        current_trust=step.trust_score,
        current_cost_band=step.cost_band,
        current_stability_band=step.stability_band,
        current_ticks=step.ticks,
        agent_ids=step.participating_agent_ids,
        agent_roles=agent_roles,
        cluster_ids=step.involved_cluster_ids,
        cluster_roles=cluster_roles,
        used_tags=step.used_tags,
        used_fields=step.used_fields
    )

    return compliant
