"""
Mesh-level correctness guarantees and state rollback logic for U.F.O. collective reasoning.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Sequence, List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.semantic_memory.core import MemoryRecord, MeshSemanticMemory
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.collective_reasoning.collective_admissibility import CollectiveStepContext, get_agent_temporal_closure_ratio
from radial_membrane_ai.shard import ShardState


@dataclass
class StateBackup:
    """
    A snapshot of state to allow exact rollback on correctness failure.
    """
    # Agent-level snapshot
    agent_activations: Dict[str, List[float]] = field(default_factory=dict)
    agent_radii: Dict[str, List[float]] = field(default_factory=dict)
    agent_boundary_deviations: Dict[str, Dict[int, float]] = field(default_factory=dict)
    agent_memory_local_store: Dict[str, Dict[str, MemoryRecord]] = field(default_factory=dict)
    agent_residuals: Dict[str, List[float]] = field(default_factory=dict)
    agent_shard_states: Dict[str, ShardState] = field(default_factory=dict)

    # Cluster-level snapshot
    cluster_activations: Dict[str, List[float]] = field(default_factory=dict)
    cluster_radii: Dict[str, List[float]] = field(default_factory=dict)
    cluster_boundary_deviations: Dict[str, Dict[int, float]] = field(default_factory=dict)
    cluster_memory_store: Dict[str, Dict[str, MemoryRecord]] = field(default_factory=dict)
    cluster_tension: Dict[str, float] = field(default_factory=dict)
    cluster_stability_band: Dict[str, str] = field(default_factory=dict)

    # Global/Mesh snapshot
    global_memory_store: Dict[str, MemoryRecord] = field(default_factory=dict)


class MeshCorrectness:
    """
    Coordinates multi-tier correctness audits, state snapshots, and rollbacks.
    """

    def __init__(self, ledger: Optional[Any] = None) -> None:
        self.ledger = ledger

    def backup_state(
        self,
        agents: Sequence[UFOAgent],
        clusters: Sequence[UFOCluster],
        global_memory: Optional[MeshSemanticMemory] = None
    ) -> StateBackup:
        """
        Takes a full deep copy of current agent and cluster states to enable flawless rollbacks.
        """
        backup = StateBackup()

        # 1. Backup Agents
        for agent in agents:
            a_id = agent.agent_id
            backup.agent_activations[a_id] = [s.activation for s in agent.membrane.strings]
            backup.agent_radii[a_id] = [s.radius for s in agent.membrane.strings]
            backup.agent_boundary_deviations[a_id] = dict(agent.boundary.radius_deviation)
            backup.agent_residuals[a_id] = list(agent.residual_history)
            backup.agent_shard_states[a_id] = agent.shard.state

            if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                backup.agent_memory_local_store[a_id] = dict(agent.semantic_memory.local_store)

        # 2. Backup Clusters
        for cluster in clusters:
            c_id = cluster.cluster_id
            backup.cluster_activations[c_id] = [s.activation for s in cluster.membrane.strings]
            backup.cluster_radii[c_id] = [s.radius for s in cluster.membrane.strings]
            backup.cluster_boundary_deviations[c_id] = dict(cluster.boundary.radius_deviation)
            backup.cluster_tension[c_id] = cluster.tension_metric
            backup.cluster_stability_band[c_id] = cluster.stability_band

            if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                backup.cluster_memory_store[c_id] = dict(cluster.semantic_memory.global_store)

        # 3. Backup Global Memory
        if global_memory is not None:
            backup.global_memory_store = dict(global_memory.global_store)

        return backup

    def audit_correctness(
        self,
        agents: Sequence[UFOAgent],
        clusters: Sequence[UFOCluster],
        step: CollectiveStepContext,
        global_envelope: PolicyEnvelope
    ) -> Tuple[bool, List[str]]:
        """
        Performs multi-level correctness checks across agents, clusters, and global levels.
        Returns (is_correct, list_of_violation_messages).
        """
        violations: List[str] = []

        participating_agents = [a for a in agents if a.agent_id in step.participating_agent_ids]
        involved_clusters = [c for c in clusters if c.cluster_id in step.involved_cluster_ids]

        # ==================== AGENT LEVEL CHECKS ====================
        for agent in participating_agents:
            a_id = agent.agent_id

            # 1. No destructive overwrite (check if latest memory action was an overwrite without residuals)
            if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                mem = agent.semantic_memory
                for key, record in mem.local_store.items():
                    # If it was an overwrite but residual state is missing, it is a destructive overwrite violation
                    if record.updated_at > record.created_at and record.residual_state is None:
                        violations.append(f"Agent {a_id}: Destructive overwrite violation on memory key '{key}'")

            # 2. Closure ratio and capacity constraints
            for s in agent.membrane.strings:
                cr = get_agent_temporal_closure_ratio(agent, s.theta)
                if cr > 1.2:  # Tolerant threshold for correctness limit
                    violations.append(f"Agent {a_id}: Extreme closure ratio violation {cr:.2f} at theta {s.theta:.2f}")

            # 3. Temporal admissibility check
            t_state = getattr(agent.membrane, "temporal_state", None)
            if t_state is not None:
                avg_t = float(np.mean(t_state.tension_history)) if t_state.tension_history else 0.0
                if avg_t > 1.5:  # Critical tension threshold
                    violations.append(f"Agent {a_id}: Critical average temporal tension {avg_t:.2f} exceeded")

            # 4. Cost compliance
            # Extract current cost factor or tax
            cost_factor = agent.shard.cost_factor
            if cost_factor > 3.0:  # Excessive cost
                violations.append(f"Agent {a_id}: Cost factor {cost_factor:.2f} exceeds permissible envelope limit")

        # ==================== CLUSTER LEVEL CHECKS ====================
        for cluster in involved_clusters:
            c_id = cluster.cluster_id

            # 1. Tension and Lyapunov energy checks
            if cluster.tension_metric > 1.5:
                violations.append(f"Cluster {c_id}: Critical cluster tension {cluster.tension_metric:.2f} exceeded")

            if cluster.stability_band == "red":
                violations.append(f"Cluster {c_id}: Stability band is 'red' (critical state)")

            # 2. Policy consistency
            # Check cluster semantic memory for any records violating global allowed tags
            if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                for key, record in cluster.semantic_memory.global_store.items():
                    if global_envelope.allowed_tags:
                        if not record.tags.issubset(global_envelope.allowed_tags):
                            violations.append(f"Cluster {c_id}: Semantic tag leak violation on key '{key}'")

            # 3. Conflicting promotions detection (e.g. key promoted with different values in same tick)
            if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                keys_seen = set()
                for key, record in cluster.semantic_memory.global_store.items():
                    # Simply flag if key origin mismatch and updated recently (indicates a rapid clash)
                    if record.updated_at > time.time() - 0.1:
                        if record.origin_agent_id and record.origin_agent_id != c_id:
                            keys_seen.add(key)
                            violations.append(f"Cluster {c_id}: Conflicting promotion detected for key '{key}'")

        # ==================== GLOBAL LEVEL CHECKS ====================
        # 1. Global stability band check
        if step.stability_band == "red" and global_envelope.stability_required_band == "green":
            violations.append("Global: Global stability band is red while green is required")

        # 2. Global policy envelope check
        if step.cost_band > global_envelope.max_cost_band:
            violations.append(f"Global: Step cost band {step.cost_band} exceeds global limit {global_envelope.max_cost_band}")

        return len(violations) == 0, violations

    def rollback(
        self,
        backup: StateBackup,
        agents: Sequence[UFOAgent],
        clusters: Sequence[UFOCluster],
        global_memory: Optional[MeshSemanticMemory] = None,
        quarantine_ticks: int = 5,
        failing_ids: Optional[Set[str]] = None
    ) -> None:
        """
        Rolls back the state of all specified agents, clusters, and global memory to the backup snapshot.
        Optionally places failing entities under temporary quarantine.
        """
        # 1. Restore Agents
        for agent in agents:
            a_id = agent.agent_id
            if a_id in backup.agent_activations:
                # Restore activations
                for i, s in enumerate(agent.membrane.strings):
                    s.activation = backup.agent_activations[a_id][i]
                    s.radius = backup.agent_radii[a_id][i]

                # Restore boundary deviations
                agent.boundary.radius_deviation = dict(backup.agent_boundary_deviations[a_id])

                # Restore residuals and state
                agent.residual_history = list(backup.agent_residuals[a_id])
                agent.shard.state = backup.agent_shard_states[a_id]

                # Restore memories
                if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                    agent.semantic_memory.local_store = dict(backup.agent_memory_local_store[a_id])

            # Quarantine if flagged as failing
            if failing_ids and a_id in failing_ids:
                agent.shard.state = ShardState.QUARANTINED
                # Store quarantine duration in a custom attribute if needed
                setattr(agent, "quarantine_timer", quarantine_ticks)

                if self.ledger is not None:
                    self.ledger.log_failure(
                        record_id=f"correctness_quarantine_{a_id}_{int(time.time() * 1000)}",
                        error_type="correctness_violation_quarantine",
                        shard_id=a_id,
                        severity="critical",
                        details={"quarantine_ticks": quarantine_ticks}
                    )

        # 2. Restore Clusters
        for cluster in clusters:
            c_id = cluster.cluster_id
            if c_id in backup.cluster_activations:
                # Restore activations
                for i, s in enumerate(cluster.membrane.strings):
                    s.activation = backup.cluster_activations[c_id][i]
                    s.radius = backup.cluster_radii[c_id][i]

                # Restore boundary deviations
                cluster.boundary.radius_deviation = dict(backup.cluster_boundary_deviations[c_id])

                # Restore tension & stability
                cluster.tension_metric = backup.cluster_tension[c_id]
                cluster.stability_band = backup.cluster_stability_band[c_id]

                # Restore memories
                if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                    cluster.semantic_memory.global_store = dict(backup.cluster_memory_store[c_id])

            # Quarantine / isolate cluster if flagged
            if failing_ids and c_id in failing_ids:
                cluster.stability_band = "red"
                setattr(cluster, "quarantine_timer", quarantine_ticks)

        # 3. Restore Global Memory
        if global_memory is not None and backup.global_memory_store:
            global_memory.global_store = dict(backup.global_memory_store)
