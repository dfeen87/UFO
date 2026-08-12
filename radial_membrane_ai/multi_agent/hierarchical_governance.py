# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Hierarchical Governance and Hierarchical Symmetric Ascension Operator (SAO) promotions.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List, Optional

from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.semantic_memory.core import (
    MeshSemanticMemory,
    MemoryRecord,
    PromotionResult,
    promote_memory
)
from radial_membrane_ai.semantic_memory.policy import AdmissibilityContext
from radial_membrane_ai.residuals import ResidualLedger, ResidualRecord
from radial_membrane_ai.envelope import BrimEnvelope
from radial_membrane_ai.shard import ShardState


class HierarchicalHolisticGovernor:
    """
    A hierarchical supervisor integrating cluster states, global V-channel metrics,
    and global policy envelopes to compute global system coherence H_hol^global(t).
    """

    def __init__(
        self,
        w_a: float = 0.4,
        w_c: float = 0.4,
        w_r: float = 0.2
    ) -> None:
        """
        Initializes the HierarchicalHolisticGovernor.

        Args:
            w_a: Weight for aggregate cluster activation.
            w_c: Weight for cluster coherence.
            w_r: Weight for cluster brim energy / tension.
        """
        self.w_a = w_a
        self.w_c = w_c
        self.w_r = w_r
        self.h_global_history: List[float] = []
        self.brim_envelope = BrimEnvelope()

    def compute_global_holistic_field(self, clusters: List[UFOCluster]) -> float:
        """
        Computes the global Holistic Governor field:
        H_hol_global(t) = (1/M) * sum_c [ (w_a * A_cluster^c + w_c * C_cluster^c - w_r * E_cluster^c) * P_cluster^c ]

        Args:
            clusters: List of UFOCluster instances.

        Returns:
            Scalar representation of global multi-cluster field coherence.
        """
        if not clusters:
            return 1.0

        field_sum = 0.0
        for cluster in clusters:
            # 1. A_cluster^c: Average cluster activation
            avg_act = float(np.mean([s.activation for s in cluster.membrane.strings]))

            # 2. C_cluster^c: Cluster coherence
            c_val = cluster.compute_cluster_coherence()

            # 3. E_cluster^c: Brim energy of the cluster-level membrane
            brim_energy = self.brim_envelope.compute_brim_energy(cluster.membrane, cluster.boundary, samples=12)

            # 4. P_cluster^c: Average policy compliance of cluster's active agents
            active_agents = [a for a in cluster.agents if a.shard.state != ShardState.QUARANTINED]
            if active_agents:
                p_compliance = sum(a.shard.policy_compliance for a in active_agents) / len(active_agents)
            else:
                p_compliance = 1.0

            cluster_term = (self.w_a * avg_act + self.w_c * c_val - self.w_r * brim_energy) * p_compliance
            field_sum += cluster_term

        h_global = field_sum / len(clusters)
        self.h_global_history.append(h_global)
        return h_global


class HierarchicalSAOPromotion:
    """
    Coordinates multi-tier Symmetric Ascension Operator (SAO) promotions:
    - Tier 1: Agent Semantic Memory -> Cluster-Local Semantic Memory (MeshSemanticMemory)
    - Tier 2: Cluster-Local Semantic Memory -> Global Semantic Memory (MeshSemanticMemory)
    """

    def __init__(self, ledger: Optional[ResidualLedger] = None) -> None:
        self.ledger = ledger if ledger is not None else ResidualLedger()

    def execute_agent_to_cluster_promotion(
        self,
        agent: UFOAgent,
        cluster: UFOCluster,
        key: str
    ) -> PromotionResult:
        """
        Promotes a local agent semantic memory record to the cluster-local shared memory.

        Args:
            agent: Source agent.
            cluster: Target cluster.
            key: Memory record key.

        Returns:
            PromotionResult containing success flag, promoted record and residual p_SAO.
        """
        if not hasattr(agent, "semantic_memory") or agent.semantic_memory is None:
            return PromotionResult(success=False, record=None, error_message="Agent has no semantic memory.")

        # Create admissibility context based on cluster metrics
        coherence = cluster.compute_cluster_coherence()
        quarantined = agent.shard.state == ShardState.QUARANTINED
        cost_factor = agent.shard.cost_factor

        # Estimate local stability energy from agent's latest history
        stability_energy = agent.residual_history[-1] if agent.residual_history else 0.0

        context = AdmissibilityContext(
            coherence=coherence,
            quarantined=quarantined,
            cost_factor=cost_factor,
            stability_energy=stability_energy
        )

        # Execute promotion using core semantic memory promotion rules
        result = promote_memory(
            mesh_memory=cluster.semantic_memory,
            agent_memory=agent.semantic_memory,
            key=key,
            context=context,
            global_ledger=self.ledger
        )

        return result

    def execute_cluster_to_global_promotion(
        self,
        cluster: UFOCluster,
        global_mesh_memory: MeshSemanticMemory,
        key: str
    ) -> PromotionResult:
        """
        Promotes a cluster-local memory record to the global shared memory layer.

        Args:
            cluster: Source cluster.
            global_mesh_memory: Destination global memory layer.
            key: Memory record key.

        Returns:
            PromotionResult.
        """
        if key not in cluster.semantic_memory.global_store:
            return PromotionResult(success=False, record=None, error_message="Key not found in cluster memory.")

        cluster_rec = cluster.semantic_memory.global_store[key]

        # Aggregate cluster metrics
        active_agents = [a for a in cluster.agents if a.shard.state != ShardState.QUARANTINED]
        if active_agents:
            avg_compliance = sum(a.shard.policy_compliance for a in active_agents) / len(active_agents)
            avg_trust = sum(a.shard.trust_score for a in active_agents) / len(active_agents)
        else:
            avg_compliance = 1.0
            avg_trust = 1.0

        # Create admissibility context
        cluster_coherence = cluster.compute_cluster_coherence()

        # Admissibility Check: Reject if cluster compliance or trust is too low
        if avg_compliance < 0.4 or avg_trust < 0.4:
            rec_id = f"cluster_promote_rejected_{cluster.cluster_id}_{int(time.time() * 1000)}"
            self.ledger.log_failure(
                record_id=rec_id,
                error_type="cluster_promotion_rejected",
                shard_id=cluster.cluster_id,
                severity="high",
                details={
                    "key": key,
                    "coherence": cluster_coherence,
                    "compliance": avg_compliance
                }
            )
            return PromotionResult(
                success=False,
                record=None,
                error_message="Cluster metrics violate global admissibility."
            )

        # Shared global store overwrite prevention
        prev_residual: Optional[ResidualRecord] = None
        if key in global_mesh_memory.global_store:
            prev_shared = global_mesh_memory.global_store[key]
            res_id = f"residual_global_mesh_{cluster.cluster_id}_{key}_{int(time.time() * 1000)}"
            prev_residual = ResidualRecord(
                record_id=res_id,
                error_type="global_mesh_memory_overwrite_residual",
                shard_id=cluster.cluster_id,
                severity="medium",
                details={
                    "key": key,
                    "overwritten_value": prev_shared.value,
                    "origin_cluster": prev_shared.origin_agent_id
                },
                timestamp=time.time(),
                escalation_tier="none"
            )
            self.ledger.log_failure(
                record_id=prev_residual.record_id,
                error_type=prev_residual.error_type,
                shard_id=prev_residual.shard_id,
                severity=prev_residual.severity,
                details=prev_residual.details,
                timestamp=prev_residual.timestamp,
                escalation_tier=prev_residual.escalation_tier
            )

        # Compute Ascension Residual p_SAO^cluster
        p_sao = max(0.0, 1.0 - cluster_coherence)

        created_at_val = (
            global_mesh_memory.global_store[key].created_at
            if key in global_mesh_memory.global_store
            else time.time()
        )
        promoted_rec = MemoryRecord(
            key=key,
            value=cluster_rec.value,
            tags=cluster_rec.tags,
            created_at=created_at_val,
            updated_at=time.time(),
            origin_agent_id=cluster.cluster_id,
            policy_envelope=cluster_rec.policy_envelope,
            residual_state=prev_residual
        )

        global_mesh_memory.global_store[key] = promoted_rec
        global_mesh_memory.add_history(cluster.cluster_id, "promote", key, "success")
        global_mesh_memory.curvature_state.update_on_write(size_factor=0.2)

        return PromotionResult(success=True, record=promoted_rec, p_sao=p_sao)
