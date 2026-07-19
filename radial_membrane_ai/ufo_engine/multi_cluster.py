"""
Multi-Cluster Governed Simulation Engine for the distributed U.F.O. architecture.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.multi_agent.coupling import InterAgentVChannel
from radial_membrane_ai.multi_agent.cross_channels import CrossClusterVChannel, ClusterPolicyEnvelope
from radial_membrane_ai.multi_agent.hierarchical_governance import (
    HierarchicalHolisticGovernor,
    HierarchicalSAOPromotion
)
from radial_membrane_ai.semantic_memory.core import MeshSemanticMemory
from radial_membrane_ai.shard import ShardState


class MultiClusterEngine:
    """
    Engine orchestrating a distributed multi-cluster system.
    Each cluster runs its own local multi-agent simulation, and clusters
    interact via cross-cluster V-channels and policy envelopes under a
    hierarchical Holistic Governor.
    """

    def __init__(self) -> None:
        """
        Initializes the Multi-Cluster Simulation Engine.
        """
        self.clusters: Dict[str, UFOCluster] = {}
        self.cross_channels: List[CrossClusterVChannel] = []
        self.cluster_coupling_channels: Dict[str, List[InterAgentVChannel]] = {}

        # Hierarchical layers
        self.global_mesh_memory = MeshSemanticMemory()
        self.hierarchical_governor = HierarchicalHolisticGovernor()
        self.hierarchical_sao = HierarchicalSAOPromotion()

        # Global history
        self.h_global_history: List[float] = []
        self.c_global_history: List[float] = []
        self.global_band_history: List[str] = []
        self.interventions: List[str] = []

    def create_cluster(
        self,
        cluster_id: str,
        role: str = "general",
        config: Optional[Dict[str, Any]] = None
    ) -> UFOCluster:
        """
        Creates a new cluster and registers it in the engine.
        """
        cluster = UFOCluster(cluster_id=cluster_id, role=role, config=config)
        self.clusters[cluster_id] = cluster
        self.cluster_coupling_channels[cluster_id] = []
        return cluster

    def assign_agent_to_cluster(self, agent: UFOAgent, cluster_id: str) -> None:
        """
        Assigns an agent to a specific cluster and refreshes coupling channels.
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster '{cluster_id}' does not exist.")

        cluster = self.clusters[cluster_id]
        cluster.add_agent(agent)

        # Ensure semantic memory is bound to agent
        from radial_membrane_ai.semantic_memory.integration import bind_to_agent
        if not hasattr(agent, "semantic_memory") or agent.semantic_memory is None:
            bind_to_agent(agent)

        # Refresh local coupling channels for this cluster
        self._refresh_cluster_coupling_channels(cluster_id)

    def _refresh_cluster_coupling_channels(self, cluster_id: str) -> None:
        """
        Configures coupled V-channels in a coupled fashion between cluster agents.
        """
        cluster = self.clusters[cluster_id]
        agents = cluster.agents
        n = len(agents)
        channels: List[InterAgentVChannel] = []

        if n >= 2:
            for i in range(n):
                src = agents[i]
                tgt = agents[(i + 1) % n]
                channels.append(InterAgentVChannel(src, tgt, "Type-W", coupling_strength=0.15))
                channels.append(InterAgentVChannel(tgt, src, "Type-T", coupling_strength=0.1))
                channels.append(InterAgentVChannel(src, tgt, "Type-R", coupling_strength=0.12))

        self.cluster_coupling_channels[cluster_id] = channels

    def add_cross_channel(
        self,
        source_cluster_id: str,
        target_cluster_id: str,
        channel_type: str = "Type-W",
        coupling_strength: float = 0.1,
        dampening: float = 0.9,
        policy_envelope: Optional[ClusterPolicyEnvelope] = None
    ) -> CrossClusterVChannel:
        """
        Adds a governed cross-cluster V-channel to the engine.
        """
        if source_cluster_id not in self.clusters:
            raise ValueError(f"Source cluster '{source_cluster_id}' does not exist.")
        if target_cluster_id not in self.clusters:
            raise ValueError(f"Target cluster '{target_cluster_id}' does not exist.")

        chan = CrossClusterVChannel(
            source_cluster=self.clusters[source_cluster_id],
            target_cluster=self.clusters[target_cluster_id],
            channel_type=channel_type,
            coupling_strength=coupling_strength,
            dampening=dampening,
            policy_envelope=policy_envelope
        )
        self.cross_channels.append(chan)
        return chan

    def query_cluster_stability(self, cluster_id: str) -> Dict[str, Any]:
        """
        Queries and returns stability, coherence, and tension metrics for a cluster.
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster '{cluster_id}' does not exist.")

        cluster = self.clusters[cluster_id]
        return {
            "cluster_id": cluster.cluster_id,
            "role": cluster.role,
            "stability_band": cluster.stability_band,
            "tension_metric": cluster.tension_metric,
            "coherence": cluster.compute_cluster_coherence(),
            "active_agents": len([a for a in cluster.agents if a.shard.state != ShardState.QUARANTINED])
        }

    def tick(self, task_value: float, default_excitation: np.ndarray) -> str:
        """
        Executes a single distributed multi-cluster tick cycle.

        Steps:
        1. Temporal decay on all local and global semantic memory curvature states.
        2. Local agent ticks within each cluster.
        3. Local agent V-channel propagation inside clusters.
        4. Cluster-level membrane and boundary state aggregation.
        5. Cross-cluster V-channel propagation.
        6. Hierarchical global governor calculations.
        7. Global stability band classification & interventions.

        Returns:
            The determined global stability band ("green", "yellow", or "red").
        """
        # Ensure default excitation format is array
        excitation = np.array(default_excitation, dtype=np.float64)

        # 1. Decay all semantic memory curvature states
        self.global_mesh_memory.curvature_state.decay(rate=0.05)
        for cluster in self.clusters.values():
            cluster.semantic_memory.curvature_state.decay(rate=0.05)
            cluster.stability_curvature.decay(rate=0.05)
            for agent in cluster.agents:
                if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                    agent.semantic_memory.curvature_state.decay(rate=0.05)

        # 2. Local agent step execution
        for cluster in self.clusters.values():
            for agent in cluster.agents:
                if agent.shard.state == ShardState.QUARANTINED:
                    continue
                agent.step(task_value, excitation)

        # 3. Local agent intra-cluster V-channel coupling
        for cluster_id, channels in self.cluster_coupling_channels.items():
            for channel in channels:
                channel.propagate()

        # 4. Cluster-level updates (aggregate membranes & update stability bands)
        for cluster in self.clusters.values():
            cluster.update_cluster_membrane()

        # 5. Cross-cluster V-channel propagation (signals crossing cluster boundaries)
        for cc_channel in self.cross_channels:
            cc_channel.propagate()

        # 6. Hierarchical global governor field calculations
        h_global = self.hierarchical_governor.compute_global_holistic_field(list(self.clusters.values()))
        self.h_global_history.append(h_global)

        # Compute global mesh coherence C_global(t)
        if self.clusters:
            c_global = float(np.mean([c.compute_cluster_coherence() for c in self.clusters.values()]))
        else:
            c_global = 1.0
        self.c_global_history.append(c_global)

        # 7. Global stability band classification and interventions
        avg_tension = float(np.mean([c.tension_metric for c in self.clusters.values()])) if self.clusters else 0.0

        # Enforce global stability band transitions
        if c_global >= 0.7 and avg_tension < 1.0:
            band = "green"
            self.interventions.append("Global Mesh in Green Band (Nominal): All clusters stable.")
        elif c_global >= 0.4 and avg_tension < 2.5:
            band = "yellow"
            self.interventions.append(
                f"Global Mesh in Yellow Band (Soft Intervention): Coherence={c_global:.4f}. "
                "Applying global excitation damping across all clusters."
            )
            # Soft intervention: damp activations slightly
            for cluster in self.clusters.values():
                for agent in cluster.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.85
                            s.radius *= 0.95
        else:
            band = "red"
            self.interventions.append(
                f"Global Mesh in Red Band (Hard Intervention): Coherence={c_global:.4f}. "
                "Throttling agents, initiating cluster audits, and hard isolation."
            )
            # Hard intervention: heavy throttling
            for cluster in self.clusters.values():
                for agent in cluster.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.5
                            s.radius *= 0.8
                # Force local cluster audit to isolate failing shards
                cluster.mesh_governance.run_mesh_audit()

        self.global_band_history.append(band)

        # Global Shared Memory Tension escalation logging
        tot_tension = self.global_mesh_memory.curvature_state.tension
        if tot_tension > 1.5:
            self.interventions.append(
                f"Global Shared Memory Tension Warning: {tot_tension:.2f}. "
                "Initiating global shard key sanitization and compliance audit."
            )

        return band
