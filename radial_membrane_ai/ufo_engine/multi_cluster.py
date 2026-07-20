"""
Multi-Cluster Governed Simulation Engine for the distributed U.F.O. architecture.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional, Set

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

# Collective reasoning imports
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.collective_reasoning.collective_admissibility import (
    CollectiveStepContext, collective_admissibility
)
from radial_membrane_ai.collective_reasoning.collective_sao import collective_sao_promote
from radial_membrane_ai.collective_reasoning.mesh_correctness import MeshCorrectness
from radial_membrane_ai.collective_reasoning.coherence import global_mesh_coherence_score


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

        # Collective reasoning layer components (defaults to False for legacy backward compatibility)
        self.collective_enabled: bool = False
        self.global_envelope = PolicyEnvelope(
            min_trust=0.4,
            allowed_roles={"planner", "critic", "safety", "analytical", "creative", "balanced", "general"},
            stability_required_band="yellow"
        )
        self.correctness_checker = MeshCorrectness(self.hierarchical_sao.ledger)

        from radial_membrane_ai.temporal import TemporalMembraneState
        self.temporal_state = TemporalMembraneState()

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
        # Give cluster standard default PolicyEnvelope
        setattr(cluster, "policy_envelope", PolicyEnvelope(
            allowed_clusters={cluster_id},
            min_trust=0.5
        ))
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
        Executes a single distributed multi-cluster tick cycle with integrated collective reasoning.
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

        from radial_membrane_ai.admissibility import angular_decomposition, dynamic_capacity_boundary_temporal
        from radial_membrane_ai.projection import closure_ratio

        # 2. Local agent step execution with temporal dynamics
        for cluster in self.clusters.values():
            for agent in cluster.agents:
                if agent.shard.state == ShardState.QUARANTINED:
                    continue

                t_state = getattr(agent.membrane, "temporal_state", None)

                agent.step(task_value, excitation)

                # Apply Hysteresis Inertial Damping on resulting activations: A_new = (1-alpha)*A_instant + alpha*A_prev
                if t_state is not None and len(agent.activation_history) > 0:
                    prior_act = agent.activation_history[-1]
                    alpha = 0.3 * t_state.get_normalized_tension_history()
                    for idx, s in enumerate(agent.membrane.strings):
                        s.activation = (1.0 - alpha) * s.activation + alpha * float(prior_act[idx])

                # Update Agent-level Temporal Membrane State
                if t_state is not None:
                    max_curv = float(max([agent.boundary.curvature(s.theta) for s in agent.membrane.strings]))
                    has_tension = any(s.tension > 0 for s in agent.membrane.strings)
                    max_tens = float(max([s.tension for s in agent.membrane.strings])) if has_tension else 0.0

                    closure_ratios = []
                    for s in agent.membrane.strings:
                        a_theta, b_theta = angular_decomposition(agent.membrane, s.theta, samples=32)
                        c_theta = dynamic_capacity_boundary_temporal(agent.boundary, s.theta, agent.membrane)
                        closure_ratios.append(closure_ratio(a_theta, b_theta, c_theta))
                    max_cl = float(max(closure_ratios)) if closure_ratios else 0.0

                    t_state.update_tick_history(max_curv, max_tens, max_cl)

                    # Temporal Tension Accumulation
                    v_load = float(sum(s.radius for s in agent.membrane.strings))
                    c_tax = float(agent.shard.cost_factor * sum(s.cost for s in agent.membrane.strings))
                    sao_intensity = 0.0
                    mem_writes = float(len(agent.semantic_memory.history)) if hasattr(agent, "semantic_memory") else 0.0
                    t_state.update_tension_accumulation(
                        v_channel_load=v_load,
                        cost_taxonomy_contrib=c_tax,
                        sao_promotions_intensity=sao_intensity,
                        mem_writes_norm=min(1.0, mem_writes / 10.0)
                    )

                    # Drift, Decay, and Recovery Dynamics
                    low_load = (v_load < 2.0)
                    t_state.decay_tension(low_load=low_load)

                    # Agent recovery rule
                    is_stable = (agent.shard.quality_score >= 0.7)
                    recovery_mu = t_state.update_recovery_dynamics(is_green=is_stable, mu=0.1)
                    t_state.apply_curvature_drift(agent.boundary, mu=recovery_mu)

        # 3. Local agent intra-cluster V-channel coupling
        for cluster_id, channels in self.cluster_coupling_channels.items():
            for channel in channels:
                channel.propagate()

        # 4. Cluster-level updates (aggregate membranes & update stability bands)
        for cluster in self.clusters.values():
            cluster.update_cluster_membrane()

            # Update Cluster-level Temporal State
            c_t_state = getattr(cluster.membrane, "temporal_state", None)
            if c_t_state is not None:
                active_agents = [a for a in cluster.agents if a.shard.state != ShardState.QUARANTINED]
                if active_agents:
                    agent_t_states = [
                        a.membrane.temporal_state for a in active_agents
                        if getattr(a.membrane, "temporal_state", None) is not None
                    ]
                    if agent_t_states:
                        mean_t = float(np.mean([ts.accumulated_tension for ts in agent_t_states]))
                        max_t = float(np.max([ts.accumulated_tension for ts in agent_t_states]))
                        cluster_tension = mean_t + 0.1 * max_t

                        cluster_curv_drift = float(np.mean([
                            cluster.boundary.curvature(s.theta) for s in cluster.membrane.strings
                        ]))

                        cluster_cl = float(np.mean([s.activation for s in cluster.membrane.strings]))

                        c_t_state.update_tick_history(cluster_curv_drift, cluster_tension, cluster_cl)
                        c_t_state.accumulated_tension = cluster_tension

        # 5. Cross-cluster V-channel propagation
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

        # Update engine-level global temporal state
        if self.temporal_state is not None and self.clusters:
            cluster_tensions = []
            for c in self.clusters.values():
                ct_state = getattr(c.membrane, "temporal_state", None)
                if ct_state is not None:
                    cluster_tensions.append(ct_state.accumulated_tension)

            global_tension = float(np.mean(cluster_tensions)) if cluster_tensions else 0.0
            global_warning = float(np.max(cluster_tensions)) if cluster_tensions else 0.0

            valid_curvs = [
                c.membrane.temporal_state.prior_curvature
                for c in self.clusters.values()
                if getattr(c.membrane, "temporal_state", None) is not None
            ]
            global_curv = float(np.mean(valid_curvs)) if valid_curvs else 0.0

            self.temporal_state.update_tick_history(global_curv, global_tension, 0.0)
            self.temporal_state.accumulated_tension = global_tension

            if global_warning > 2.0:
                self.interventions.append(
                    f"Global Mesh Temporal Warning: max cluster tension={global_warning:.2f}"
                )

            avg_tension = avg_tension + 0.5 * self.temporal_state.accumulated_tension

        # Enforce global stability band transitions
        if c_global >= 0.7 and avg_tension < 1.0:
            band = "green"
            self.interventions.append("Global Mesh in Green Band (Nominal): All clusters stable.")
        elif c_global >= 0.4 and avg_tension < 2.5:
            band = "yellow"
            self.interventions.append(
                f"Global Mesh in Yellow Band (Soft Intervention): Coherence={c_global:.4f}."
            )
            for cluster in self.clusters.values():
                for agent in cluster.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.85
                            s.radius *= 0.95
        else:
            band = "red"
            self.interventions.append(
                f"Global Mesh in Red Band (Hard Intervention): Coherence={c_global:.4f}."
            )
            for cluster in self.clusters.values():
                for agent in cluster.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.5
                            s.radius *= 0.8
                cluster.mesh_governance.run_mesh_audit()

        self.global_band_history.append(band)

        # Global Shared Memory Tension escalation logging
        tot_tension = self.global_mesh_memory.curvature_state.tension
        if tot_tension > 1.5:
            self.interventions.append(
                f"Global Shared Memory Tension Warning: {tot_tension:.2f}."
            )

        # 8. COLLECTIVE REASONING STEPS INTEGRATION FOR MULTI-CLUSTER ENGINE
        all_agents = [a for c in self.clusters.values() for a in c.agents]
        all_clusters = list(self.clusters.values())

        if self.collective_enabled and all_clusters:
            # Backup multi-cluster state
            state_backup = self.correctness_checker.backup_state(all_agents, all_clusters, self.global_mesh_memory)

            # Build CollectiveStepContext
            used_tags: Set[str] = set()
            used_fields: Set[str] = set()
            for cluster in all_clusters:
                if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                    for key, rec in cluster.semantic_memory.global_store.items():
                        used_tags.update(rec.tags)
                        used_fields.add(key)

            step_context = CollectiveStepContext(
                proposed_activation=excitation,
                participating_agent_ids={a.agent_id for a in all_agents if a.shard.state != ShardState.QUARANTINED},
                involved_cluster_ids=set(self.clusters.keys()),
                cost_band=1 if band == "yellow" else (2 if band == "red" else 0),
                stability_band=band,
                trust_score=1.0,
                ticks=self.temporal_state.consecutive_admissible_ticks,
                used_tags=used_tags,
                used_fields=used_fields
            )

            # A. Collective admissibility check (cluster + global)
            is_admissible = collective_admissibility(
                agents=all_agents,
                clusters=all_clusters,
                step=step_context,
                global_envelope=self.global_envelope
            )

            # B. Coherence detection (global)
            global_coherence, _ = global_mesh_coherence_score(all_clusters)

            # C. Collective SAO promotion (cluster -> global, mesh -> global semantic)
            if is_admissible and band != "red":
                # For each cluster, execute Cluster SAO promotion (cluster -> global mesh)
                # And Global SAO if ticks >= 50 and global_coherence >= 0.8
                for cluster in all_clusters:
                    if hasattr(cluster, "semantic_memory") and cluster.semantic_memory is not None:
                        for key in list(cluster.semantic_memory.global_store.keys()):
                            sb_dict = {"global": band, cluster.cluster_id: cluster.stability_band}

                            # Cluster SAO
                            res_sao_c = collective_sao_promote(
                                level="cluster",
                                source_entity=cluster,
                                target_entity=self.global_mesh_memory,
                                key=key,
                                global_envelope=self.global_envelope,
                                coherence_score_val=global_coherence,
                                stability_bands=sb_dict,
                                current_ticks=self.temporal_state.consecutive_admissible_ticks,
                                ledger=self.hierarchical_sao.ledger
                            )
                            success_c, _, _ = res_sao_c

                            # Global SAO
                            if success_c and self.temporal_state.consecutive_admissible_ticks >= 50:
                                collective_sao_promote(
                                    level="global",
                                    source_entity=self.global_mesh_memory,
                                    target_entity=self.global_mesh_memory,
                                    key=key,
                                    global_envelope=self.global_envelope,
                                    coherence_score_val=global_coherence,
                                    stability_bands=sb_dict,
                                    current_ticks=self.temporal_state.consecutive_admissible_ticks,
                                    ledger=self.hierarchical_sao.ledger
                                )

            # D. Mesh correctness checks & rollbacks
            is_correct, violations = self.correctness_checker.audit_correctness(
                all_agents, all_clusters, step_context, self.global_envelope
            )

            if not is_correct or not is_admissible:
                self.correctness_checker.rollback(
                    backup=state_backup,
                    agents=all_agents,
                    clusters=all_clusters,
                    global_memory=self.global_mesh_memory,
                    quarantine_ticks=5,
                    failing_ids={c.cluster_id for c in all_clusters} if violations else None
                )
                self.interventions.append(f"Global Collective Reasoning Rollback Triggered. Violations: {violations}")

        return band
