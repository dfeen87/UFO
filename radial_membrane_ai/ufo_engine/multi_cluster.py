# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

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

# Kernel Regime imports
from radial_membrane_ai.kernel_regimes.manager import RegimeManager
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.utils import set_deterministic_env


class MultiClusterEngine:
    """
    Engine orchestrating a distributed multi-cluster system.
    Each cluster runs its own local multi-agent simulation, and clusters
    interact via cross-cluster V-channels and policy envelopes under a
    hierarchical Holistic Governor, supporting governed behavioral regimes.
    """

    def __init__(self, seed: int = 0) -> None:
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

        # Call deterministic environment seeding
        set_deterministic_env(seed)

        # Kernel Regime Expansion Layer components
        self.regime_manager = RegimeManager()
        self.tick_count: int = 0

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
        Executes a single distributed multi-cluster tick cycle with integrated collective reasoning
        and multiple governed behavioral regimes.
        """
        self.tick_count += 1

        # Check Cluster Quarantine timers
        for cluster in self.clusters.values():
            if getattr(cluster, "quarantine_timer", 0) > 0:
                cluster.quarantine_timer -= 1
                if cluster.quarantine_timer == 0:
                    self.interventions.append(f"Cluster {cluster.cluster_id} released from quarantine.")

        # Ensure default excitation format is array
        excitation = np.array(default_excitation, dtype=np.float64)

        # 0. Backup State
        back_t = self.temporal_state
        backup = {
            "clusters": [],
            "global_memory": dict(self.global_mesh_memory.global_store) if self.global_mesh_memory else {},
            "t_state_accumulated_tension": (
                back_t.accumulated_tension if hasattr(self, "temporal_state") else 0.0
            ),
            "t_state_consecutive_ticks": (
                back_t.consecutive_admissible_ticks if hasattr(self, "temporal_state") else 0
            ),
            "t_state_green_ticks": (
                back_t.green_ticks_count if hasattr(self, "temporal_state") else 0
            ),
            "t_state_tension_history": (
                list(back_t.tension_history) if hasattr(self, "temporal_state") else []
            ),
            "t_state_curvature_history": (
                list(back_t.curvature_history) if hasattr(self, "temporal_state") else []
            ),
            "t_state_admissibility_history": (
                list(back_t.admissibility_history) if hasattr(self, "temporal_state") else []
            ),
            "h_global_history": list(self.h_global_history),
            "c_global_history": list(self.c_global_history),
            "global_band_history": list(self.global_band_history),
            "interventions": list(self.interventions),
        }

        for c_id, cluster in self.clusters.items():
            cluster_backup = {
                "cluster_id": c_id,
                "stability_band": cluster.stability_band,
                "tension_metric": cluster.tension_metric,
                "local_memory": (
                    dict(cluster.semantic_memory.global_store)
                    if hasattr(cluster, "semantic_memory") and cluster.semantic_memory else {}
                ),
                "agents": []
            }
            for agent in cluster.agents:
                a_ts = agent.membrane.temporal_state if hasattr(agent.membrane, "temporal_state") else None
                a_sm = agent.semantic_memory if hasattr(agent, "semantic_memory") else None
                agent_backup = {
                    "agent_id": agent.agent_id,
                    "activations": [s.activation for s in agent.membrane.strings],
                    "radii": [s.radius for s in agent.membrane.strings],
                    "deviations": dict(agent.boundary.radius_deviation),
                    "state": agent.shard.state,
                    "trust_score": agent.shard.trust_score,
                    "policy_compliance": agent.shard.policy_compliance,
                    "coherence_history": list(agent.coherence_history),
                    "activation_history": list(agent.activation_history),
                    "residual_history": list(agent.residual_history),
                    "t_state_accumulated_tension": a_ts.accumulated_tension if a_ts else 0.0,
                    "t_state_consecutive_ticks": a_ts.consecutive_admissible_ticks if a_ts else 0,
                    "t_state_green_ticks": a_ts.green_ticks_count if a_ts else 0,
                    "t_state_tension_history": list(a_ts.tension_history) if a_ts else [],
                    "t_state_curvature_history": list(a_ts.curvature_history) if a_ts else [],
                    "t_state_admissibility_history": list(a_ts.admissibility_history) if a_ts else [],
                    "semantic_memory_store": dict(a_sm.local_store) if a_sm else {},
                    "semantic_memory_history": list(a_sm.history) if a_sm else [],
                }
                cluster_backup["agents"].append(agent_backup)  # type: ignore
            backup["clusters"].append(cluster_backup)  # type: ignore

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
        original_get_radii = {}

        try:
            for cluster in self.clusters.values():
                if getattr(cluster, "quarantine_timer", 0) > 0:
                    continue

                # Evaluate active cluster regime
                cluster_regime = self.regime_manager.get_regime_for_cluster(cluster.cluster_id)
                if cluster_regime.regime_type == KernelRegimeType.MULTI_PHASE:
                    c_t_state = getattr(cluster.membrane, "temporal_state", None)
                    c_t_bar = c_t_state.accumulated_tension if c_t_state is not None else 0.0
                    self.regime_manager.tick_multiphase_state(cluster, c_t_bar)

                for agent in cluster.agents:
                    if agent.shard.state == ShardState.QUARANTINED:
                        continue

                    t_state = getattr(agent.membrane, "temporal_state", None)

                    # Step agent
                    agent.step(task_value, excitation)

                    # Evaluate active regime
                    active_regime = self.regime_manager.get_regime_for_agent(agent.agent_id)
                    temp_rule_res = active_regime.temporal_rule(agent, cluster, self)
                    disable_hyst = temp_rule_res.get("disable_hysteresis", False)

                    # Apply Hysteresis Inertial Damping on resulting activations
                    if not disable_hyst and t_state is not None and len(agent.activation_history) > 0:
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

                    # Evaluate agent switching triggers
                    t_bar_eval = t_state.accumulated_tension if t_state is not None else 0.0
                    coherence_eval = agent.shard.quality_score
                    current_band_eval = cluster.stability_band
                    max_curv_eval = (
                        float(max([agent.boundary.curvature(s.theta) for s in agent.membrane.strings]))
                        if t_state is not None else 0.0
                    )

                    active_regime = self.regime_manager.evaluate_switching_triggers(
                        entity=agent,
                        t_bar=t_bar_eval,
                        coherence=coherence_eval,
                        stability_band=current_band_eval,
                        curvature=max_curv_eval
                    )
                    temp_rule_res = active_regime.temporal_rule(agent, cluster, self)

                    if active_regime.regime_type == KernelRegimeType.MULTI_PHASE:
                        self.regime_manager.tick_multiphase_state(agent, t_bar_eval)

                    # Apply stochastic activation noise
                    act_noise_func = temp_rule_res.get("activation_noise", None)
                    if act_noise_func is not None:
                        for s in agent.membrane.strings:
                            s.activation = max(0.0, min(1.0, s.activation + act_noise_func()))

                    # Capacity scaling
                    capacity_scale = active_regime.capacity_rule(agent, cluster, self)
                    if capacity_scale is None:
                        capacity_scale = 1.0

                    original_get_radii[agent.agent_id] = agent.boundary.get_radius
                    agent.boundary._custom_radius_scale = capacity_scale

                    # Check closure ratio rules
                    cl_res = active_regime.closure_ratio_rule(agent, cluster, self)
                    if cl_res.get("violation", False):
                        if cl_res.get("clamped_activations") is not None:
                            for idx, s in enumerate(agent.membrane.strings):
                                s.activation = cl_res["clamped_activations"][idx]
                        self.interventions.append(
                            f"Agent {agent.agent_id} Determinibility Violation: "
                            f"Clamped activations, soft rollback."
                        )
                        matching_ab = None
                        for cb_data in backup["clusters"]:  # type: ignore
                            for ab_data in cb_data["agents"]:  # type: ignore
                                if ab_data["agent_id"] == agent.agent_id:  # type: ignore
                                    matching_ab = ab_data  # type: ignore
                                    break
                        if matching_ab is not None:
                            for idx, s in enumerate(agent.membrane.strings):
                                s.activation = matching_ab["activations"][idx]  # type: ignore

                    # Update agent temporal metrics based on physics
                    if t_state is not None:
                        v_load = float(sum(s.radius for s in agent.membrane.strings))
                        c_tax = float(agent.shard.cost_factor * sum(s.cost for s in agent.membrane.strings))
                        sao_intensity = 0.0
                        mem_writes = float(len(agent.semantic_memory.history)) \
                            if hasattr(agent, "semantic_memory") else 0.0

                        tension_mult = temp_rule_res.get("tension_accumulation_multiplier", 1.0)
                        tension_noise_val = active_regime.tension_rule(agent, cluster, self)

                        t_state.update_tension_accumulation(
                            v_channel_load=v_load,
                            cost_taxonomy_contrib=c_tax,
                            sao_promotions_intensity=sao_intensity,
                            mem_writes_norm=min(1.0, mem_writes / 10.0)
                        )

                        if tension_mult != 1.0:
                            t_state.accumulated_tension *= tension_mult

                        if tension_noise_val != 0.0:
                            t_state.accumulated_tension = max(0.0, t_state.accumulated_tension + tension_noise_val)

                        # Decay, Recovery and drift
                        low_load = (v_load < 2.0)
                        t_state.decay_tension(low_load=low_load)

                        is_stable = (agent.shard.quality_score >= 0.7)
                        recovery_mu = t_state.update_recovery_dynamics(is_green=is_stable, mu=0.1)
                        t_state.apply_curvature_drift(agent.boundary, mu=recovery_mu)

            # 3. Local agent intra-cluster V-channel coupling
            for cluster_id, channels in self.cluster_coupling_channels.items():
                for channel in channels:
                    channel.propagate()

            # 4. Cluster-level updates (aggregate membranes & update stability bands)
            for cluster in self.clusters.values():
                if getattr(cluster, "quarantine_timer", 0) > 0:
                    continue
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
                    if getattr(c, "quarantine_timer", 0) > 0:
                        continue
                    ct_state = getattr(c.membrane, "temporal_state", None)
                    if ct_state is not None:
                        cluster_tensions.append(ct_state.accumulated_tension)

                global_tension = float(np.mean(cluster_tensions)) if cluster_tensions else 0.0
                global_warning = float(np.max(cluster_tensions)) if cluster_tensions else 0.0

                valid_curvs = [
                    c.membrane.temporal_state.prior_curvature
                    for c in self.clusters.values()
                    if getattr(c.membrane, "temporal_state", None) is not None
                    and getattr(c, "quarantine_timer", 0) == 0
                ]
                global_curv = float(np.mean(valid_curvs)) if valid_curvs else 0.0

                self.temporal_state.update_tick_history(global_curv, global_tension, 0.0)
                self.temporal_state.accumulated_tension = global_tension

                if global_warning > 2.0:
                    self.interventions.append(
                        f"Global Mesh Temporal Warning: max cluster tension={global_warning:.2f}"
                    )

                avg_tension = avg_tension + 0.5 * self.temporal_state.accumulated_tension

            # Evaluate global switching triggers
            global_regime = self.regime_manager.evaluate_switching_triggers(
                entity=None,
                t_bar=self.temporal_state.accumulated_tension if self.temporal_state is not None else 0.0,
                coherence=c_global,
                stability_band=self.global_band_history[-1] if self.global_band_history else "green"
            )

            # Stability thresholds multiplier
            glob_stab_res = global_regime.stability_rule(None, None, self)
            thresh_mult = glob_stab_res.get("threshold_multiplier", 1.0)
            effective_c_green = 0.7 * thresh_mult
            effective_c_red = 0.4 * thresh_mult

            # Enforce global stability band transitions
            if c_global >= effective_c_green and avg_tension < 1.0:
                band = "green"
                self.interventions.append("Global Mesh in Green Band (Nominal): All clusters stable.")
            elif c_global >= effective_c_red and avg_tension < 2.5:
                band = "yellow"
                self.interventions.append(
                    f"Global Mesh in Yellow Band (Soft Intervention): Coherence={c_global:.4f}."
                )
                for cluster in self.clusters.values():
                    if getattr(cluster, "quarantine_timer", 0) > 0:
                        continue
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
                    if getattr(cluster, "quarantine_timer", 0) > 0:
                        continue
                    for agent in cluster.agents:
                        if agent.shard.state != ShardState.QUARANTINED:
                            for s in agent.membrane.strings:
                                s.activation *= 0.5
                                s.radius *= 0.8
                    cluster.mesh_governance.run_mesh_audit()

            self.global_band_history.append(band)

            # Evaluate global stability rule validity based on newly determined band
            glob_stab_res = global_regime.stability_rule(None, None, self)

            # Global Shared Memory Tension escalation logging
            tot_tension = self.global_mesh_memory.curvature_state.tension
            if tot_tension > 1.5:
                self.interventions.append(
                    f"Global Shared Memory Tension Warning: {tot_tension:.2f}."
                )

            # 8. COLLECTIVE REASONING STEPS INTEGRATION FOR MULTI-CLUSTER ENGINE
            all_agents = [
                a for c in self.clusters.values() for a in c.agents if getattr(c, "quarantine_timer", 0) == 0
            ]
            all_clusters = [c for c in self.clusters.values() if getattr(c, "quarantine_timer", 0) == 0]

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
                    participating_agent_ids={
                        a.agent_id for a in all_agents if a.shard.state != ShardState.QUARANTINED
                    },
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
                    for cluster in all_clusters:
                        # Check SAO restrictions
                        reg = self.regime_manager.get_regime_for_cluster(cluster.cluster_id)
                        s_rule = reg.sao_rule(None, cluster, self)

                        # Cluster promotion represents mid-to-long-range
                        allowed = s_rule.get("allowed_ranges", {"short", "mid", "long"})
                        blocked = ("mid" not in allowed)
                        if s_rule.get("only_at_phase_boundary", False) and not s_rule.get("is_boundary", False):
                            blocked = True
                        if s_rule.get("restrict_to_cluster", False):
                            # Restricted to cluster level: cannot promote to global!
                            blocked = True

                        if not blocked:
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
                    self.interventions.append(
                        f"Global Collective Reasoning Rollback Triggered. Violations: {violations}"
                    )

            # 9. Evaluate multi-cluster stability violations and rollback
            global_rollback = False
            for cluster in self.clusters.values():
                if getattr(cluster, "quarantine_timer", 0) > 0:
                    continue
                reg = self.regime_manager.get_regime_for_cluster(cluster.cluster_id)
                st_res = reg.stability_rule(None, cluster, self)
                if not st_res.get("valid", True):
                    global_rollback = True
                    if st_res.get("quarantine", False):
                        cluster.quarantine_timer = 5
                        cluster.stability_band = "red"
                        self.interventions.append(
                            f"Cluster {cluster.cluster_id} placed under quarantine for stability violation."
                        )

            # Check global stability rule violation
            if not glob_stab_res.get("valid", True):
                global_rollback = True

            if global_rollback:
                # Restore multi-cluster backup
                for cb in backup["clusters"]:  # type: ignore
                    cl = self.clusters[cb["cluster_id"]]  # type: ignore
                    cl.stability_band = cb["stability_band"]  # type: ignore
                    cl.tension_metric = cb["tension_metric"]  # type: ignore
                    if hasattr(cl, "semantic_memory") and cl.semantic_memory:
                        cl.semantic_memory.global_store = dict(cb["local_memory"])  # type: ignore
                    for ab in cb["agents"]:  # type: ignore
                        ag = next(a for a in cl.agents if a.agent_id == ab["agent_id"])  # type: ignore
                        for i, s in enumerate(ag.membrane.strings):
                            s.activation = ab["activations"][i]  # type: ignore
                            s.radius = ab["radii"][i]  # type: ignore
                        ag.boundary.radius_deviation = dict(ab["deviations"])  # type: ignore
                        ag.shard.state = ab["state"]  # type: ignore
                        ag.shard.trust_score = ab["trust_score"]  # type: ignore
                        ag.shard.policy_compliance = ab["policy_compliance"]  # type: ignore
                        ag.coherence_history = list(ab["coherence_history"])  # type: ignore
                        ag.activation_history = list(ab["activation_history"])  # type: ignore
                        ag.residual_history = list(ab["residual_history"])  # type: ignore
                        if hasattr(ag.membrane, "temporal_state"):
                            ag.membrane.temporal_state.accumulated_tension = (
                                ab["t_state_accumulated_tension"]  # type: ignore
                            )
                            ag.membrane.temporal_state.consecutive_admissible_ticks = (
                                ab["t_state_consecutive_ticks"]  # type: ignore
                            )
                            ag.membrane.temporal_state.green_ticks_count = (
                                ab["t_state_green_ticks"]  # type: ignore
                            )
                            ag.membrane.temporal_state.tension_history.clear()
                            ag.membrane.temporal_state.tension_history.extend(
                                ab["t_state_tension_history"]  # type: ignore
                            )
                            ag.membrane.temporal_state.curvature_history.clear()
                            ag.membrane.temporal_state.curvature_history.extend(
                                ab["t_state_curvature_history"]  # type: ignore
                            )
                            ag.membrane.temporal_state.admissibility_history.clear()
                            ag.membrane.temporal_state.admissibility_history.extend(
                                ab["t_state_admissibility_history"]  # type: ignore
                            )
                        if hasattr(ag, "semantic_memory") and ag.semantic_memory:
                            ag.semantic_memory.local_store = dict(
                                ab["semantic_memory_store"]  # type: ignore
                            )
                            ag.semantic_memory.history = list(
                                ab["semantic_memory_history"]  # type: ignore
                            )

                if self.global_mesh_memory:
                    self.global_mesh_memory.global_store = dict(
                        backup["global_memory"]  # type: ignore
                    )
                if hasattr(self, "temporal_state"):
                    self.temporal_state.accumulated_tension = (
                        backup["t_state_accumulated_tension"]  # type: ignore
                    )
                    self.temporal_state.consecutive_admissible_ticks = (
                        backup["t_state_consecutive_ticks"]  # type: ignore
                    )
                    self.temporal_state.green_ticks_count = (
                        backup["t_state_green_ticks"]  # type: ignore
                    )
                    self.temporal_state.tension_history.clear()
                    self.temporal_state.tension_history.extend(
                        backup["t_state_tension_history"]  # type: ignore
                    )
                    self.temporal_state.curvature_history.clear()
                    self.temporal_state.curvature_history.extend(
                        backup["t_state_curvature_history"]  # type: ignore
                    )
                    self.temporal_state.admissibility_history.clear()
                    self.temporal_state.admissibility_history.extend(
                        backup["t_state_admissibility_history"]  # type: ignore
                    )
                self.h_global_history = list(backup["h_global_history"])  # type: ignore
                self.c_global_history = list(backup["c_global_history"])  # type: ignore
                self.global_band_history = list(backup["global_band_history"])  # type: ignore
                self.interventions = list(backup["interventions"])  # type: ignore

                self.interventions.append(
                    f"Regime Stability Violation: Global Multi-Cluster Rollback triggered (band was '{band}')."
                )

            # Post-tick invariant assertions and near-violation warning logs
            hard_energy_limit = 15.0
            hard_tension_limit = 10.0
            hard_stiffness_limit = 2.0
            hard_curvature_limit = 50.0
            hard_deviation_limit = 5.0

            # Check Lyapunov energy
            latest_energy = self.h_global_history[-1] if self.h_global_history else 0.0
            if latest_energy > hard_energy_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Multi-cluster energy ({latest_energy:.4f}) "
                    f"exceeded hard limit ({hard_energy_limit})."
                )
            elif latest_energy >= 0.95 * hard_energy_limit:
                self.interventions.append(
                    f"Near-violation warning: Multi-cluster energy ({latest_energy:.4f}) "
                    f"is within 5% of hard limit ({hard_energy_limit})."
                )

            # Check Tension
            latest_tension = self.temporal_state.accumulated_tension if self.temporal_state is not None else 0.0
            if latest_tension > hard_tension_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Global mesh tension ({latest_tension:.4f}) "
                    f"exceeded hard limit ({hard_tension_limit})."
                )
            elif latest_tension >= 0.95 * hard_tension_limit:
                self.interventions.append(
                    f"Near-violation warning: Global mesh tension ({latest_tension:.4f}) "
                    f"is within 5% of hard limit ({hard_tension_limit})."
                )

            # Check active clusters and agents within them
            active_clusters = [c for c in self.clusters.values() if getattr(c, "quarantine_timer", 0) == 0]
            active_agents = [a for c in active_clusters for a in c.agents if a.shard.state != ShardState.QUARANTINED]
            if active_agents:
                max_stiffness = float(max(s.stiffness for a in active_agents for s in a.membrane.strings))
                if max_stiffness > hard_stiffness_limit:
                    from radial_membrane_ai.exceptions import GovernanceError
                    raise GovernanceError(
                        f"Governed bound violated: Stiffness ({max_stiffness:.4f}) "
                        f"exceeded hard limit ({hard_stiffness_limit})."
                    )
                elif max_stiffness >= 0.95 * hard_stiffness_limit:
                    self.interventions.append(
                        f"Near-violation warning: Stiffness ({max_stiffness:.4f}) "
                        f"is within 5% of hard limit ({hard_stiffness_limit})."
                    )

                max_curvature = float(max(
                    abs(a.boundary.curvature(s.theta)) for a in active_agents for s in a.membrane.strings
                ))
                if max_curvature > hard_curvature_limit:
                    from radial_membrane_ai.exceptions import GovernanceError
                    raise GovernanceError(
                        f"Governed bound violated: Boundary curvature ({max_curvature:.4f}) "
                        f"exceeded hard limit ({hard_curvature_limit})."
                    )
                elif max_curvature >= 0.95 * hard_curvature_limit:
                    self.interventions.append(
                        f"Near-violation warning: Boundary curvature ({max_curvature:.4f}) "
                        f"is within 5% of hard limit ({hard_curvature_limit})."
                    )

                max_dev = float(max(abs(dev) for a in active_agents for dev in a.boundary.radius_deviation.values()))
                if max_dev > hard_deviation_limit:
                    from radial_membrane_ai.exceptions import GovernanceError
                    raise GovernanceError(
                        f"Governed bound violated: Radius deviation ({max_dev:.4f}) "
                        f"exceeded hard limit ({hard_deviation_limit})."
                    )
                elif max_dev >= 0.95 * hard_deviation_limit:
                    self.interventions.append(
                        f"Near-violation warning: Radius deviation ({max_dev:.4f}) "
                        f"is within 5% of hard limit ({hard_deviation_limit})."
                    )

            # Check Coherence
            latest_coh = self.c_global_history[-1] if self.c_global_history else 1.0
            if latest_coh < 0.0:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Coherence ({latest_coh:.4f}) "
                    f"fell below lower limit (0.0)."
                )

        finally:
            for cluster in self.clusters.values():
                for agent in cluster.agents:
                    agent.boundary._custom_radius_scale = 1.0

        return band
