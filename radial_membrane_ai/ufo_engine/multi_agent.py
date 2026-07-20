"""
Multi-Agent Governed Simulation Engine for the U.F.O. architecture.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.semantic_memory.core import MeshSemanticMemory

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.coupling import InterAgentVChannel, GlobalHolisticGovernor
from radial_membrane_ai.multi_agent.governance import MultiAgentMeshGovernance
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.ufo_engine.config import CostWeights, StabilityBandConfig

# Collective Reasoning imports
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.collective_reasoning.collective_admissibility import (
    CollectiveStepContext, collective_admissibility
)
from radial_membrane_ai.collective_reasoning.collective_sao import collective_sao_promote
from radial_membrane_ai.collective_reasoning.mesh_correctness import MeshCorrectness
from radial_membrane_ai.collective_reasoning.coherence import coherence_score


@dataclass
class MultiAgentRunResult:
    """
    Structured results of a multi-agent simulation run.
    """
    h_hol_history: List[float] = field(default_factory=list)
    c_mesh_history: List[float] = field(default_factory=list)
    band_history: List[str] = field(default_factory=list)
    agent_coherences: Dict[str, List[float]] = field(default_factory=dict)
    interventions: List[str] = field(default_factory=list)
    sao_events: List[Dict[str, Any]] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)


class MultiAgentEngine:
    """
    Engine running the deterministic tick loop for a multi-agent UFO system.
    Coordinates N agents, typed inter-agent channels, Holistic Governor fields,
    mesh coherence, shard isolation/reintegration, and paired-state SAO promotions.
    """

    run_result: dict[str, Any] | MultiAgentRunResult | None = None
    mesh_memory: MeshSemanticMemory

    def __init__(
        self,
        n_agents: int = 3,
        cost_weights: CostWeights | None = None,
        band_config: StabilityBandConfig | None = None,
        custom_agents: List[UFOAgent] | None = None
    ) -> None:
        """
        Initializes the Multi-Agent Engine.
        """
        self.cost_weights = cost_weights if cost_weights is not None else CostWeights()
        self.band_config = band_config if band_config is not None else StabilityBandConfig()

        if custom_agents is not None:
            self.agents = custom_agents
        else:
            # Create standard agent ensemble
            self.agents = []
            regimes = ["analytical", "creative", "balanced", "balanced"]
            sensitivities = [1.0, 2.0, 1.2, 1.0]
            for i in range(n_agents):
                self.agents.append(
                    UFOAgent(
                        agent_id=f"agent_{i+1}",
                        cost_sensitivity=sensitivities[i % len(sensitivities)],
                        kernel_regime=regimes[i % len(regimes)],
                        state=ShardState.IDLE,
                        trust_score=0.9,
                        policy_compliance=0.95,
                        latency=12.0 + i * 5.0
                    )
                )

        self.global_governor = GlobalHolisticGovernor()
        self.mesh_governance = MultiAgentMeshGovernance(self.agents)

        # Set up typed inter-agent V-channels
        self.channels: List[InterAgentVChannel] = []
        self._init_coupling_channels()

        # History trackers
        self.h_hol_history: List[float] = []
        self.c_mesh_history: List[float] = []
        self.band_history: List[str] = []
        self.agent_coherences: Dict[str, List[float]] = {a.agent_id: [] for a in self.agents}
        self.interventions: List[str] = []
        self.sao_events: List[Dict[str, Any]] = []
        self.residual_history: List[float] = []

        # Collective reasoning layer components (defaults to False for legacy backward compatibility)
        self.collective_enabled: bool = False
        self.global_envelope = PolicyEnvelope(
            min_trust=0.4,
            allowed_roles={"analytical", "creative", "balanced", "general"},
            stability_required_band="yellow"
        )
        self.correctness_checker = MeshCorrectness(self.mesh_governance.ledger)

        # Bind Policy-Bound Semantic Memory Layer
        from radial_membrane_ai.semantic_memory.integration import bind_to_mesh, bind_to_agent
        bind_to_mesh(self)
        for agent in self.agents:
            bind_to_agent(agent)
            # Give agents a default PolicyEnvelope
            setattr(agent, "policy_envelope", PolicyEnvelope(
                allowed_agents={agent.agent_id},
                min_trust=0.5
            ))

        from radial_membrane_ai.temporal import TemporalMembraneState
        self.temporal_state = TemporalMembraneState()

    def _init_coupling_channels(self) -> None:
        """
        Sets up type-W (workload), type-T (tension) and type-R (residuals) channels between agents.
        """
        n = len(self.agents)
        if n < 2:
            return

        for i in range(n):
            src = self.agents[i]
            tgt = self.agents[(i + 1) % n]
            self.channels.append(InterAgentVChannel(src, tgt, "Type-W", coupling_strength=0.15))
            self.channels.append(InterAgentVChannel(tgt, src, "Type-T", coupling_strength=0.1))
            self.channels.append(InterAgentVChannel(src, tgt, "Type-R", coupling_strength=0.12))

    def tick(self, task_value: float, excitation: np.ndarray) -> str:
        """
        Runs a single multi-agent tick cycle with integrated collective reasoning governance.
        """
        # Ensure correct array format
        excitation = np.array(excitation, dtype=np.float64)

        # Call Semantic Memory on_tick_start hook
        from radial_membrane_ai.semantic_memory.integration import on_tick_start, on_tick_end, on_sao_promotion
        on_tick_start(self)

        from radial_membrane_ai.admissibility import angular_decomposition, dynamic_capacity_boundary_temporal
        from radial_membrane_ai.projection import closure_ratio

        # 1. Update individual agents locally (skipping quarantined) with temporal dynamics
        active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
        for agent in active_agents:
            t_state = getattr(agent.membrane, "temporal_state", None)

            agent.step(task_value, excitation)

            # Apply Hysteresis Inertial Damping on resulting activations: A_new = (1-alpha)*A_instant + alpha*A_prev
            if t_state is not None and len(agent.activation_history) > 0:
                prior_act = agent.activation_history[-1]
                alpha = 0.3 * t_state.get_normalized_tension_history()
                for idx, s in enumerate(agent.membrane.strings):
                    s.activation = (1.0 - alpha) * s.activation + alpha * float(prior_act[idx])

            # 2. Update Agent-level Temporal Membrane State
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

        # 3. Propagate inter-agent coupling V-channels
        for channel in self.channels:
            channel.propagate()

        # 4. Compute global fields
        h_hol = self.global_governor.compute_global_holistic_field(self.agents)
        self.h_hol_history.append(h_hol)

        # 5. Compute mesh coherence score C_mesh(t)
        c_mesh = self.mesh_governance.compute_mesh_coherence(
            w_q=0.4, w_t=0.3, w_e=0.2, w_r=0.2, w_p=0.2, w_l=0.1, w_f=0.2
        )
        self.c_mesh_history.append(c_mesh)

        for agent in self.agents:
            self.agent_coherences[agent.agent_id].append(agent.shard.quality_score)

        # Update global temporal state of the engine
        if self.temporal_state is not None and active_agents:
            agent_t_states = [
                a.membrane.temporal_state for a in active_agents
                if getattr(a.membrane, "temporal_state", None) is not None
            ]
            if agent_t_states:
                mean_tension = float(np.mean([ts.accumulated_tension for ts in agent_t_states]))
                max_tension = float(np.max([ts.accumulated_tension for ts in agent_t_states]))
                mean_curv = float(np.mean([ts.prior_curvature for ts in agent_t_states]))
                mean_cl = float(np.mean([
                    ts.admissibility_history[-1] for ts in agent_t_states
                    if ts.admissibility_history
                ]))

                self.temporal_state.update_tick_history(mean_curv, mean_tension, mean_cl)
                self.temporal_state.accumulated_tension = mean_tension

                if max_tension > 2.0:
                    self.interventions.append(
                        f"Global Temporal Tension Warning: max={max_tension:.2f}, mean={mean_tension:.2f}"
                    )

                c_mesh_eff = max(0.0, min(1.0, c_mesh - 0.25 * self.temporal_state.accumulated_tension))
            else:
                c_mesh_eff = c_mesh
        else:
            c_mesh_eff = c_mesh

        # Stability Band Check
        if c_mesh_eff >= self.band_config.c_green:
            band = "green"
            self.interventions.append("Green Band (Nominal): Stable multi-agent routing operating optimally.")
        elif c_mesh_eff >= self.band_config.c_red:
            band = "yellow"
            self.interventions.append(
                f"Yellow Band (Soft Intervention): Coherence={c_mesh_eff:.4f}. "
                "Damping activations."
            )
            for agent in self.agents:
                if agent.shard.state != ShardState.QUARANTINED:
                    for s in agent.membrane.strings:
                        s.activation *= 0.85
                        s.radius *= 0.95
        else:
            band = "red"
            self.interventions.append(
                f"Red Band (Hard Intervention): Coherence={c_mesh_eff:.4f}. "
                "Throttling."
            )
            for agent in self.agents:
                if agent.shard.state != ShardState.QUARANTINED:
                    for s in agent.membrane.strings:
                        s.activation *= 0.5
                        s.radius *= 0.8
            self.mesh_governance.run_mesh_audit()

            # Fallback checks: If only one agent is left active, log fallback
            active_agents_after_audit = [
                a for a in self.agents if a.shard.state != ShardState.QUARANTINED
            ]
            if len(active_agents_after_audit) == 1:
                msg = f"Fallback triggered: Sole active agent is {active_agents_after_audit[0].agent_id}."
                self.interventions.append(msg)

        self.band_history.append(band)

        # 6. COLLECTIVE REASONING STEPS INTEGRATION (TICK ENHANCEMENTS)
        if self.collective_enabled and active_agents:
            # Backup state
            state_backup = self.correctness_checker.backup_state(self.agents, [], self.mesh_memory)

            # Build CollectiveStepContext
            used_tags = set()
            used_fields = set()
            for a in active_agents:
                if hasattr(a, "semantic_memory") and a.semantic_memory is not None:
                    for key, rec in a.semantic_memory.local_store.items():
                        used_tags.update(rec.tags)
                        used_fields.add(key)

            step_context = CollectiveStepContext(
                proposed_activation=excitation,
                participating_agent_ids={a.agent_id for a in active_agents},
                involved_cluster_ids=set(),
                cost_band=1 if band == "yellow" else (2 if band == "red" else 0),
                stability_band=band,
                trust_score=float(np.mean([a.shard.trust_score for a in active_agents])),
                ticks=self.temporal_state.consecutive_admissible_ticks,
                used_tags=used_tags,
                used_fields=used_fields
            )

            # A. Collective admissibility check
            is_admissible = collective_admissibility(
                agents=self.agents,
                clusters=[],
                step=step_context,
                global_envelope=self.global_envelope
            )

            # B. Coherence detection
            cluster_coherence = coherence_score(active_agents)

            # C. Collective SAO promotion (local -> cluster)
            if is_admissible and cluster_coherence >= 0.7 and band != "red":
                # Promote active records
                for agent in active_agents:
                    if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                        for key in list(agent.semantic_memory.local_store.keys()):
                            # Map stability bands
                            sb_dict = {"global": band}
                            res_sao = collective_sao_promote(
                                level="local",
                                source_entity=agent,
                                target_entity=self.mesh_memory,
                                key=key,
                                global_envelope=self.global_envelope,
                                coherence_score_val=cluster_coherence,
                                stability_bands=sb_dict,
                                current_ticks=self.temporal_state.consecutive_admissible_ticks,
                                ledger=self.mesh_governance.ledger
                            )
                            _, _, _ = res_sao

            # D. Mesh correctness checks & rollbacks
            is_correct, violations = self.correctness_checker.audit_correctness(
                self.agents, [], step_context, self.global_envelope
            )

            if not is_correct or not is_admissible:
                # Trigger complete rollback to the exact backup state
                self.correctness_checker.rollback(
                    backup=state_backup,
                    agents=self.agents,
                    clusters=[],
                    global_memory=self.mesh_memory,
                    quarantine_ticks=5,
                    failing_ids={a.agent_id for a in active_agents} if violations else None
                )
                self.interventions.append(f"Collective reasoning rollback triggered. Violations: {violations}")

        # 7. Execute paired agent SAO promotions (legacy fallback / alignment checks)
        p_sao_sum = 0.0
        promo_count = 0
        if len(self.agents) >= 2:
            for i in range(len(self.agents) - 1):
                agent_l = self.agents[i]
                agent_r = self.agents[i + 1]
                if agent_l.shard.state != ShardState.QUARANTINED and agent_r.shard.state != ShardState.QUARANTINED:
                    limit = max(0.02, 0.5 * c_mesh)
                    verdict, p_sao, proj_state = self.mesh_governance.execute_sao_promotion(
                        agent_l, agent_r, shared_capacity_limit=limit
                    )
                    sao_rec = {
                        "agent_l": agent_l.agent_id,
                        "agent_r": agent_r.agent_id,
                        "verdict": verdict,
                        "p_sao": p_sao
                    }
                    self.sao_events.append(sao_rec)
                    p_sao_sum += p_sao
                    promo_count += 1

                    if verdict == "ascend":
                        if hasattr(agent_l, "semantic_memory") and agent_l.semantic_memory is not None:
                            for key in list(agent_l.semantic_memory.local_store.keys()):
                                on_sao_promotion(agent_l.agent_id, key, self)
                        if hasattr(agent_r, "semantic_memory") and agent_r.semantic_memory is not None:
                            for key in list(agent_r.semantic_memory.local_store.keys()):
                                on_sao_promotion(agent_r.agent_id, key, self)

        avg_p_sao = (p_sao_sum / promo_count) if promo_count > 0 else 0.0
        self.residual_history.append(avg_p_sao)

        # Call Semantic Memory on_tick_end hook
        on_tick_end(self)

        return band

    def run(
        self,
        n_steps: int,
        task_value_sequence: List[float],
        excitation_sequence: List[np.ndarray]
    ) -> MultiAgentRunResult:
        """
        Runs the multi-agent engine for a sequence of steps.
        """
        for i in range(n_steps):
            t_val = task_value_sequence[i % len(task_value_sequence)]
            excite = excitation_sequence[i % len(excitation_sequence)]
            self.tick(t_val, excite)

        return MultiAgentRunResult(
            h_hol_history=list(self.h_hol_history),
            c_mesh_history=list(self.c_mesh_history),
            band_history=list(self.band_history),
            agent_coherences={k: list(v) for k, v in self.agent_coherences.items()},
            interventions=list(self.interventions),
            sao_events=list(self.sao_events),
            residual_history=list(self.residual_history)
        )
