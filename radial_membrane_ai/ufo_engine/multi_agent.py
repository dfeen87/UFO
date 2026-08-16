# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

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
from radial_membrane_ai.utils import set_deterministic_env
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

# Kernel Regime imports
from radial_membrane_ai.kernel_regimes.manager import RegimeManager
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType


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
        custom_agents: List[UFOAgent] | None = None,
        seed: int = 0,
        enable_legacy_handshake: bool = False,
        legacy_handshake_mode: str = "strict",
    ) -> None:
        """
        Initializes the Multi-Agent Engine.
        """
        self.cost_weights = cost_weights if cost_weights is not None else CostWeights()
        self.band_config = band_config if band_config is not None else StabilityBandConfig()
        self.enable_legacy_handshake = enable_legacy_handshake
        self.legacy_handshake_mode = legacy_handshake_mode
        from radial_membrane_ai.admissibility import AdmissibilityGate
        self.admissibility_gate = AdmissibilityGate(legacy_mode=legacy_handshake_mode)

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

        # Call deterministic environment seeding
        set_deterministic_env(seed)

        # Kernel Regime Expansion Layer components
        self.regime_manager = RegimeManager()
        self.tick_count: int = 0

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
        Runs a single multi-agent tick cycle with integrated collective reasoning governance
        and multiple governed behavioral regimes.
        """
        self.tick_count += 1

        # Process Quarantine decay/release for quarantined agents
        for agent in self.agents:
            if getattr(agent, "quarantine_timer", 0) > 0:
                agent.quarantine_timer -= 1
                if agent.quarantine_timer == 0 and agent.shard.state == ShardState.QUARANTINED:
                    agent.shard.state = ShardState.IDLE
                    self.interventions.append(f"Agent {agent.agent_id} released from quarantine.")

        # Ensure correct array format
        excitation = np.array(excitation, dtype=np.float64)

        # 0. Backup State
        back_t = self.temporal_state
        backup = {
            "agents_state": [],
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
            "mesh_memory_store": (
                dict(self.mesh_memory.global_store)
                if hasattr(self, "mesh_memory") and self.mesh_memory else {}
            ),
            "mesh_memory_history": (
                list(self.mesh_memory.history)
                if hasattr(self, "mesh_memory") and self.mesh_memory else []
            ),
            "ledger_records": (
                list(self.mesh_governance.ledger.records)
                if hasattr(self, "mesh_governance") else []
            ),
            "band_history": list(self.band_history),
            "h_hol_history": list(self.h_hol_history),
            "c_mesh_history": list(self.c_mesh_history),
            "interventions": list(self.interventions),
            "sao_events": list(self.sao_events),
            "residual_history": list(self.residual_history),
        }

        for agent in self.agents:
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
                "t_state_accumulated_tension": (
                    a_ts.accumulated_tension if a_ts else 0.0
                ),
                "t_state_consecutive_ticks": (
                    a_ts.consecutive_admissible_ticks if a_ts else 0
                ),
                "t_state_green_ticks": (
                    a_ts.green_ticks_count if a_ts else 0
                ),
                "t_state_tension_history": (
                    list(a_ts.tension_history) if a_ts else []
                ),
                "t_state_curvature_history": (
                    list(a_ts.curvature_history) if a_ts else []
                ),
                "t_state_admissibility_history": (
                    list(a_ts.admissibility_history) if a_ts else []
                ),
                "semantic_memory_store": dict(a_sm.local_store) if a_sm else {},
                "semantic_memory_history": list(a_sm.history) if a_sm else [],
            }
            backup["agents_state"].append(agent_backup)  # type: ignore

        # Call Semantic Memory on_tick_start hook
        from radial_membrane_ai.semantic_memory.integration import on_tick_start, on_tick_end, on_sao_promotion
        on_tick_start(self)

        from radial_membrane_ai.admissibility import angular_decomposition, dynamic_capacity_boundary_temporal
        from radial_membrane_ai.projection import closure_ratio

        # 1. Update individual agents locally (skipping quarantined)
        active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]

        try:
            for agent in active_agents:
                t_state = getattr(agent.membrane, "temporal_state", None)

                # Local updates step
                agent.step(task_value, excitation)

                # Evaluate active regime
                active_regime = self.regime_manager.get_regime_for_agent(agent.agent_id)
                temp_rule_res = active_regime.temporal_rule(agent, None, self)
                disable_hyst = temp_rule_res.get("disable_hysteresis", False)

                # Apply Hysteresis Inertial Damping on resulting activations if not disabled
                if not disable_hyst and t_state is not None and len(agent.activation_history) > 0:
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

                # 3. Regime Evaluation (agent-level)
                t_bar_eval = t_state.accumulated_tension if t_state is not None else 0.0
                coherence_eval = agent.shard.quality_score
                current_band_eval = self.band_history[-1] if self.band_history else "green"
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
                temp_rule_res = active_regime.temporal_rule(agent, None, self)

                # If Multi-Phase, update phase tick
                if active_regime.regime_type == KernelRegimeType.MULTI_PHASE:
                    self.regime_manager.tick_multiphase_state(agent, t_bar_eval)

                # 4. Regime-Specific Physics Application
                # Apply stochastic activation noise
                act_noise_func = temp_rule_res.get("activation_noise", None)
                if act_noise_func is not None:
                    for s in agent.membrane.strings:
                        s.activation = max(0.0, min(1.0, s.activation + act_noise_func()))

                # Apply capacity rule scaling
                capacity_scale = active_regime.capacity_rule(agent, None, self)
                if capacity_scale is None:
                    capacity_scale = 1.0

                # Set capacity scaling directly on the boundary
                agent.boundary._custom_radius_scale = capacity_scale

                # Check closure ratio rules
                cl_res = active_regime.closure_ratio_rule(agent, None, self)
                if cl_res.get("violation", False):
                    if cl_res.get("clamped_activations") is not None:
                        for idx, s in enumerate(agent.membrane.strings):
                            s.activation = cl_res["clamped_activations"][idx]
                    self.interventions.append(
                        f"Agent {agent.agent_id} Deterministic Admissibility Violation: "
                        f"Clamped activations, soft rollback."
                    )
                    # Restore previous activations from backup
                    matching_ab = next(
                        ab for ab in backup["agents_state"] if ab["agent_id"] == agent.agent_id  # type: ignore
                    )
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
                    tension_noise_val = active_regime.tension_rule(agent, None, self)

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

            # Evaluate global switches
            global_regime = self.regime_manager.evaluate_switching_triggers(
                entity=None,
                t_bar=self.temporal_state.accumulated_tension if self.temporal_state is not None else 0.0,
                coherence=c_mesh,
                stability_band=self.band_history[-1] if self.band_history else "green"
            )

            # Stability thresholds multiplier from global regime stability rule
            glob_stab_res = global_regime.stability_rule(None, None, self)
            thresh_mult = glob_stab_res.get("threshold_multiplier", 1.0)
            effective_c_green = self.band_config.c_green * thresh_mult
            effective_c_red = self.band_config.c_red * thresh_mult

            # Stability Band Check
            if c_mesh_eff >= effective_c_green:
                band = "green"
                self.interventions.append("Green Band (Nominal): Stable multi-agent routing operating optimally.")
            elif c_mesh_eff >= effective_c_red:
                band = "yellow"
                self.interventions.append(
                    f"Yellow Band (Soft Intervention): Coherence={c_mesh_eff:.4f}. Damping activations."
                )
                for agent in self.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.85
                            s.radius *= 0.95
            else:
                band = "red"
                self.interventions.append(
                    f"Red Band (Hard Intervention): Coherence={c_mesh_eff:.4f}. Throttling."
                )
                for agent in self.agents:
                    if agent.shard.state != ShardState.QUARANTINED:
                        for s in agent.membrane.strings:
                            s.activation *= 0.5
                            s.radius *= 0.8
                self.mesh_governance.run_mesh_audit()

                # Fallback checks
                active_agents_after_audit = [
                    a for a in self.agents if a.shard.state != ShardState.QUARANTINED
                ]
                if len(active_agents_after_audit) == 1:
                    msg = f"Fallback triggered: Sole active agent is {active_agents_after_audit[0].agent_id}."
                    self.interventions.append(msg)

            self.band_history.append(band)

            # Evaluate global stability rule validity based on newly determined band
            glob_stab_res = global_regime.stability_rule(None, None, self)

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

                # Legacy Hardware Invariant Handshake Check if enabled
                if self.enable_legacy_handshake:
                    from radial_membrane_ai.admissibility import AdmissibilityGate
                    if not hasattr(self, "admissibility_gate"):
                        self.admissibility_gate = AdmissibilityGate(legacy_mode=self.legacy_handshake_mode)
                    for ag in active_agents:
                        act_list = [s.activation for s in ag.membrane.strings]
                        ag_cost = ag.cost_history[-1].total_cost() if ag.cost_history else 0.5
                        ag_lyap = ag.governor.compute_lyapunov_energy(ag.membrane)
                        v_p = float(np.mean([abs(s.activation - 0.5) for s in ag.membrane.strings])) + 1.0
                        self.admissibility_gate.check_legacy_handshake(
                            string_activations=act_list,
                            compute_cost=ag_cost,
                            lyapunov_energy=ag_lyap,
                            v_channel_pressure=v_p,
                            legacy_mode=self.legacy_handshake_mode,
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
                    self.interventions.append(
                        f"Collective reasoning rollback triggered. Violations: {violations}"
                    )

            # 7. Execute paired agent SAO promotions (legacy fallback / alignment checks)
            p_sao_sum = 0.0
            promo_count = 0
            if len(self.agents) >= 2:
                for i in range(len(self.agents) - 1):
                    agent_l = self.agents[i]
                    agent_r = self.agents[i + 1]
                    if agent_l.shard.state != ShardState.QUARANTINED \
                            and agent_r.shard.state != ShardState.QUARANTINED:
                        reg_l = self.regime_manager.get_regime_for_agent(agent_l.agent_id)
                        reg_r = self.regime_manager.get_regime_for_agent(agent_r.agent_id)

                        sao_rule_l = reg_l.sao_rule(agent_l, None, self)
                        sao_rule_r = reg_r.sao_rule(agent_r, None, self)

                        # Paired promotions are mid-range/cluster level.
                        blocked_l = ("mid" not in sao_rule_l.get("allowed_ranges", {"short", "mid", "long"}))
                        blocked_r = ("mid" not in sao_rule_r.get("allowed_ranges", {"short", "mid", "long"}))

                        if sao_rule_l.get("only_at_phase_boundary", False) \
                                and not sao_rule_l.get("is_boundary", False):
                            blocked_l = True
                        if sao_rule_r.get("only_at_phase_boundary", False) \
                                and not sao_rule_r.get("is_boundary", False):
                            blocked_r = True

                        if blocked_l or blocked_r:
                            verdict = "block"
                            p_sao = 0.5
                        else:
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

            # 9. Rollback if stability rule is violated for any agent or globally
            # Let's check agent-level stability rules
            global_rollback = False
            for agent in active_agents:
                reg = self.regime_manager.get_regime_for_agent(agent.agent_id)
                st_res = reg.stability_rule(agent, None, self)
                if not st_res.get("valid", True):
                    global_rollback = True
                    if st_res.get("quarantine", False):
                        agent.quarantine_timer = 5
                        agent.shard.state = ShardState.QUARANTINED
                        self.interventions.append(
                            f"Agent {agent.agent_id} placed under quarantine for stability violation."
                        )

            # Check global stability rule
            if not glob_stab_res.get("valid", True):
                global_rollback = True

            if global_rollback:
                # Execute full restore of backup
                for ab in backup["agents_state"]:  # type: ignore
                    ag = next(a for a in self.agents if a.agent_id == ab["agent_id"])  # type: ignore
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
                if hasattr(self, "mesh_memory") and self.mesh_memory:
                    self.mesh_memory.global_store = dict(backup["mesh_memory_store"])  # type: ignore
                    self.mesh_memory.history = list(backup["mesh_memory_history"])  # type: ignore
                if hasattr(self, "mesh_governance"):
                    self.mesh_governance.ledger.records = list(backup["ledger_records"])  # type: ignore
                self.band_history = list(backup["band_history"])  # type: ignore
                self.h_hol_history = list(backup["h_hol_history"])  # type: ignore
                self.c_mesh_history = list(backup["c_mesh_history"])  # type: ignore
                self.interventions = list(backup["interventions"])  # type: ignore
                self.sao_events = list(backup["sao_events"])  # type: ignore
                self.residual_history = list(backup["residual_history"])  # type: ignore

                self.interventions.append(
                    f"Regime Stability Violation: Complete Rollback triggered globally (band was '{band}')."
                )

            # Post-tick invariant assertions and near-violation warning logs
            hard_energy_limit = 15.0
            hard_tension_limit = 10.0
            hard_stiffness_limit = 2.0
            hard_curvature_limit = 50.0
            hard_deviation_limit = 5.0

            # Check Lyapunov energy
            latest_energy = self.h_hol_history[-1] if self.h_hol_history else 0.0
            if latest_energy > hard_energy_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Multi-agent energy ({latest_energy:.4f}) "
                    f"exceeded hard limit ({hard_energy_limit})."
                )
            elif latest_energy >= 0.95 * hard_energy_limit:
                self.interventions.append(
                    f"Near-violation warning: Multi-agent energy ({latest_energy:.4f}) "
                    f"is within 5% of hard limit ({hard_energy_limit})."
                )

            # Check Tension
            latest_tension = self.temporal_state.accumulated_tension if self.temporal_state is not None else 0.0
            if latest_tension > hard_tension_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Global accumulated tension ({latest_tension:.4f}) "
                    f"exceeded hard limit ({hard_tension_limit})."
                )
            elif latest_tension >= 0.95 * hard_tension_limit:
                self.interventions.append(
                    f"Near-violation warning: Global accumulated tension ({latest_tension:.4f}) "
                    f"is within 5% of hard limit ({hard_tension_limit})."
                )

            # Check Stiffness, Curvature, Deviation across active agents
            active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
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
            latest_coh = self.c_mesh_history[-1] if self.c_mesh_history else 1.0
            if latest_coh < 0.0:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Coherence ({latest_coh:.4f}) "
                    f"fell below lower limit (0.0)."
                )

        finally:
            # Restore agent boundaries get_radius scale
            for agent in self.agents:
                agent.boundary._custom_radius_scale = 1.0

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
