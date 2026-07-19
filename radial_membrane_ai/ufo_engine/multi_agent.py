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

        # Bind Policy-Bound Semantic Memory Layer
        from radial_membrane_ai.semantic_memory.integration import bind_to_mesh, bind_to_agent
        bind_to_mesh(self)
        for agent in self.agents:
            bind_to_agent(agent)

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
        Runs a single multi-agent tick cycle.

        Steps:
        1. Local agent step execution.
        2. Inter-agent channel propagation (V-Channels).
        3. Global Holistic Governor field H_hol calculation.
        4. Mesh coherence score C_mesh calculation.
        5. Band transition checking and intervention execution.
        6. SAO promotions of shared paired-state representations.
        7. Audit mesh & log results.

        Returns:
            The determined stability band for the tick ("green", "yellow", or "red").
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

            # Apply Hysteresis Inertial Damping on resulting activations: A_new = (1 - alpha) * A_instant + alpha * A_prev
            if t_state is not None and len(agent.activation_history) > 0:
                prior_act = agent.activation_history[-1]
                # Previous tension acts as damping on activation updates
                alpha = 0.3 * t_state.get_normalized_tension_history()
                for idx, s in enumerate(agent.membrane.strings):
                    s.activation = (1.0 - alpha) * s.activation + alpha * float(prior_act[idx])

            # Update Agent-level Temporal Membrane State
            if t_state is not None:
                max_curv = float(max([agent.boundary.curvature(s.theta) for s in agent.membrane.strings]))
                max_tens = float(max([s.tension for s in agent.membrane.strings])) if any(s.tension > 0 for s in agent.membrane.strings) else 0.0

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
                sao_intensity = 0.0  # Filled dynamically during promotion
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

                # Agent recovery rule: gradual relaxation based on local stability
                is_stable = (agent.shard.quality_score >= 0.7)
                recovery_mu = t_state.update_recovery_dynamics(is_green=is_stable, mu=0.1)
                t_state.apply_curvature_drift(agent.boundary, mu=recovery_mu)

        # 2. Propagate inter-agent coupling V-channels
        for channel in self.channels:
            channel.propagate()

        # 3. Compute global fields
        h_hol = self.global_governor.compute_global_holistic_field(self.agents)
        self.h_hol_history.append(h_hol)

        # 4. Compute mesh coherence score C_mesh(t)
        # We adjust weights according to our configured cost weighting
        # We also scale by quality factor / cost weights to enforce proper dynamic coherence
        c_mesh = self.mesh_governance.compute_mesh_coherence(
            w_q=0.4, w_t=0.3, w_e=0.2, w_r=0.2, w_p=0.2, w_l=0.1, w_f=0.2
        )
        self.c_mesh_history.append(c_mesh)

        for agent in self.agents:
            self.agent_coherences[agent.agent_id].append(agent.shard.quality_score)

        # Update global temporal state of the engine
        if self.temporal_state is not None and active_agents:
            agent_t_states = [a.membrane.temporal_state for a in active_agents if getattr(a.membrane, "temporal_state", None) is not None]
            if agent_t_states:
                mean_tension = float(np.mean([ts.accumulated_tension for ts in agent_t_states]))
                max_tension = float(np.max([ts.accumulated_tension for ts in agent_t_states]))
                mean_curv = float(np.mean([ts.prior_curvature for ts in agent_t_states]))
                mean_cl = float(np.mean([ts.admissibility_history[-1] for ts in agent_t_states if ts.admissibility_history]))

                # Global mesh/cluster temporal updates
                self.temporal_state.update_tick_history(mean_curv, mean_tension, mean_cl)
                self.temporal_state.accumulated_tension = mean_tension

                if max_tension > 2.0:
                    self.interventions.append(f"Global Temporal Tension Warning: max={max_tension:.2f}, mean={mean_tension:.2f}")

                # Scale dynamic coherence based on global temporal tension
                c_mesh_eff = max(0.0, min(1.0, c_mesh - 0.25 * self.temporal_state.accumulated_tension))
            else:
                c_mesh_eff = c_mesh
        else:
            c_mesh_eff = c_mesh

        # 5. Stability Band Check & Interventions (evaluated on effective coherence)
        if c_mesh_eff >= self.band_config.c_green:
            band = "green"
            self.interventions.append("Green Band (Nominal): Stable multi-agent routing operating optimally.")
        elif c_mesh_eff >= self.band_config.c_red:
            band = "yellow"
            self.interventions.append(
                f"Yellow Band (Soft Intervention): Coherence={c_mesh_eff:.4f}. "
                "Damping activations and rebalancing routes."
            )
            # Soft interventions: damp active strings slightly on all non-quarantined agents
            for agent in self.agents:
                if agent.shard.state != ShardState.QUARANTINED:
                    for s in agent.membrane.strings:
                        s.activation *= 0.85
                        s.radius *= 0.95
        else:
            band = "red"
            self.interventions.append(
                f"Red Band (Hard Intervention): Coherence={c_mesh_eff:.4f}. "
                "Throttling, shard quarantine, and fallback checks."
            )
            # Hard interventions: throttle activations heavily
            for agent in self.agents:
                if agent.shard.state != ShardState.QUARANTINED:
                    for s in agent.membrane.strings:
                        s.activation *= 0.5
                        s.radius *= 0.8

            # Run mesh audit to isolate/quarantine low performing/violating agents
            self.mesh_governance.run_mesh_audit()

            # Fallback checks: If only one agent is left active, log fallback
            active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
            if len(active_agents) == 1:
                self.interventions.append(f"Fallback triggered: Sole active agent is {active_agents[0].agent_id}.")

        self.band_history.append(band)

        # 6. Execute SAO promotions on paired agents
        # Record promotion residuals and audit failures
        p_sao_sum = 0.0
        promo_count = 0
        if len(self.agents) >= 2:
            for i in range(len(self.agents) - 1):
                agent_l = self.agents[i]
                agent_r = self.agents[i + 1]
                if agent_l.shard.state != ShardState.QUARANTINED and agent_r.shard.state != ShardState.QUARANTINED:
                    # Dynamically adjust threshold based on mesh coherence
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

                    # Call on_sao_promotion when successful promotion occurs
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
