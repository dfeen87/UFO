# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Single-Agent Governed Simulation Engine for the U.F.O. architecture.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.governor import Governor
from radial_membrane_ai.envelope import BrimEnvelope
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.cost import RuntimeCostVector, reduce_avoidable_cost
from radial_membrane_ai.residuals import ResidualLedger
from radial_membrane_ai.channels import update_radius_along_channel, channel_coherence
from radial_membrane_ai.projection import (
    closure_ratio,
    project_to_admissible,
    residual_deformation
)
from radial_membrane_ai.admissibility import angular_decomposition
from radial_membrane_ai.facet import FacetVector, TensionAutomaton, TensionState
from radial_membrane_ai.coherence import closure_coherence
from radial_membrane_ai.ufo_engine.config import CostWeights, StabilityBandConfig
from radial_membrane_ai.utils import set_deterministic_env

# Kernel Regime imports
from radial_membrane_ai.kernel_regimes.manager import RegimeManager
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType


@dataclass
class SingleAgentRunResult:
    """
    Structured results of a single-agent simulation run.
    """
    v_history: List[float] = field(default_factory=list)
    band_history: List[str] = field(default_factory=list)
    cost_history: List[Dict[str, float]] = field(default_factory=list)
    observable_cost_history: List[float] = field(default_factory=list)
    sao_events: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    activation_history: List[np.ndarray] = field(default_factory=list)
    radius_history: List[np.ndarray] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)


class SingleAgentEngine:
    """
    Engine running the deterministic tick loop for a single-agent UFO membrane.
    Integrates membrane geometry, routing, envelopes, cost, governor and SAO.
    """

    def __init__(
        self,
        membrane: RadialMembrane | None = None,
        governor: Governor | None = None,
        boundary: BoundaryGeometry | None = None,
        cost_weights: CostWeights | None = None,
        band_config: StabilityBandConfig | None = None,
        r_max: float = 2.0,
        r_growth_rate: float = 0.3,
        r_relaxation: float = 0.2,
        seed: int = 0
    ) -> None:
        """
        Initializes the single-agent engine.
        """
        self.membrane = membrane if membrane is not None else RadialMembrane()
        self.governor = governor if governor is not None else Governor()
        self.boundary = boundary if boundary is not None else BoundaryGeometry()
        self.cost_weights = cost_weights if cost_weights is not None else CostWeights()
        self.band_config = band_config if band_config is not None else StabilityBandConfig()

        self.r_max = r_max
        self.r_growth_rate = r_growth_rate
        self.r_relaxation = r_relaxation

        self.envelope = BrimEnvelope(energy_threshold=1.5)
        self.sao_promotor = SAOPromotor(promotion_threshold=0.4)
        self.automaton = TensionAutomaton()
        self.ledger = ResidualLedger()

        # In-memory history tracking
        self.v_history: List[float] = []
        self.band_history: List[str] = []
        self.cost_history: List[RuntimeCostVector] = []
        self.observable_cost_history: List[float] = []
        self.sao_events: List[Dict[str, Any]] = []
        self.interventions: List[str] = []
        self.coherence_history: List[float] = []
        self.activation_history: List[np.ndarray] = []
        self.radius_history: List[np.ndarray] = []
        self.residual_history: List[float] = []

        # Call deterministic environment seeding
        set_deterministic_env(seed)

        # Kernel Regime Expansion Layer components
        self.regime_manager = RegimeManager()
        self.quarantine_timer: int = 0
        self.tick_count: int = 0

        # Bind Policy-Bound Semantic Memory Layer
        from radial_membrane_ai.semantic_memory.integration import bind_to_mesh
        from radial_membrane_ai.semantic_memory.core import AgentSemanticMemory
        bind_to_mesh(self)
        self.semantic_memory = AgentSemanticMemory(agent_id="single_agent")

        # Record initial state
        self._record_state()

    def _update_facet_vectors(self, task_value: float, excitation: np.ndarray) -> np.ndarray:
        """
        Performs the Pythagorean projection and updates facets for all 12 strings.
        Also returns the closure coherence Q-matrix.
        """
        Q_matrix = np.zeros((12, 12), dtype=np.float64)
        for i in range(12):
            for j in range(12):
                Q_matrix[i, j] = closure_coherence(self.membrane, self.boundary, i + 1, j + 1, samples=8)

        for idx, s in enumerate(self.membrane.strings):
            a_orig, b_orig = angular_decomposition(self.membrane, s.theta, samples=64)
            c = self.boundary.get_radius(s.theta)

            i_val = closure_ratio(a_orig, b_orig, c)
            a_proj, b_proj = project_to_admissible(a_orig, b_orig, c, metric="euclidean")

            res_a, res_b = residual_deformation(a_orig, b_orig, a_proj, b_proj)
            p_magnitude = math.sqrt(res_a**2 + res_b**2)

            current_state = s.facet.state if s.facet is not None else TensionState.RELAXED
            next_state = self.automaton.transition(current_state, i_val, p_magnitude)
            policy_priority = float(excitation[idx]) * task_value

            s.facet = FacetVector(
                facet_id=s.name,
                state=next_state,
                activation=s.activation,
                capacity=c,
                residual=p_magnitude,
                policy_priority=policy_priority
            )
            # Retain residual in the string's internal record
            s.cost = max(s.cost, p_magnitude)

        return Q_matrix

    def _record_state(self) -> None:
        """
        Records the current state snapshot into history.
        """
        self.activation_history.append(self.membrane.get_activation_vector())
        self.radius_history.append(np.array([s.radius for s in self.membrane.strings], dtype=np.float64))

    def compute_local_coherence(self) -> float:
        """
        Computes the average coherence across all active pairs on the membrane.
        """
        coherences = []
        for i in range(12):
            for j in range(12):
                if i != j:
                    coherences.append(channel_coherence(self.membrane, i + 1, j + 1, samples=16))
        return float(np.mean(coherences)) if coherences else 1.0

    def tick(
        self,
        task_value: float,
        excitation: np.ndarray,
        tool_loads: list[float] | np.ndarray | None = None,
        context_loads: list[float] | np.ndarray | None = None
    ) -> str:
        """
        Runs a single tick cycle of the single-agent engine, supporting multiple
        governed behavioral regimes.
        """
        self.tick_count += 1

        # Check Quarantine
        if self.quarantine_timer > 0:
            self.quarantine_timer -= 1
            self.band_history.append("quarantined")
            return "quarantined"

        # Ensure correct array format
        excitation = np.array(excitation, dtype=np.float64)

        # 0. Backup State for precise rollback
        m_t = self.membrane.temporal_state
        backup: Dict[str, Any] = {
            "activations": [s.activation for s in self.membrane.strings],
            "radii": [s.radius for s in self.membrane.strings],
            "deviations": dict(self.boundary.radius_deviation),
            "t_state_accumulated_tension": (
                m_t.accumulated_tension if hasattr(self.membrane, "temporal_state") else 0.0
            ),
            "t_state_consecutive_ticks": (
                m_t.consecutive_admissible_ticks if hasattr(self.membrane, "temporal_state") else 0
            ),
            "t_state_green_ticks": (
                m_t.green_ticks_count if hasattr(self.membrane, "temporal_state") else 0
            ),
            "t_state_tension_history": (
                list(m_t.tension_history) if hasattr(self.membrane, "temporal_state") else []
            ),
            "t_state_curvature_history": (
                list(m_t.curvature_history) if hasattr(self.membrane, "temporal_state") else []
            ),
            "t_state_admissibility_history": (
                list(m_t.admissibility_history) if hasattr(self.membrane, "temporal_state") else []
            ),
            "semantic_memory_store": (
                dict(self.semantic_memory.local_store)
                if hasattr(self, "semantic_memory") and self.semantic_memory else {}
            ),
            "semantic_memory_history": (
                list(self.semantic_memory.history)
                if hasattr(self, "semantic_memory") and self.semantic_memory else []
            ),
            "ledger_records": list(self.ledger.records) if hasattr(self, "ledger") else [],
            "residual_history": list(self.residual_history),
        }

        # 1. Local updates (Governor activation update & routing & boundary update)
        from radial_membrane_ai.semantic_memory.integration import on_tick_start, on_tick_end
        on_tick_start(self)

        self.governor.update_membrane(
            membrane=self.membrane,
            task_value=task_value,
            task_excitation=excitation,
            tool_loads=tool_loads,
            context_loads=context_loads
        )

        # Apply Hysteresis Inertial Damping if not disabled by active regime
        active_regime = self.regime_manager.get_regime_for_agent("single_agent")
        temp_rule_res = active_regime.temporal_rule(None, None, self)
        disable_hyst = temp_rule_res.get("disable_hysteresis", False)

        if not disable_hyst and len(self.activation_history) > 0:
            prior_act = self.activation_history[-1]
            t_state = getattr(self.membrane, "temporal_state", None)
            alpha = 0.3 * t_state.get_normalized_tension_history() if t_state is not None else 0.0
            for idx, s in enumerate(self.membrane.strings):
                s.activation = (1.0 - alpha) * s.activation + alpha * float(prior_act[idx])

        # V-Channel Routing (Radius Propagation)
        old_radii = [s.radius for s in self.membrane.strings]
        for t_idx in range(12):
            target = self.membrane.strings[t_idx]
            r_internal = target.activation * self.r_max * self.r_growth_rate

            r_propagated = 0.0
            for s_idx in range(12):
                if s_idx != t_idx:
                    source = self.membrane.strings[s_idx]
                    orig_radius = source.radius
                    source.radius = old_radii[s_idx]

                    r_prop = update_radius_along_channel(
                        source=source,
                        target=target,
                        r_max=self.r_max
                    )
                    source.radius = orig_radius
                    if r_prop > r_propagated:
                        r_propagated = r_prop

            target_radius_target = max(r_internal, r_propagated)
            target.radius = (1.0 - self.r_relaxation) * target.radius + self.r_relaxation * target_radius_target

        # 2. Update Temporal Membrane State (First part of temporal tracking)
        t_state = getattr(self.membrane, "temporal_state", None)
        if t_state is not None:
            max_curv = float(max([self.boundary.curvature(s.theta) for s in self.membrane.strings]))
            has_tens = any(s.tension > 0 for s in self.membrane.strings)
            max_tens = float(max([s.tension for s in self.membrane.strings])) if has_tens else 0.0

            # Compute max closure ratio
            closure_ratios = []
            for s in self.membrane.strings:
                a_theta, b_theta = angular_decomposition(self.membrane, s.theta, samples=32)
                from radial_membrane_ai.admissibility import dynamic_capacity_boundary_temporal
                c_theta = dynamic_capacity_boundary_temporal(self.boundary, s.theta, self.membrane)
                closure_ratios.append(closure_ratio(a_theta, b_theta, c_theta))
            max_cl = float(max(closure_ratios)) if closure_ratios else 0.0

            t_state.update_tick_history(max_curv, max_tens, max_cl)

        # 3. Regime Evaluation (Evaluating switched triggers based on latest temporal metrics)
        t_bar_eval = t_state.accumulated_tension if t_state is not None else 0.0
        coherence_eval = self.compute_local_coherence()
        current_band_eval = self.band_history[-1] if self.band_history else "green"
        max_curv_eval = float(max([self.boundary.curvature(s.theta) for s in self.membrane.strings])) \
            if t_state is not None else 0.0

        active_regime = self.regime_manager.evaluate_switching_triggers(
            entity="single_agent",
            t_bar=t_bar_eval,
            coherence=coherence_eval,
            stability_band=current_band_eval,
            curvature=max_curv_eval
        )
        temp_rule_res = active_regime.temporal_rule(None, None, self)

        # If Multi-Phase, update phase tick
        if active_regime.regime_type == KernelRegimeType.MULTI_PHASE:
            self.regime_manager.tick_multiphase_state("single_agent", t_bar_eval)

        # 4. Regime-Specific Physics Application
        # Apply stochastic activation noise if provided
        act_noise_func = temp_rule_res.get("activation_noise", None)
        if act_noise_func is not None:
            for s in self.membrane.strings:
                s.activation = max(0.0, min(1.0, s.activation + act_noise_func()))

        # Evaluate capacity scaling
        capacity_scale = active_regime.capacity_rule(None, None, self)
        if capacity_scale is None:
            capacity_scale = 1.0

        # Scale boundary scale factor
        self.boundary._custom_radius_scale = capacity_scale

        try:
            # Check closure ratio rules
            cl_res = active_regime.closure_ratio_rule(None, None, self)
            if cl_res.get("violation", False):
                # Clamp activations and trigger soft rollback
                if cl_res.get("clamped_activations") is not None:
                    for idx, s in enumerate(self.membrane.strings):
                        s.activation = cl_res["clamped_activations"][idx]
                self.interventions.append(
                    "Deterministic Admissibility Violation: Clamped activations, soft rollback."
                )
                # Restore previous activations from backup
                for idx, s in enumerate(self.membrane.strings):
                    s.activation = backup["activations"][idx]

            # Pythagorean Projection Layer & Facets
            Q_matrix = self._update_facet_vectors(task_value, excitation)

            # Boundary Deformation
            self.boundary.update_boundary(
                membrane=self.membrane,
                task_value=task_value
            )

            # 5. Collective Admissibility (None for Single-Agent)

            # 6. Coherence Detection
            self.coherence_history.append(self.compute_local_coherence())

            # 7. SAO Promotion Gate with regime rules
            # Map target range based on allowed SAO rules
            sao_rule_res = active_regime.sao_rule(None, None, self)
            allowed_ranges = sao_rule_res.get("allowed_ranges", {"short", "mid", "long"})
            restrict_to_cluster = sao_rule_res.get("restrict_to_cluster", False)
            only_at_phase_boundary = sao_rule_res.get("only_at_phase_boundary", False)

            # Let's promote to Holistic Governor (short range)
            target_layer = "Holistic Governor"
            target_range = "short"
            sao_blocked = (target_range not in allowed_ranges) or restrict_to_cluster
            if only_at_phase_boundary and not sao_rule_res.get("is_boundary", False):
                sao_blocked = True

            if sao_blocked:
                sao_verdict = "block"
                p_sao = 0.5
                sao_meta = {"blocked_by_regime": True}
            else:
                sao_verdict, p_sao, sao_meta = self.sao_promotor.promote(self.membrane, self.boundary, target_layer)

            sao_record = {
                "verdict": sao_verdict,
                "p_sao": p_sao,
                "meta": sao_meta
            }
            self.sao_events.append(sao_record)

            if sao_verdict in ("block", "constrain"):
                self.ledger.log_failure(
                    record_id=f"single_agent_sao_{len(self.ledger.records)}",
                    error_type="sao_promotion_restriction",
                    shard_id="single_agent",
                    severity="high" if sao_verdict == "block" else "medium",
                    details={"p_sao": p_sao, "verdict": sao_verdict}
                )
                self.residual_history.append(p_sao)
            else:
                self.residual_history.append(0.0)

            # 8. Correctness Checks & Stability Rules Evaluator
            # Lyapunov Energy & Band Analysis
            energy = self.governor.compute_lyapunov_energy(self.membrane)
            self.v_history.append(energy)

            # Stability thresholds multiplier
            stab_res_init = active_regime.stability_rule(None, None, self)
            thresh_mult = stab_res_init.get("threshold_multiplier", 1.0)
            effective_v_green = self.band_config.v_green * thresh_mult
            effective_v_red = self.band_config.v_red * thresh_mult

            # Determine stability band with effective thresholds
            if energy <= effective_v_green:
                band = "green"
            elif energy <= effective_v_red:
                band = "yellow"
            else:
                band = "red"
            self.band_history.append(band)

            # Evaluate stability rule validity based on newly determined band
            stab_res = active_regime.stability_rule(None, None, self)

            # Governor Interventions
            if band == "green":
                self.interventions.append("Green Band (Nominal): No intervention required.")
            elif band == "yellow":
                self.interventions.append(
                    "Yellow Band (Soft Intervention): Damping high-cost strings and tightening envelope."
                )
                for s in self.membrane.strings:
                    if s.cost > task_value:
                        s.activation *= 0.85
                        s.radius *= 0.95
            else:  # "red"
                self.interventions.append(
                    "Red Band (Hard Intervention): Throttling activations, "
                    "aggressive suppression of non-essentials."
                )
                for s in self.membrane.strings:
                    s.activation *= 0.5
                    s.radius *= 0.8

            # Brim Compute Envelope Evaluation
            brim_verdict, brim_meta = self.envelope.evaluate_envelope(self.membrane, self.boundary)
            if brim_verdict == "block":
                self.interventions.append("Brim Envelope Block: Strong fallback containment.")
                for s in self.membrane.strings:
                    s.activation *= 0.5
            elif brim_verdict == "constrain":
                self.interventions.append("Brim Envelope Constrain: Soft activation restriction.")
                for s in self.membrane.strings:
                    s.activation *= 0.8

            # 9. Rollback if Stability Rule Violations occur
            if not stab_res.get("valid", True):
                if stab_res.get("rollback", False):
                    # Complete rollback
                    for idx, s in enumerate(self.membrane.strings):
                        s.activation = backup["activations"][idx]
                        s.radius = backup["radii"][idx]
                    self.boundary.radius_deviation = dict(backup["deviations"])
                    if hasattr(self.membrane, "temporal_state"):
                        self.membrane.temporal_state.accumulated_tension = (
                            backup["t_state_accumulated_tension"]  # type: ignore
                        )
                        self.membrane.temporal_state.consecutive_admissible_ticks = (
                            backup["t_state_consecutive_ticks"]  # type: ignore
                        )
                        self.membrane.temporal_state.green_ticks_count = (
                            backup["t_state_green_ticks"]  # type: ignore
                        )
                        self.membrane.temporal_state.tension_history.clear()
                        self.membrane.temporal_state.tension_history.extend(
                            backup["t_state_tension_history"]  # type: ignore
                        )
                        self.membrane.temporal_state.curvature_history.clear()
                        self.membrane.temporal_state.curvature_history.extend(
                            backup["t_state_curvature_history"]  # type: ignore
                        )
                        self.membrane.temporal_state.admissibility_history.clear()
                        self.membrane.temporal_state.admissibility_history.extend(
                            backup["t_state_admissibility_history"]  # type: ignore
                        )
                    if hasattr(self, "semantic_memory") and self.semantic_memory:
                        self.semantic_memory.local_store = (
                            dict(backup["semantic_memory_store"])  # type: ignore
                        )
                        self.semantic_memory.history = (
                            list(backup["semantic_memory_history"])  # type: ignore
                        )
                    if hasattr(self, "ledger"):
                        self.ledger.records = list(backup["ledger_records"])  # type: ignore
                    self.residual_history = list(backup["residual_history"])  # type: ignore

                    self.interventions.append(
                        f"Regime Stability Violation: Complete Rollback triggered for band '{band}'."
                    )

                if stab_res.get("quarantine", False):
                    self.quarantine_timer = 5
                    self.interventions.append("Agent placed under Quarantine for 5 ticks due to stability violation.")

            # Update temporal state for future ticks (tension accumulation, decay, etc.)
            if t_state is not None:
                # Tension accumulation
                v_load = float(sum(s.radius for s in self.membrane.strings))
                sao_intensity = p_sao
                mem_writes = float(len(self.semantic_memory.history)) if hasattr(self, "semantic_memory") else 0.0

                # Determine active regime tension multiplier or noise
                tension_mult = temp_rule_res.get("tension_accumulation_multiplier", 1.0)
                tension_noise_val = active_regime.tension_rule(None, None, self)

                # Temporarily calculate raw cost for taxonomy
                total_act = sum(s.activation for s in self.membrane.strings)
                total_rad = sum(s.radius for s in self.membrane.strings)
                avg_q_coh = float(np.mean(Q_matrix)) if Q_matrix.size > 0 else 0.5

                raw_cost = RuntimeCostVector(
                    tokens=total_act * 35.0,
                    depth=total_rad * 2.5,
                    context=float(sum(context_loads)) if context_loads is not None else 100.0,
                    retrievals=float(sum(tool_loads)) * 1.5 if tool_loads is not None else 1.0,
                    tool_calls=float(sum(tool_loads)) if tool_loads is not None else 0.0,
                    latency=total_rad * 0.35 + total_act * 0.1,
                    corrections=(1.0 - avg_q_coh) * 7.5,
                    recovery=10.0 if band == "red" else (4.0 if band == "yellow" else 0.0)
                )

                reduced_cost = reduce_avoidable_cost(raw_cost, avg_q_coh)
                obs_cost = reduced_cost.weighted_cost(self.cost_weights.to_dict(), quality_signal=avg_q_coh)

                # Apply tension logic
                t_state.update_tension_accumulation(
                    v_channel_load=v_load,
                    cost_taxonomy_contrib=obs_cost,
                    sao_promotions_intensity=sao_intensity,
                    mem_writes_norm=min(1.0, mem_writes / 10.0)
                )

                if tension_mult != 1.0:
                    t_state.accumulated_tension *= tension_mult

                if tension_noise_val != 0.0:
                    t_state.accumulated_tension = max(0.0, t_state.accumulated_tension + tension_noise_val)

                # Decay and drift
                low_load = (v_load < 2.0)
                t_state.decay_tension(low_load=low_load)

                is_green = (band == "green")
                recovery_mu = t_state.update_recovery_dynamics(is_green=is_green, mu=0.1)
                t_state.apply_curvature_drift(self.boundary, mu=recovery_mu)

                # Integrate accumulated tension back into Lyapunov stability bands via Effective Energy correction
                if len(t_state.tension_history) >= 5:
                    eff_energy = energy + 0.5 * t_state.accumulated_tension
                    self.v_history[-1] = eff_energy

                    # Re-evaluate stability band
                    if eff_energy <= effective_v_green:
                        updated_band = "green"
                    elif eff_energy <= effective_v_red:
                        updated_band = "yellow"
                    else:
                        updated_band = "red"

                    self.band_history[-1] = updated_band
                    band = updated_band

            # Finalize Cost & Coh tracking
            total_act = sum(s.activation for s in self.membrane.strings)
            total_rad = sum(s.radius for s in self.membrane.strings)
            avg_q_coh = float(np.mean(Q_matrix)) if Q_matrix.size > 0 else 0.5
            if band == "red":
                avg_q_coh *= 0.5

            raw_cost = RuntimeCostVector(
                tokens=total_act * 35.0,
                depth=total_rad * 2.5,
                context=float(sum(context_loads)) if context_loads is not None else 100.0,
                retrievals=float(sum(tool_loads)) * 1.5 if tool_loads is not None else 1.0,
                tool_calls=float(sum(tool_loads)) if tool_loads is not None else 0.0,
                latency=total_rad * 0.35 + total_act * 0.1,
                corrections=(1.0 - avg_q_coh) * 7.5,
                recovery=10.0 if band == "red" else (4.0 if band == "yellow" else 0.0)
            )

            reduced_cost = reduce_avoidable_cost(raw_cost, avg_q_coh)
            self.cost_history.append(reduced_cost)

            obs_cost = reduced_cost.weighted_cost(self.cost_weights.to_dict(), quality_signal=avg_q_coh)
            self.observable_cost_history.append(obs_cost)

            # Post-tick invariant assertions and near-violation warning logs
            hard_energy_limit = 15.0
            hard_tension_limit = 10.0
            hard_stiffness_limit = 2.0
            hard_curvature_limit = 50.0
            hard_deviation_limit = 5.0

            # Check Lyapunov energy
            latest_energy = self.v_history[-1] if self.v_history else 0.0
            if latest_energy > hard_energy_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Lyapunov energy ({latest_energy:.4f}) "
                    f"exceeded hard limit ({hard_energy_limit})."
                )
            elif latest_energy >= 0.95 * hard_energy_limit:
                self.interventions.append(
                    f"Near-violation warning: Lyapunov energy ({latest_energy:.4f}) "
                    f"is within 5% of hard limit ({hard_energy_limit})."
                )

            # Check Tension
            latest_tension = t_state.accumulated_tension if t_state is not None else 0.0
            if latest_tension > hard_tension_limit:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Temporal tension ({latest_tension:.4f}) "
                    f"exceeded hard limit ({hard_tension_limit})."
                )
            elif latest_tension >= 0.95 * hard_tension_limit:
                self.interventions.append(
                    f"Near-violation warning: Temporal tension ({latest_tension:.4f}) "
                    f"is within 5% of hard limit ({hard_tension_limit})."
                )

            # Check Stiffness
            max_stiffness = (
                float(max(s.stiffness for s in self.membrane.strings))
                if self.membrane.strings else 0.0
            )
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

            # Check Channel Coherence
            min_coherence = self.compute_local_coherence()
            if min_coherence < 0.0:
                from radial_membrane_ai.exceptions import GovernanceError
                raise GovernanceError(
                    f"Governed bound violated: Channel coherence ({min_coherence:.4f}) "
                    f"fell below lower limit (0.0)."
                )

            # Check Boundary Curvature
            max_curvature = (
                float(max(abs(self.boundary.curvature(s.theta)) for s in self.membrane.strings))
                if self.membrane.strings else 0.0
            )
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

            # Check Radius Deviation
            max_dev = (
                float(max(abs(dev) for dev in self.boundary.radius_deviation.values()))
                if self.boundary.radius_deviation else 0.0
            )
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

            self._record_state()

        finally:
            # Restore boundary scale factor
            self.boundary._custom_radius_scale = 1.0

        on_tick_end(self)
        return band

    def run(
        self,
        n_steps: int,
        task_value_sequence: List[float],
        excitation_sequence: List[np.ndarray],
        tool_loads_sequence: List[List[float]] | None = None,
        context_loads_sequence: List[List[float]] | None = None
    ) -> SingleAgentRunResult:
        """
        Runs the simulation for multiple ticks.
        """
        for i in range(n_steps):
            t_val = task_value_sequence[i % len(task_value_sequence)]
            excite = excitation_sequence[i % len(excitation_sequence)]
            t_load = tool_loads_sequence[i % len(tool_loads_sequence)] if tool_loads_sequence else None
            c_load = context_loads_sequence[i % len(context_loads_sequence)] if context_loads_sequence else None

            self.tick(
                task_value=t_val,
                excitation=excite,
                tool_loads=t_load,
                context_loads=c_load
            )

        return SingleAgentRunResult(
            v_history=list(self.v_history),
            band_history=list(self.band_history),
            cost_history=[c.__dict__ for c in self.cost_history],
            observable_cost_history=list(self.observable_cost_history),
            sao_events=list(self.sao_events),
            interventions=list(self.interventions),
            coherence_history=list(self.coherence_history),
            activation_history=list(self.activation_history),
            radius_history=list(self.radius_history),
            residual_history=list(self.residual_history)
        )
