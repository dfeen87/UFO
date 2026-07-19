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
        r_relaxation: float = 0.2
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
        Runs a single tick cycle of the single-agent engine.

        Steps:
        1. Local cost calculations & activation update via Governor.
        2. V-Channel reasoning radius propagation (Depth).
        3. Pythagorean projection, tension updating, and facet resolution.
        4. Boundary geometry deformation.
        5. Lyapunov energy calculation and band determination.
        6. Soft/Hard interventions by Governor.
        7. Bounded compute envelope check (Brim).
        8. SAO Promotion gate.
        9. Cost taxonomy evaluation and quality-preserving reduction.
        10. Logging of states, residuals, and interventions.

        Returns:
            The determined stability band for the tick ("green", "yellow", or "red").
        """
        # Ensure correct array format
        excitation = np.array(excitation, dtype=np.float64)

        # Call Semantic Memory on_tick_start hook
        from radial_membrane_ai.semantic_memory.integration import on_tick_start, on_tick_end
        on_tick_start(self)

        # 1. Update activations via Governor
        self.governor.update_membrane(
            membrane=self.membrane,
            task_value=task_value,
            task_excitation=excitation,
            tool_loads=tool_loads,
            context_loads=context_loads
        )

        # 2. V-Channel Routing (Radius Propagation)
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

        # 3. Pythagorean Projection Layer & Facets
        Q_matrix = self._update_facet_vectors(task_value, excitation)

        # 4. Boundary Deformation
        self.boundary.update_boundary(
            membrane=self.membrane,
            task_value=task_value
        )

        # 5. Lyapunov Energy & Band Analysis
        energy = self.governor.compute_lyapunov_energy(self.membrane)
        self.v_history.append(energy)

        # Determine stability band
        if energy <= self.band_config.v_green:
            band = "green"
        elif energy <= self.band_config.v_red:
            band = "yellow"
        else:
            band = "red"
        self.band_history.append(band)

        # 6. Governor Interventions
        if band == "green":
            self.interventions.append("Green Band (Nominal): No intervention required.")
        elif band == "yellow":
            self.interventions.append(
                "Yellow Band (Soft Intervention): Damping high-cost strings and tightening envelope."
            )
            # Damp activations on high-cost strings (> task_value)
            for s in self.membrane.strings:
                if s.cost > task_value:
                    s.activation *= 0.85
                    s.radius *= 0.95
        else:  # "red"
            self.interventions.append(
                "Red Band (Hard Intervention): Throttling activations, "
                "aggressive suppression of non-essentials."
            )
            # Aggressive suppression of all strings, especially non-analytical/non-contextual ones
            for s in self.membrane.strings:
                s.activation *= 0.5
                s.radius *= 0.8

        # 7. Brim Compute Envelope Evaluation
        brim_verdict, brim_meta = self.envelope.evaluate_envelope(self.membrane, self.boundary)
        if brim_verdict == "block":
            self.interventions.append("Brim Envelope Block: Strong fallback containment.")
            for s in self.membrane.strings:
                s.activation *= 0.5
        elif brim_verdict == "constrain":
            self.interventions.append("Brim Envelope Constrain: Soft activation restriction.")
            for s in self.membrane.strings:
                s.activation *= 0.8

        # 8. SAO Promotion Gate
        sao_verdict, p_sao, sao_meta = self.sao_promotor.promote(self.membrane, self.boundary, "Holistic Governor")
        sao_record = {
            "verdict": sao_verdict,
            "p_sao": p_sao,
            "meta": sao_meta
        }
        self.sao_events.append(sao_record)

        if sao_verdict in ("block", "constrain"):
            # Record promotion residual failure in our ledger
            self.ledger.log_failure(
                record_id=f"single_agent_sao_{len(self.ledger.records)}",
                error_type="sao_promotion_restriction",
                shard_id="single_agent",
                severity="high" if sao_verdict == "block" else "medium",
                details={"p_sao": p_sao, "verdict": sao_verdict, "energy": energy}
            )
            # Append residual to history
            self.residual_history.append(p_sao)
        else:
            self.residual_history.append(0.0)

        # 9. Cost Taxonomy Integration
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

        # 10. Coh & State tracking
        self.coherence_history.append(self.compute_local_coherence())
        self._record_state()

        # Call Semantic Memory on_tick_end hook
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
