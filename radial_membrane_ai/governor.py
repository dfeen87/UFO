"""
Governor and Stability Control module.

This module implements the governor layer, stability checks,
and Lyapunov-style stability control described in Feeney (2025) & (2026).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from radial_membrane_ai.membrane import BehavioralString, RadialMembrane
from radial_membrane_ai.channels import channel_coherence
from radial_membrane_ai.admissibility import apply_lyapunov_dissipation, phase_smoothing


@dataclass
class GovernorConfig:
    """
    Configuration coefficients for the local cost function and stability control.

    Ref: Section 5.1 of Feeney (2025).

    Attributes:
        w_d: Cost weight for reasoning depth r_i^2.
        w_a: Cost weight for amplitude/activation a_i.
        w_l: Cost weight for tool/agentic load ell_i.
        w_c: Cost weight for context usage log(1 + context_i).
        w_T: Cost weight for tension tau_i.
        w_K: Cost weight for stiffness K_i.
        lyapunov_alpha: Lyapunov energy coefficient for squared radii.
        lyapunov_beta: Lyapunov energy coefficient for costs.
        lyapunov_gamma: Lyapunov energy coefficient for squared tensions.
        learning_rate: Base update rate for activation changes.
        suppression_weight: Coefficient for cost-based expansion suppression.
    """
    w_d: float = 0.2
    w_a: float = 0.1
    w_l: float = 0.3
    w_c: float = 0.15
    w_T: float = 0.1
    w_K: float = 0.05
    lyapunov_alpha: float = 0.5
    lyapunov_beta: float = 0.3
    lyapunov_gamma: float = 0.2
    learning_rate: float = 0.1
    suppression_weight: float = 0.5


def compute_local_cost(
    string: BehavioralString,
    tool_load: float,
    context_load: float,
    config: GovernorConfig
) -> float:
    """
    Computes local cost c_i for a behavioral string.

    c_i = w_d * r_i^2 + w_a * a_i + w_l * ell_i + w_c * log(1 + context_i) + w_T * tau_i + w_K * K_i

    Args:
        string: The BehavioralString.
        tool_load: Tool/agentic load ell_i.
        context_load: Context usage context_i.
        config: GovernorConfig containing weights.

    Returns:
        The computed local cost.
    """
    cost_r = config.w_d * (string.radius ** 2)
    cost_a = config.w_a * string.activation
    cost_l = config.w_l * tool_load
    cost_c = config.w_c * math.log1p(context_load)
    cost_T = config.w_T * string.tension
    cost_K = config.w_K * string.stiffness

    return cost_r + cost_a + cost_l + cost_c + cost_T + cost_K


def governed_propagation(
    source: BehavioralString,
    target: BehavioralString,
    config: GovernorConfig,
    tool_load: float = 0.0,
    context_load: float = 0.0,
    lambda_: float = 1.0,
    cost_variant: str = "exponential"
) -> float:
    """
    Computes inverse-cost governed propagation P(s -> t).

    P(s -> t) = sigmoid(a_t) * ((1 + cos(theta_s - theta_t)) / 2) * W_t

    Where W_t is computed using the local cost of the target string.

    Args:
        source: Source BehavioralString.
        target: Target BehavioralString.
        config: GovernorConfig weights to calculate local cost.
        tool_load: Target's tool load.
        context_load: Target's context load.
        lambda_: Inverse cost decay rate.
        cost_variant: Cost weight formula, either 'simple' or 'exponential'.

    Returns:
        Governed propagation strength.
    """
    # 1. Compute target cost
    c_t = compute_local_cost(target, tool_load=tool_load, context_load=context_load, config=config)

    # 2. Compute W_t
    if cost_variant == "simple":
        w_t = 1.0 / (1.0 + c_t)
    elif cost_variant == "exponential":
        w_t = math.exp(-lambda_ * c_t)
    else:
        raise ValueError("cost_variant must be 'simple' or 'exponential'")

    # 3. Compute sigmoid(a_t)
    # A standard logistic sigmoid centered at 0 with scaling or standard sigmoid of activation
    sigmoid_a_t = 1.0 / (1.0 + math.exp(-target.activation))

    # 4. Compute alignment
    alignment = (1.0 + math.cos(source.theta - target.theta)) / 2.0

    return sigmoid_a_t * alignment * w_t


class Governor:
    """
    Governor Layer and Stability Control.

    Ref: Sections 5.3 & 5.4 of Feeney (2025).
    Controls the activation update of the membrane, suppresses excessive cost,
    and computes Lyapunov-style stability energy.
    """

    def __init__(self, config: GovernorConfig | None = None) -> None:
        self.config = config if config is not None else GovernorConfig()
        self.energy_history: list[float] = []

    def compute_local_costs(
        self,
        membrane: RadialMembrane,
        tool_loads: list[float] | np.ndarray | None = None,
        context_loads: list[float] | np.ndarray | None = None
    ) -> list[float]:
        """
        Computes local costs for all 12 strings of the membrane.
        """
        costs = []
        for idx in range(12):
            t_load = float(tool_loads[idx]) if tool_loads is not None else 0.0
            c_load = float(context_loads[idx]) if context_loads is not None else 0.0
            costs.append(
                compute_local_cost(
                    membrane.strings[idx],
                    tool_load=t_load,
                    context_load=c_load,
                    config=self.config
                )
            )
        return costs

    def update_membrane(
        self,
        membrane: RadialMembrane,
        task_value: float,
        task_excitation: list[float] | np.ndarray | None = None,
        tool_loads: list[float] | np.ndarray | None = None,
        context_loads: list[float] | np.ndarray | None = None,
        coherence_samples: int = 16
    ) -> None:
        """
        Computes a governed delta vector and applies it to the membrane's activation.

        Ref: Section 5.3.
        Allows expansions justified by task value and coherence.
        Suppresses expansions that are high-cost and low-value.

        Args:
            membrane: The RadialMembrane instance to update.
            task_value: The value/importance of the current task.
            task_excitation: Optional input excitation vector of length 12.
                             Defaults to uniform excitation of 1.0.
            tool_loads: Optional tool loads for each of the 12 strings.
            context_loads: Optional context loads for each of the 12 strings.
            coherence_samples: Angular samples used to compute channel coherence.
        """
        # Ensure task_excitation is populated
        if task_excitation is None:
            excitation = np.ones(12, dtype=np.float64)
        else:
            excitation = np.array(task_excitation, dtype=np.float64)

        # 1. Update each string's cost inside the membrane before evaluating update
        local_costs = self.compute_local_costs(membrane, tool_loads, context_loads)
        for idx, cost_val in enumerate(local_costs):
            membrane.strings[idx].cost = cost_val

        # 2. Compute coherence for each string (average coherence with other strings)
        coherences = np.zeros(12, dtype=np.float64)
        for i in range(12):
            c_sum = 0.0
            for j in range(12):
                if i != j:
                    c_sum += channel_coherence(membrane, i + 1, j + 1, samples=coherence_samples)
            coherences[i] = c_sum / 11.0

        # 3. Calculate delta for each string
        delta = np.zeros(12, dtype=np.float64)
        epsilon = 1e-6

        for i in range(12):
            s = membrane.strings[i]
            c_i = s.cost
            excitation_i = float(excitation[i])

            # Expansion pressure: higher when task value is high, target excitation is high, and coherence is high
            expansion = task_value * excitation_i * (1.0 + coherences[i])

            # Suppression pressure: higher when local cost is high and task value is low
            suppression = self.config.suppression_weight * c_i / (task_value + epsilon)

            # Net rate of change
            delta_val = self.config.learning_rate * (expansion - suppression)

            # If there's high cost but low task value, we force a stronger negative delta to suppress it
            if c_i > task_value and task_value < 0.3:
                delta_val -= self.config.learning_rate * (c_i - task_value)

            delta[i] = delta_val

        # 4. Apply delta to membrane
        membrane.update_activation(delta)

        # 4.5. Phase smoothing circular filter to prevent high-frequency oscillations
        phase_smoothing(membrane, window_size=3)

        # 5. Track Lyapunov energy
        self.track_energy(membrane)

        # 5.5. Apply energy dissipation step if unstable
        if not self.is_stable():
            apply_lyapunov_dissipation(membrane, self, target_energy=None, dissipation_rate=0.08)

    def compute_lyapunov_energy(self, membrane: RadialMembrane) -> float:
        """
        Computes the Lyapunov-style stability energy E(t).

        E(t) = sum_i ( alpha * r_i^2 + beta * c_i + gamma * tau_i^2 )

        Args:
            membrane: The RadialMembrane instance.

        Returns:
            Computed energy scalar value.
        """
        total_energy = 0.0
        for s in membrane.strings:
            r_term = self.config.lyapunov_alpha * (s.radius ** 2)
            c_term = self.config.lyapunov_beta * s.cost
            tau_term = self.config.lyapunov_gamma * (s.tension ** 2)
            total_energy += r_term + c_term + tau_term
        return total_energy

    def track_energy(self, membrane: RadialMembrane) -> float:
        """
        Computes current Lyapunov energy and appends it to history.
        """
        energy = self.compute_lyapunov_energy(membrane)
        self.energy_history.append(energy)
        return energy

    def is_stable(self, window: int = 5, tolerance: float = 1e-4) -> bool:
        """
        Returns True if Lyapunov energy is trending downward or bounded.

        Stability is determined if:
        - Energy is strictly non-increasing over the last `window` steps (on average),
        - Or energy variance is below `tolerance` (indicating bounded/stabilized state).

        Args:
            window: Number of recent history steps to evaluate.
            tolerance: Variance threshold for a stabilized/bounded signal.
        """
        if len(self.energy_history) < 2:
            return True

        recent = self.energy_history[-window:]
        if len(recent) < 2:
            return True

        # Check bounded (variance is extremely small)
        variance = float(np.var(recent))
        if variance < tolerance:
            return True

        # Check trend: average difference between consecutive elements is <= 0
        diffs = np.diff(recent)
        avg_diff = float(np.mean(diffs))

        return avg_diff <= tolerance
