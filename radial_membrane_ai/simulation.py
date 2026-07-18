"""
Project Rainbow-Style Toy Simulator module.

This module implements the toy simulation environment described in Section 6
of Feeney (2025). It coordinates the RadialMembrane, Governor, BoundaryGeometry,
and RuntimeCostVector to run dynamic multi-step simulations.
"""

from __future__ import annotations
import math
import numpy as np

from radial_membrane_ai.membrane import RadialMembrane, BehavioralString
from radial_membrane_ai.channels import update_radius_along_channel, channel_coherence
from radial_membrane_ai.governor import Governor, GovernorConfig
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.cost import RuntimeCostVector, reduce_avoidable_cost


class RainbowSimulation:
    """
    Project Rainbow-Style Toy Simulator.

    Ref: Section 6 & 8 of Feeney (2025).
    Simulates the interaction between task demand (excitation), radial activation,
    V-channel routing (reasoning depth), governor regulation, boundary deformation,
    and runtime costs over discrete time steps.
    """

    def __init__(
        self,
        membrane: RadialMembrane | None = None,
        governor: Governor | None = None,
        boundary: BoundaryGeometry | None = None,
        r_max: float = 2.0,
        r_growth_rate: float = 0.3,
        r_relaxation: float = 0.2
    ) -> None:
        """
        Initializes the simulator.

        Args:
            membrane: An optional custom RadialMembrane.
            governor: An optional custom Governor.
            boundary: An optional custom BoundaryGeometry.
            r_max: Maximum reasoning radius allowed during V-channel routing.
            r_growth_rate: Speed of internal radius growth under activation.
            r_relaxation: Temporal relaxation factor for radius updates (smooth dynamics).
        """
        self.membrane = membrane if membrane is not None else RadialMembrane()
        self.governor = governor if governor is not None else Governor()
        self.boundary = boundary if boundary is not None else BoundaryGeometry()

        self.r_max = r_max
        self.r_growth_rate = r_growth_rate
        self.r_relaxation = r_relaxation

        # History tracking
        self.activation_history: list[np.ndarray] = []
        self.radius_history: list[np.ndarray] = []
        self.boundary_snapshots: list[np.ndarray] = []  # List of radius sample arrays
        self.cost_history: list[RuntimeCostVector] = []
        self.observable_cost_history: list[float] = []

        # Standard Cost Weights for scalar projection
        self.cost_weights = {
            "tokens": 0.05,
            "depth": 0.4,
            "context": 0.01,
            "retrievals": 0.2,
            "tool_calls": 0.3,
            "latency": 0.5,
            "corrections": 0.6,
            "recovery": 0.8
        }

        # Define Task Signatures
        # Each task type has an excitation vector (12 floats), a task value, and associated workloads.
        # String Order:
        # Analytical (1-4): depth, precision, technical_detail, structural_rigor
        # Contextual (5-6): context_sensitivity, transparency
        # Generative (7-9): initiative, exploration, creativity
        # Interpersonal (10-12): tone, emotional_warmth, conciseness
        from typing import Any
        self.task_signatures: dict[str, dict[str, Any]] = {
            "technical_deep_analysis": {
                "excitation": [1.0, 1.0, 1.0, 1.0,  0.1, 0.1,  0.1, 0.1, 0.1,  0.1, 0.1, 0.1],
                "task_value": 0.9,
                "tool_loads": [0.3, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "context_loads": [150.0, 50.0, 300.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "supportive_concise_reply": {
                "excitation": [0.1, 0.1, 0.1, 0.1,  0.1, 0.1,  0.1, 0.1, 0.1,  1.0, 1.0, 1.0],
                "task_value": 0.6,
                "tool_loads": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                "context_loads": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 10.0, 10.0]
            },
            "planning": {
                "excitation": [0.1, 0.1, 0.1, 0.1,  1.0, 1.0,  1.0, 1.0, 1.0,  0.1, 0.1, 0.1],
                "task_value": 0.8,
                "tool_loads": [0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.5, 0.6, 0.4, 0.0, 0.0, 0.0],
                "context_loads": [0.0, 0.0, 0.0, 0.0, 250.0, 80.0, 120.0, 150.0, 100.0, 0.0, 0.0, 0.0]
            }
        }

        # Save initial state
        self._record_state(samples=64)

    def _record_state(self, samples: int = 64) -> None:
        """
        Saves current state into history tracking.
        """
        self.activation_history.append(self.membrane.get_activation_vector())
        self.radius_history.append(np.array([s.radius for s in self.membrane.strings], dtype=np.float64))

        # Sample boundary geometry radius
        boundary_angles = np.linspace(0, 2 * math.pi, samples, endpoint=False)
        snap = np.array([self.boundary.get_radius(theta) for theta in boundary_angles], dtype=np.float64)
        self.boundary_snapshots.append(snap)

    def run_step(self, task_type: str) -> None:
        """
        Applies a single simulation cycle.

        Steps:
        1. Parse task signature.
        2. Apply governor update to membrane activations.
        3. Compute V-channel routing and update reasoning radii.
        4. Apply boundary deformation.
        5. Evaluate runtime cost and perform quality-preserving cost reduction.
        6. Append updated state to history.

        Args:
            task_type: Key in self.task_signatures.
        """
        if task_type not in self.task_signatures:
            raise ValueError(f"Unknown task type '{task_type}'. Available: {list(self.task_signatures.keys())}")

        sig = self.task_signatures[task_type]
        excitation = sig["excitation"]
        task_value = sig["task_value"]
        tool_loads = sig["tool_loads"]
        context_loads = sig["context_loads"]

        # 1. Update activations via Governor (this also updates string.cost inside the membrane)
        self.governor.update_membrane(
            membrane=self.membrane,
            task_value=task_value,
            task_excitation=excitation,
            tool_loads=tool_loads,
            context_loads=context_loads
        )

        # 2. Update reasoning radii along V-channels
        # Create a copy of current radii to prevent order-of-evaluation bias during propagation
        old_radii = [s.radius for s in self.membrane.strings]

        for t_idx in range(12):
            target = self.membrane.strings[t_idx]

            # Internal growth from active attention
            r_internal = target.activation * self.r_max * self.r_growth_rate

            # Propagated growth along V-channels
            r_propagated = 0.0
            for s_idx in range(12):
                if s_idx != t_idx:
                    source = self.membrane.strings[s_idx]
                    # Temporarily set source's radius to old value to propagate cleanly
                    orig_radius = source.radius
                    source.radius = old_radii[s_idx]

                    # Compute propagated radius along channel
                    r_prop = update_radius_along_channel(
                        source=source,
                        target=target,
                        r_max=self.r_max
                    )
                    # Restore source radius
                    source.radius = orig_radius

                    if r_prop > r_propagated:
                        r_propagated = r_prop

            # Target radius target is maximum of internal attention and V-channel propagation
            target_radius_target = max(r_internal, r_propagated)

            # Temporal relaxation (smooth update)
            target.radius = (1.0 - self.r_relaxation) * target.radius + self.r_relaxation * target_radius_target

        # 3. Update Boundary Geometry
        self.boundary.update_boundary(
            membrane=self.membrane,
            task_value=task_value
        )

        # 4. Evaluate Cost Vector
        total_act = sum(s.activation for s in self.membrane.strings)
        total_rad = sum(s.radius for s in self.membrane.strings)

        # Compute Quality Signal as average coherence of active behavioral strings
        coherence_sum = 0.0
        active_count = 0
        for i in range(1, 13):
            s = self.membrane.strings[i - 1]
            if s.activation > 0.1:
                # Compute coherence of this string with others
                c_sum_j = 0.0
                for j in range(1, 13):
                    if i != j:
                        c_sum_j += channel_coherence(self.membrane, i, j, samples=8)
                coherence_sum += (c_sum_j / 11.0)
                active_count += 1

        quality_signal = (coherence_sum / active_count) if active_count > 0 else 0.5
        # If the governor detects instability, decay the quality signal
        if not self.governor.is_stable():
            quality_signal *= 0.5

        # Construct raw cost vector
        raw_cost_vector = RuntimeCostVector(
            tokens=total_act * 40.0,
            depth=total_rad * 3.0,
            context=float(sum(context_loads)) if context_loads is not None else 100.0,
            retrievals=float(sum(tool_loads)) * 1.5 if tool_loads is not None else 1.0,
            tool_calls=float(sum(tool_loads)) if tool_loads is not None else 0.0,
            latency=total_rad * 0.4 + total_act * 0.1,
            corrections=(1.0 - quality_signal) * 8.0,
            recovery=12.0 if not self.governor.is_stable() else 0.0
        )

        # Reduce avoidable costs using quality signal
        reduced_cost_vector = reduce_avoidable_cost(raw_cost_vector, quality_signal)
        self.cost_history.append(reduced_cost_vector)

        # Project to scalar observable cost
        obs_cost = reduced_cost_vector.weighted_cost(self.cost_weights, quality_signal=quality_signal)
        self.observable_cost_history.append(obs_cost)

        # 5. Record state
        self._record_state(samples=64)

    def run(self, n_steps: int, task_sequence: list[str]) -> None:
        """
        Runs the simulation for a sequence of tasks over multiple steps.

        If len(task_sequence) < n_steps, the sequence is cycled.

        Args:
            n_steps: Number of simulation steps to run.
            task_sequence: List of task type string keys.
        """
        if not task_sequence:
            raise ValueError("Task sequence cannot be empty.")

        for step in range(n_steps):
            task_type = task_sequence[step % len(task_sequence)]
            self.run_step(task_type)

    def get_activation_history(self) -> list[np.ndarray]:
        """
        Returns history of activation vectors.
        """
        return self.activation_history

    def get_energy_history(self) -> list[float]:
        """
        Returns history of Lyapunov stability energy.
        """
        return self.governor.energy_history

    def get_boundary_snapshots(self, samples: int = 64) -> list[np.ndarray]:
        """
        Returns boundary radius snaps over time.
        """
        # Re-sample if needed, but we already tracked snapshot values of length 64
        return self.boundary_snapshots

    def render_summary(self) -> None:
        """
        Prints a text summary of the simulation results.
        """
        print("=== RAINBOW SIMULATION RUN SUMMARY ===")
        print(f"Total Steps run: {len(self.activation_history) - 1}")
        if self.observable_cost_history:
            print(f"Final Observable Cost: {self.observable_cost_history[-1]:.4f}")
            print(f"Average Observable Cost: {np.mean(self.observable_cost_history):.4f}")
        if self.governor.energy_history:
            print(f"Final Lyapunov Energy: {self.governor.energy_history[-1]:.4f}")
            print(f"Governor Stable: {self.governor.is_stable()}")

        # Quadrant Activations at final step
        print("\nFinal Quadrant Activations:")
        for q in ["analytical", "contextual", "generative", "interpersonal"]:
            print(f"  {q.capitalize()}: {self.membrane.get_quadrant_activation(q):.4f}")
        print("======================================")
