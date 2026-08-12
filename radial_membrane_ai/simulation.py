# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Project Rainbow-Style Toy Simulator module.

This module implements the toy simulation environment described in Section 6
of Feeney (2025). It coordinates the RadialMembrane, Governor, BoundaryGeometry,
and RuntimeCostVector to run dynamic multi-step simulations, fully integrated
with the Pythagorean Projection layer (Feeney, 2026).
"""

from __future__ import annotations
import math
import numpy as np

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.channels import update_radius_along_channel
from radial_membrane_ai.governor import Governor
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.cost import RuntimeCostVector, reduce_avoidable_cost

# Pythagorean layer imports
from radial_membrane_ai.projection import closure_ratio, project_to_admissible, residual_deformation
from radial_membrane_ai.admissibility import angular_decomposition, KernelEvolution
from radial_membrane_ai.coherence import closure_coherence
from radial_membrane_ai.facet import FacetVector, TensionAutomaton, TensionState
from radial_membrane_ai.holistic import compute_holistic_field, HolisticGovernorField
from radial_membrane_ai.envelope import BrimEnvelope
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.mesh import FederatedShardMesh, FederatedShard, ShardState
from radial_membrane_ai.utils import set_deterministic_env
from radial_membrane_ai.exceptions import GovernanceError
import sys
import time
import os


class RainbowSimulation:
    """
    Project Rainbow-Style Toy Simulator with Pythagorean Projection.

    Ref: Section 6 & 8 of Feeney (2026).
    Simulates the interaction between task demand (excitation), radial activation,
    V-channel routing (reasoning depth), governor regulation, boundary deformation,
    and runtime costs over discrete time steps, unified around the Pythagorean invariant.
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

        # Kernel evolution tracker
        self.kernel_evo = KernelEvolution()

        # Tension state automaton
        self.automaton = TensionAutomaton()

        # Layer Stack integrations
        self.brim_envelope = BrimEnvelope(energy_threshold=1.5)
        self.sao_promotor = SAOPromotor(promotion_threshold=0.4)
        self.shard_mesh = FederatedShardMesh()
        self.holistic_gov = HolisticGovernorField()

        # Populate a default set of federated shards to simulate mesh operations
        for i in range(3):
            self.shard_mesh.add_shard(
                FederatedShard(
                    shard_id=f"shard_node_{i}",
                    state=ShardState.IDLE,
                    capacity=1.5,
                    trust_score=0.9,
                    cost_factor=1.0,
                    latency=15.0
                )
            )

        # History tracking
        self.activation_history: list[np.ndarray] = []
        self.radius_history: list[np.ndarray] = []
        self.boundary_snapshots: list[np.ndarray] = []  # List of radius sample arrays
        self.cost_history: list[RuntimeCostVector] = []
        self.observable_cost_history: list[float] = []
        self.holistic_history: list[float] = []
        self.mesh_coherence_history: list[float] = []
        self.brim_verdicts: list[str] = []
        self.sao_verdicts: list[str] = []

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

        # Save initial state and initialize facets
        self._update_pythagorean_facets(task_value=0.5, excitation=np.ones(12, dtype=np.float64))
        self._record_state(samples=64)

    def _update_pythagorean_facets(
        self,
        task_value: float,
        excitation: np.ndarray
    ) -> tuple[list[FacetVector], np.ndarray]:
        """
        Performs the complete Pythagorean projection pipeline:
        1. Decomposes membrane state into orthogonal legs (a, b).
        2. Computes the closure ratio i(t, theta).
        3. Projects (a, b) to admissible boundary.
        4. Computes the residual deformation p(t, theta).
        5. Updates tension states and constructs FacetVectors for all 12 strings.
        6. Updates self.membrane strings.
        7. Computes and returns the coherence matrix Q_matrix.
        """
        facets = []

        # We will also compute the 12x12 Q_matrix of shared closure boundaries
        Q_matrix = np.zeros((12, 12), dtype=np.float64)
        for i in range(12):
            for j in range(12):
                Q_matrix[i, j] = closure_coherence(self.membrane, self.boundary, i + 1, j + 1, samples=8)

        # Update each string with its new facet state
        for idx, s in enumerate(self.membrane.strings):
            # Extract orthogonal legs
            a_orig, b_orig = angular_decomposition(self.membrane, s.theta, samples=64)
            c = self.boundary.get_radius(s.theta)

            # Local closure ratio
            i_val = closure_ratio(a_orig, b_orig, c)

            # Project to safety boundary
            a_proj, b_proj = project_to_admissible(a_orig, b_orig, c, metric="euclidean")

            # Residual vector and scalar magnitude
            res_a, res_b = residual_deformation(a_orig, b_orig, a_proj, b_proj)
            p_magnitude = math.sqrt(res_a**2 + res_b**2)

            # Determine tension state via state machine
            current_state = s.facet.state if s.facet is not None else TensionState.RELAXED
            next_state = self.automaton.transition(current_state, i_val, p_magnitude)

            # Policy priority maps directly to task excitation / value
            policy_priority = float(excitation[idx]) * task_value

            facet_vec = FacetVector(
                facet_id=s.name,
                state=next_state,
                activation=s.activation,
                capacity=c,
                residual=p_magnitude,
                policy_priority=policy_priority
            )

            s.facet = facet_vec
            facets.append(facet_vec)

        return facets, Q_matrix

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
        Applies a single simulation cycle, integrating the Pythagorean Projection layers.

        Steps:
        1. Parse task signature.
        2. Evolve kernels and update membrane activations via Governor.
        3. Perform Pythagorean decomposition, projection, and facet state-transitions.
        4. Compute V-channel routing (using state-aware routing weights) and update reasoning radii.
        5. Apply boundary deformation (closure-ratio aware).
        6. Compute holistic field H_hol(t).
        7. Evaluate runtime cost and perform quality-preserving cost reduction.
        8. Append updated state to history.

        Args:
            task_type: Key in self.task_signatures.
        """
        if task_type not in self.task_signatures:
            raise ValueError(f"Unknown task type '{task_type}'. Available: {list(self.task_signatures.keys())}")

        sig = self.task_signatures[task_type]
        excitation = np.array(sig["excitation"], dtype=np.float64)
        task_value = sig["task_value"]
        tool_loads = sig["tool_loads"]
        context_loads = sig["context_loads"]

        # Step A: Kernel Leg Evolution
        total_excite = float(np.sum(excitation))
        v_activity = sum(s.activation * s.radius for s in self.membrane.strings)
        self.kernel_evo.step(input_excitation=total_excite, v_channel_activity=v_activity)

        # 1. Update activations via Governor (this also updates string.cost inside the membrane)
        self.governor.update_membrane(
            membrane=self.membrane,
            task_value=task_value,
            task_excitation=excitation,
            tool_loads=tool_loads,
            context_loads=context_loads
        )

        # 1.5. Execute Pythagorean projection layer to update facet states before channel routing
        facets, Q_matrix = self._update_pythagorean_facets(task_value, excitation)

        # 2. Update reasoning radii along V-channels
        # Create a copy of current radii to prevent order-of-evaluation bias during propagation
        old_radii = [s.radius for s in self.membrane.strings]

        for t_idx in range(12):
            target = self.membrane.strings[t_idx]

            # Internal growth from active attention, scaled by the compute-aware effective kernel
            cost_press = target.cost
            k_eff = self.kernel_evo.get_effective_kernel(target.theta, cost_press)
            r_internal = target.activation * self.r_max * self.r_growth_rate * k_eff

            # Propagated growth along V-channels
            r_propagated = 0.0
            for s_idx in range(12):
                if s_idx != t_idx:
                    source = self.membrane.strings[s_idx]
                    # Temporarily set source's radius to old value to propagate cleanly
                    orig_radius = source.radius
                    source.radius = old_radii[s_idx]

                    # Compute propagated radius along channel
                    # (using updated state-aware routing weights inside channels.py)
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

        # 3. Update Boundary Geometry (incorporating closure-ratio awareness)
        self.boundary.update_boundary(
            membrane=self.membrane,
            task_value=task_value
        )

        # 3.5. Re-evaluate Pythagorean facets post-deformation and compute holistic field
        facets, Q_matrix = self._update_pythagorean_facets(task_value, excitation)
        h_hol = compute_holistic_field(self.membrane, self.boundary, facets, Q_matrix)
        self.holistic_history.append(h_hol)

        # Step B: Brim Compute Envelope evaluation (Multi-layer Fallback & Admission)
        brim_verdict, brim_meta = self.brim_envelope.evaluate_envelope(self.membrane, self.boundary)
        self.brim_verdicts.append(brim_verdict)

        if brim_verdict == "block":
            # Force activation containment / fallback recovery
            for s in self.membrane.strings:
                s.activation *= 0.5
        elif brim_verdict == "constrain":
            for s in self.membrane.strings:
                s.activation *= 0.8
        elif brim_verdict == "re-project":
            # Trigger re-projection recovery
            facets, Q_matrix = self._update_pythagorean_facets(task_value, excitation)

        # Step C: SAO Promotion Step
        sao_verdict, p_sao, sao_meta = self.sao_promotor.promote(self.membrane, self.boundary, "Holistic Governor")
        self.sao_verdicts.append(sao_verdict)

        # Step D: Federated Mesh Workload Routing & Governance execution
        workload = {
            "capacity": float(np.mean([s.radius for s in self.membrane.strings])),
            "privacy": 0.5,
            "latency": 50.0,
            "cost": 2.0
        }
        self.shard_mesh.route_and_execute(workload)
        self.shard_mesh.run_mesh_governance()
        mesh_coh = self.shard_mesh.compute_mesh_coherence()
        self.mesh_coherence_history.append(mesh_coh)

        # 4. Evaluate Cost Vector
        total_act = sum(s.activation for s in self.membrane.strings)
        total_rad = sum(s.radius for s in self.membrane.strings)

        # Compute Quality Signal based on average closure coherence Q_ij(t)
        quality_signal = float(np.mean(Q_matrix)) if Q_matrix.size > 0 else 0.5

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
        if self.holistic_history:
            print(f"Final Holistic Governor Field H_hol: {self.holistic_history[-1]:.4f}")

        # Quadrant Activations at final step
        print("\nFinal Quadrant Activations:")
        for q in ["analytical", "contextual", "generative", "interpersonal"]:
            print(f"  {q.capitalize()}: {self.membrane.get_quadrant_activation(q):.4f}")
        print("======================================")


# Color formatting constants
ANSI_PURPLE = "\033[95m"
ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def print_banner(text: str, color: str = ANSI_CYAN) -> None:
    border = "=" * 75
    print(f"{color}{border}")
    print(f"{ANSI_BOLD}{text.center(75)}{ANSI_RESET}{color}")
    print(f"{border}{ANSI_RESET}\n")


def run_governed_simulation() -> None:
    """
    Runs a deterministic, 10-tick governed simulation using RainbowSimulation.
    Supports animated progression, color-coded metrics, and robust GovernanceError handling.
    """
    # 1. Deterministic Seeding Confirmation
    set_deterministic_env(0)
    print(f"{ANSI_GREEN}{ANSI_BOLD}[Deterministic Seed Initialized: 0]{ANSI_RESET}\n")

    # 2. Cinematic Banner
    print_banner("🛸 UFO SIMULATION MODULE EXECUTING 🛸", ANSI_PURPLE)

    # 3. Instantiate
    sim = RainbowSimulation()

    steps = 1 if os.getenv("UFO_TEST_RUN") else 10
    task_sequence = [
        "technical_deep_analysis",
        "supportive_concise_reply",
        "planning"
    ] * 4  # Cycles through available tasks (at least 12 steps, sliced to 10)

    try:
        for step_idx in range(steps):
            tick_num = step_idx + 1

            # Animated tick prefix
            anim_chars = ["◐", "◓", "◑", "◒"]
            anim = anim_chars[step_idx % len(anim_chars)]
            print(f"{ANSI_BOLD}{ANSI_CYAN}{anim} Tick {tick_num}/{steps}...{ANSI_RESET}", end="\r")
            sys.stdout.flush()
            time.sleep(0.05)

            # Run step
            sim.run_step(task_sequence[step_idx])

            # Extract and format metrics
            curv = max([sim.boundary.curvature(s.theta) for s in sim.membrane.strings])
            t_state = getattr(sim.membrane, "temporal_state", None)
            tens = t_state.accumulated_tension if t_state is not None else 0.0
            coh = sim.mesh_coherence_history[-1] if sim.mesh_coherence_history else 1.0
            energy = sim.governor.energy_history[-1] if sim.governor.energy_history else 0.0

            if energy <= 1.0:
                band = "GREEN"
                band_color = ANSI_GREEN
            elif energy <= 2.0:
                band = "YELLOW"
                band_color = ANSI_YELLOW
            else:
                band = "RED"
                band_color = ANSI_RED

            # Color format strings
            curv_str = f"curvature: {ANSI_PURPLE}{curv:.4f}{ANSI_RESET}"
            tens_str = f"tension: {ANSI_BLUE}{tens:.4f}{ANSI_RESET}"
            coh_str = f"coherence: {ANSI_GREEN}{coh:.4f}{ANSI_RESET}"
            band_str = f"stability band: {band_color}{band}{ANSI_RESET}"

            tick_p = f"[{ANSI_BOLD}Tick {tick_num:2d}/{steps:2d}{ANSI_RESET}]"
            print(f"{tick_p} - {curv_str} | {tens_str} | {coh_str} | {band_str}                  ")

        print()  # newline after steps complete
        sim.render_summary()

    except GovernanceError as e:
        print("\n" + "=" * 75)
        print(f"{ANSI_RED}{ANSI_BOLD}🚨 GOVERNED HALT ENFORCED 🚨{ANSI_RESET}")
        print("=" * 75)
        print(f"{ANSI_RED}{ANSI_BOLD}Violation Details:{ANSI_RESET} {e}")
        print(
            f"{ANSI_YELLOW}{ANSI_BOLD}Halt Ledger Code:{ANSI_RESET} "
            "GovernanceError raised during simulation execution."
        )
        print("=" * 75 + "\n")
        sim.render_summary()

    print_banner("🛸 MODULE EXECUTION COMPLETE — RETURNING TO SHELL 🛸", ANSI_PURPLE)
    input("Press Enter to exit...")


if __name__ == "__main__":
    run_governed_simulation()
