# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Angular Decomposition, Dynamic Capacity, Stability Pipeline, and Kernel-Based Leg Evolution module.

This module implements the angular decomposition of the membrane field, local closure test,
stability pipeline, kernel leg evolution, and boundary loop trace as described in Feeney (2026).
"""

from __future__ import annotations
import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.governor import Governor
    from radial_membrane_ai.boundary import BoundaryGeometry

from radial_membrane_ai.projection import closure_ratio, admissibility_test


def angular_decomposition(
    membrane: RadialMembrane,
    theta: float,
    samples: int = 128
) -> tuple[float, float]:
    """
    Treats the membrane field M(theta, t) as an element of L^2([0, 2*pi]) and extracts
    the orthogonal legs a(t, theta) and b(t, theta) based on projection onto basis functions:
    - a(t, theta) = (1 / pi) * Integral_0^{2*pi} M(phi, t) * cos(phi - theta) dphi
    - b(t, theta) = (1 / pi) * Integral_0^{2*pi} M(phi, t) * sin(phi - theta) dphi

    Args:
        membrane: The RadialMembrane instance.
        theta: The focal angle in radians.
        samples: Numerical integration resolution.

    Returns:
        A tuple (a, b) of orthogonal legs.
    """
    integral_cos = 0.0
    integral_sin = 0.0

    # Numerical integration using midpoint or trapezoid rule on [0, 2*pi]
    dphi = (2.0 * math.pi) / samples
    for k in range(samples):
        phi = (k + 0.5) * dphi
        field_val = membrane.field_value(phi)

        integral_cos += field_val * math.cos(phi - theta) * dphi
        integral_sin += field_val * math.sin(phi - theta) * dphi

    a = integral_cos / math.pi
    b = integral_sin / math.pi
    return a, b


def local_closure_test(a: float, b: float, c: float) -> bool:
    """
    Tests if the local closure is admissible:
    i(t, theta) <= 1.0

    Args:
        a: Orthogonal leg a.
        b: Orthogonal leg b.
        c: Dynamic admissible capacity c.

    Returns:
        True if admissible, False otherwise.
    """
    i = closure_ratio(a, b, c)
    return admissibility_test(i)


class KernelEvolution:
    """
    Tracks time-evolving kernels K(t), K_HLV(t), and K_eff(t, theta) as leg lengths.

    Equations:
    - K(t+1) = (1 - alpha_K) * K(t) + gamma_K * input_excitation
    - K_HLV(t+1) = K_HLV(t) * exp(-beta_HLV) + delta_HLV * V_channel_activity
    - K_eff(t, theta) = K(t) * K_HLV(t) / (1 + cost_pressure)
    """

    def __init__(
        self,
        alpha_K: float = 0.1,
        gamma_K: float = 0.2,
        beta_HLV: float = 0.05,
        delta_HLV: float = 0.1,
        initial_K: float = 1.0,
        initial_K_HLV: float = 1.0
    ) -> None:
        self.alpha_K = alpha_K
        self.gamma_K = gamma_K
        self.beta_HLV = beta_HLV
        self.delta_HLV = delta_HLV

        self.K = initial_K
        self.K_HLV = initial_K_HLV

    def step(self, input_excitation: float, v_channel_activity: float) -> tuple[float, float]:
        """
        Advances the baseline and HLV kernels by one time step.

        Returns:
            The new tuple (K, K_HLV).
        """
        self.K = (1.0 - self.alpha_K) * self.K + self.gamma_K * input_excitation
        self.K = max(0.0, self.K)

        self.K_HLV = self.K_HLV * math.exp(-self.beta_HLV) + self.delta_HLV * v_channel_activity
        self.K_HLV = max(0.0, self.K_HLV)

        return self.K, self.K_HLV

    def get_effective_kernel(self, theta: float, cost_pressure: float) -> float:
        """
        Computes K_eff(t, theta).
        """
        return (self.K * self.K_HLV) / (1.0 + max(0.0, cost_pressure))


def apply_lyapunov_dissipation(
    membrane: RadialMembrane,
    governor: Governor,
    target_energy: float | None = None,
    dissipation_rate: float = 0.05
) -> None:
    """
    Applies Lyapunov-style global energy dissipation. If the current energy exceeds
    a target energy, damp/scale down all membrane activations.

    Args:
        membrane: The RadialMembrane instance.
        governor: The Governor instance (which tracks energy).
        target_energy: The maximum acceptable energy limit.
        dissipation_rate: The scaling dampening factor to apply to activations.
    """
    current_energy = governor.compute_lyapunov_energy(membrane)

    if target_energy is None:
        # If no target, use average of history if history exists, else do nothing
        if len(governor.energy_history) > 1:
            target_energy = float(np.mean(governor.energy_history[:-1]))
        else:
            return

    if current_energy > target_energy:
        # Scale down activations to dissipate energy
        factor = 1.0 - dissipation_rate
        for s in membrane.strings:
            s.activation = max(0.0, s.activation * factor)


def phase_smoothing(membrane: RadialMembrane, window_size: int = 3) -> None:
    """
    Applies phase-smoothing across active behavioral strings on S^1 using a
    circular sliding window average of activations.

    Args:
        membrane: The RadialMembrane instance.
        window_size: Odd integer indicating the number of neighboring strings to smooth.
    """
    if window_size < 3:
        return

    half_w = window_size // 2
    n = len(membrane.strings)
    old_activations = [s.activation for s in membrane.strings]

    for i in range(n):
        total = 0.0
        count = 0
        for k in range(-half_w, half_w + 1):
            idx = (i + k) % n
            total += old_activations[idx]
            count += 1
        membrane.strings[i].activation = total / count


def boundary_loop_trace(boundary: BoundaryGeometry, samples: int = 256) -> float:
    """
    Computes the polar coordinate line integral (arc length) of the deformable boundary:
    L = Integral_0^{2*pi} sqrt( R(theta)^2 + (dR/dtheta)^2 ) dtheta

    Args:
        boundary: The BoundaryGeometry instance.
        samples: Number of subdivision intervals.

    Returns:
        The boundary loop trace (arc length).
    """
    total_length = 0.0
    dtheta = (2.0 * math.pi) / samples

    for k in range(samples):
        theta = k * dtheta
        r = boundary.get_radius(theta)
        # Numerical derivative dR/dtheta
        dr_dtheta = boundary.tangent(theta)

        total_length += math.sqrt(r**2 + dr_dtheta**2) * dtheta

    return total_length


def radial_depth_contribution(membrane: RadialMembrane, theta: float) -> float:
    """
    Computes the radial depth contribution at a given angle theta by interpolating
    the radii (depths) of the behavioral strings using Gaussian basis functions.
    """
    total = 0.0
    total_weight = 0.0
    for s in membrane.strings:
        dist = membrane.angular_distance(theta, s.theta)
        phi = math.exp(-(dist ** 2) / (2.0 * (membrane.basis_sigma ** 2)))
        total += phi * s.radius
        total_weight += phi
    return total / total_weight if total_weight > 0.0 else 0.0


def dynamic_capacity_boundary(boundary: BoundaryGeometry, theta: float) -> float:
    """
    Returns the dynamic capacity boundary c(t, theta) induced by boundary deformation.
    """
    return boundary.get_radius(theta)


def dynamic_capacity_boundary_temporal(
    boundary: BoundaryGeometry,
    theta: float,
    membrane: RadialMembrane,
    delta_scaling: float = 0.3,
    beta_scaling: float = 0.25
) -> float:
    """
    Returns the dynamic capacity boundary c_dyn(t, theta) under temporal effects,
    incorporating geometric hysteresis and time-weighted admissibility.

    Formula:
        c_dyn = c_base * (1 - beta * H_T) * (1 - delta * avg_T)

    Where:
        - H_T is the normalized tension history (0-1).
        - avg_T is the exponentially decaying average of tension.
    """
    c_base = boundary.get_radius(theta)
    t_state = getattr(membrane, "temporal_state", None)
    if t_state is None:
        return c_base

    h_t = t_state.get_normalized_tension_history()
    _, avg_t = t_state.compute_exponential_decay_averages(eta=0.7)

    # Scale the boundary by both geometric hysteresis and time-weighted tension
    c_dyn = c_base * (1.0 - beta_scaling * h_t) * (1.0 - delta_scaling * avg_t)
    return max(c_base * 0.1, c_dyn)  # Maintain a safety floor of 10% base radius


def angular_projections(
    membrane: RadialMembrane,
    theta: float,
    samples: int = 128
) -> tuple[float, float]:
    """
    Extracts the orthogonal contributions a(t, theta) and b(t, theta) from the membrane field.
    An alias/wrapper for angular_decomposition to match the Pythagorean projection grammar.
    """
    return angular_decomposition(membrane, theta, samples)


def local_closure_test_at_angle(
    membrane: RadialMembrane,
    boundary: BoundaryGeometry,
    theta: float,
    samples: int = 128
) -> bool:
    """
    Performs a complete local closure test at angle theta.
    Decomposes field to extract a and b, computes dynamic capacity boundary c,
    and checks if the admissibility condition holds: i(t, theta) <= 1.0
    """
    a, b = angular_projections(membrane, theta, samples)
    c = dynamic_capacity_boundary(boundary, theta)
    return local_closure_test(a, b, c)


def global_closure_aggregation(
    membrane: RadialMembrane,
    boundary: BoundaryGeometry,
    samples: int = 32
) -> dict[str, float | bool]:
    """
    Aggregates closure ratios across the entire membrane.
    Samples angles uniformly and computes the average ratio, max ratio,
    and a global admissibility flag (True if all sampled points are admissible).
    """
    ratios = []
    dtheta = (2.0 * math.pi) / samples
    all_admissible = True

    for k in range(samples):
        theta = k * dtheta
        a, b = angular_projections(membrane, theta, samples=64)
        c = dynamic_capacity_boundary(boundary, theta)
        i_ratio = closure_ratio(a, b, c)
        ratios.append(i_ratio)
        if i_ratio > 1.0 + 1e-9:
            all_admissible = False

    return {
        "average_ratio": float(np.mean(ratios)),
        "max_ratio": float(np.max(ratios)),
        "is_admissible": all_admissible
    }
