# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
V-Channel Routing and Governed Flow module.

This module implements the phase-aligned V-channel routing, cost-aware propagation,
bounded depth updates, and channel coherence diagnostics described in Section 4
of Feeney (2025), updated with the state-aware routing rule described in Feeney (2026).
"""

from __future__ import annotations
import math
from typing import Callable
import numpy as np

from radial_membrane_ai.membrane import BehavioralString, RadialMembrane
from radial_membrane_ai.facet import route_signal
from radial_membrane_ai.numeric import require_finite_real


def phase_alignment(theta_s: float, theta_t: float) -> float:
    """
    Computes the phase alignment function f(|theta_s - theta_t|).

    f(|theta_s - theta_t|) = (1 + cos(theta_s - theta_t)) / 2

    Args:
        theta_s: Angle of the source behavioral string.
        theta_t: Angle of the target behavioral string.

    Returns:
        Alignment value in [0, 1].
    """
    theta_s = require_finite_real(theta_s, "theta_s")
    theta_t = require_finite_real(theta_t, "theta_t")
    return min(1.0, max(0.0, (1.0 + math.cos(theta_s - theta_t)) / 2.0))


def activation_weight(activation: float, k: float = 10.0, x0: float = 0.5) -> float:
    """
    Computes alpha(a_t), an increasing function of target activation.
    Using a standard sigmoid of the activation.

    alpha(a_t) = 1 / (1 + exp(-k * (activation - x0)))

    Args:
        activation: Activation coefficient a_t in [0, 1].
        k: Logistic growth rate.
        x0: Sigmoid midpoint.

    Returns:
        Weight value in [0, 1].
    """
    activation = require_finite_real(activation, "activation")
    k = require_finite_real(k, "k")
    x0 = require_finite_real(x0, "x0")
    if not 0.0 <= activation <= 1.0:
        raise ValueError("activation must be in [0, 1].")
    if k < 0.0:
        raise ValueError("k must be non-negative.")
    exponent = k * (activation - x0)
    if not math.isfinite(exponent):
        raise ValueError("activation sigmoid exponent must remain finite.")
    if exponent >= 0.0:
        return 1.0 / (1.0 + math.exp(-exponent))
    exp_value = math.exp(exponent)
    return exp_value / (1.0 + exp_value)


def base_propagation(source: BehavioralString, target: BehavioralString) -> float:
    """
    Computes the base propagation P(s -> t).

    P(s -> t) = alpha(a_t) * f(|theta_s - theta_t|)

    Args:
        source: Source BehavioralString.
        target: Target BehavioralString.

    Returns:
        Propagation strength.
    """
    alpha_val = activation_weight(target.activation)
    f_val = phase_alignment(source.theta, target.theta)
    return alpha_val * f_val


def inverse_cost_weight(cost: float, lambda_: float = 1.0, variant: str = "exponential") -> float:
    """
    Computes the inverse cost weight W_t.

    Variant 'simple': W_t = 1 / (1 + cost)
    Variant 'exponential': W_t = exp(-lambda_ * cost)

    Args:
        cost: The local cost of the target behavioral string.
        lambda_: Decay constant (for exponential variant).
        variant: 'simple' or 'exponential'.

    Returns:
        Inverse cost weight.
    """
    cost = require_finite_real(cost, "cost")
    lambda_ = require_finite_real(lambda_, "lambda_")
    if cost < 0.0:
        raise ValueError("cost must be non-negative.")
    if lambda_ < 0.0:
        raise ValueError("lambda_ must be non-negative.")
    if variant == "simple":
        return 1.0 / (1.0 + cost)
    elif variant == "exponential":
        return math.exp(-lambda_ * cost)
    else:
        raise ValueError("variant must be 'simple' or 'exponential'")


def cost_aware_propagation(
    source: BehavioralString,
    target: BehavioralString,
    lambda_: float = 1.0,
    cost_variant: str = "exponential"
) -> float:
    """
    Computes cost-aware propagation P_cost(s -> t) = P(s -> t) * W_t.
    If facet vectors are configured on source and target, applies the state-aware routing rule.

    Args:
        source: Source BehavioralString.
        target: Target BehavioralString.
        lambda_: Decay parameter for cost weight.
        cost_variant: Cost variant, either 'simple' or 'exponential'.

    Returns:
        Cost-aware propagation strength.
    """
    p_base = base_propagation(source, target)
    w_t = inverse_cost_weight(target.cost, lambda_=lambda_, variant=cost_variant)
    p_cost = p_base * w_t

    # Apply state-aware routing if both source and target have facet vectors
    if source.facet is not None and target.facet is not None:
        p_cost = route_signal(source.facet, target.facet, p_cost)

    return p_cost


def update_radius_along_channel(
    source: BehavioralString,
    target: BehavioralString,
    r_max: float,
    h_func: Callable[[float], float] | None = None,
    lambda_: float = 1.0,
    cost_variant: str = "exponential"
) -> float:
    """
    Computes bounded depth update along channel:
    r_t' = min(r_max, r_s * h(P_cost(s -> t)))

    Args:
        source: Source BehavioralString.
        target: Target BehavioralString.
        r_max: Global maximum reasoning radius.
        h_func: Mapping function from propagation strength to depth preservation/attenuation.
                If None, linear mapping h(p) = p is used.
        lambda_: Decay parameter for cost weight.
        cost_variant: Cost variant, either 'simple' or 'exponential'.

    Returns:
        The new reasoning radius for the target behavioral string.
    """
    r_max = require_finite_real(r_max, "r_max")
    if r_max < 0.0:
        raise ValueError("r_max must be non-negative.")
    source_radius = require_finite_real(source.radius, "source.radius")
    if source_radius < 0.0:
        raise ValueError("source.radius must be non-negative.")
    p_cost = cost_aware_propagation(source, target, lambda_=lambda_, cost_variant=cost_variant)
    if h_func is None:
        # Linear default
        h_val = p_cost
    else:
        h_val = h_func(p_cost)

    h_val = require_finite_real(h_val, "h(P)")
    if h_val < 0.0:
        raise ValueError("h(P) must be non-negative.")
    candidate = source_radius * h_val
    if not math.isfinite(candidate):
        raise ValueError("depth candidate must remain finite.")

    return min(r_max, candidate)


def channel_coherence(
    membrane: RadialMembrane,
    i: int,
    j: int,
    samples: int = 64,
    epsilon: float = 1e-8
) -> float:
    """
    Computes the channel coherence ratio C_ij(t) = bar_Phi_ij(t) / (Phi_max(t) + epsilon).

    Where bar_Phi_ij(t) is approximated numerically by sampling angles along the shortest
    path on S^1 between theta_i and theta_j.

    Args:
        membrane: The RadialMembrane instance.
        i: 1-based index of source string (1 to 12).
        j: 1-based index of target string (1 to 12).
        samples: Number of numerical integration samples.
        epsilon: Small stability constant to prevent division by zero.

    Returns:
        Coherence ratio in [0, 1].
    """
    if (isinstance(i, bool) or not isinstance(i, int) or isinstance(j, bool)
            or not isinstance(j, int) or i < 1 or i > 12 or j < 1 or j > 12):
        raise ValueError("String indices i and j must be in range [1, 12].")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer.")
    epsilon = require_finite_real(epsilon, "epsilon")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")

    theta_i = membrane.strings[i - 1].theta
    theta_j = membrane.strings[j - 1].theta

    # Find shortest path on S^1
    diff = (theta_j - theta_i + math.pi) % (2.0 * math.pi) - math.pi

    # If samples is 1, just compute average of endpoints
    if samples <= 1:
        angles = [theta_i, theta_j]
    else:
        angles = [theta_i + (diff * float(k) / float(samples - 1)) for k in range(samples)]

    # Compute average field value
    total_field = sum(membrane.field_value(theta) for theta in angles)
    bar_phi = total_field / len(angles)

    phi_max = max_field_activation(membrane)
    result = bar_phi / (phi_max + epsilon)
    if not math.isfinite(result):
        raise ValueError("channel coherence must remain finite.")
    return min(1.0, max(0.0, result))


def max_field_activation(membrane: RadialMembrane, samples: int = 256) -> float:
    """
    Approximates the maximum of the activation field Phi_max(t) = max_theta Phi(theta, t)
    by uniform sampling on S^1.

    Args:
        membrane: The RadialMembrane instance.
        samples: Number of sampling points on [0, 2*pi].

    Returns:
        Maximum field activation.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer.")
    max_val = 0.0
    for k in range(samples):
        theta = (2.0 * math.pi * k) / samples
        val = require_finite_real(membrane.field_value(theta), "membrane field value")
        if val > max_val:
            max_val = val
    return max_val


class ChannelDiagnostics:
    """
    Helper class providing diagnostic metrics for V-channel routing and flow.
    """

    def __init__(self, membrane: RadialMembrane) -> None:
        self.membrane = membrane

    def compute_all_coherences(self, samples: int = 64) -> np.ndarray:
        """
        Computes a 12x12 coherence matrix for all pairs of behavioral strings.

        Args:
            samples: Numerical sampling resolution.

        Returns:
            Numpy matrix of shape (12, 12).
        """
        matrix = np.zeros((12, 12), dtype=np.float64)
        for i in range(12):
            for j in range(12):
                matrix[i, j] = channel_coherence(self.membrane, i + 1, j + 1, samples=samples)
        return matrix

    def get_highest_coherence_channels(self, threshold: float = 0.5) -> list[tuple[int, int, float]]:
        """
        Returns pairs of (source_idx, target_idx, coherence) that exceed the coherence threshold.
        Excludes self-coherence.
        """
        channels = []
        for i in range(12):
            for j in range(12):
                if i != j:
                    c_val = channel_coherence(self.membrane, i + 1, j + 1)
                    if c_val >= threshold:
                        channels.append((i + 1, j + 1, c_val))
        return sorted(channels, key=lambda x: x[2], reverse=True)
