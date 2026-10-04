# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Holistic Governor Field module.

This module implements the global coherence field H_hol(t) as described in Feeney (2026).
"""

from __future__ import annotations
import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.boundary import BoundaryGeometry
    from radial_membrane_ai.facet import FacetVector
from radial_membrane_ai.numeric import require_finite_real


def compute_holistic_field(
    membrane: RadialMembrane,
    boundary: BoundaryGeometry,
    facets: list[FacetVector],
    Q_matrix: np.ndarray,
    global_policy_weight: float = 1.0
) -> float:
    """
    Computes the global coherence field H_hol(t), integrating membrane geometry,
    facet state, V-channel topology, and policy hierarchy:

    H_hol(t) = w_G * (1 / 12) * sum_i [ (a_i * c_i) * (1 - p_i^2) * pi_i ] + w_C * Mean(Q_matrix)
    where:
    - a_i: Activation of facet i.
    - c_i: Admissible capacity of facet i.
    - p_i: Residual deformation of facet i.
    - pi_i: Policy priority of facet i.
    - Q_matrix: Matrix of coherence values Q_ij(t).
    - w_G: Geometric weight (default 0.7)
    - w_C: Coherence weight (default 0.3)

    Args:
        membrane: The RadialMembrane instance.
        boundary: The BoundaryGeometry instance.
        facets: List of 12 FacetVector instances.
        Q_matrix: 12x12 numpy array representing the coherence matrix.
        global_policy_weight: Multiplier applied to the policy priorities.

    Returns:
        The scalar value of the holistic governor field H_hol(t).
    """
    if len(facets) != 12:
        raise ValueError("holistic field requires exactly twelve facets.")
    matrix = np.asarray(Q_matrix)
    if matrix.shape != (12, 12) or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Q_matrix must be a numeric 12x12 matrix.")
    try:
        finite_matrix = np.asarray(matrix, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Q_matrix must contain representable finite values.") from exc
    if not np.all(np.isfinite(finite_matrix)):
        raise ValueError("Q_matrix must contain representable finite values.")
    global_policy_weight = require_finite_real(global_policy_weight, "global_policy_weight")
    if global_policy_weight < 0.0:
        raise ValueError("global_policy_weight must be non-negative.")
    w_G = 0.7
    w_C = 0.3

    geom_sum = 0.0
    for facet in facets:
        a = require_finite_real(facet.activation, "facet.activation")
        c = require_finite_real(facet.capacity, "facet.capacity")
        p = require_finite_real(facet.residual, "facet.residual")
        priority = require_finite_real(facet.policy_priority, "facet.policy_priority")
        if not 0.0 <= a <= 1.0 or c < 0.0 or p < 0.0 or priority < 0.0:
            raise ValueError("facet evidence is outside its model domain.")
        pi = priority * global_policy_weight

        facet_term = (a * c) * (1.0 - p**2) * pi
        geom_sum += facet_term

    geom_avg = geom_sum / 12.0 if len(facets) > 0 else 0.0
    coherence_avg = float(np.mean(finite_matrix))

    result = w_G * geom_avg + w_C * coherence_avg
    if not math.isfinite(result):
        raise ValueError("holistic field must remain finite.")
    return result


class HolisticGovernorField:
    """
    Tracks and manages the holistic governor field, stability bands, policy tensions,
    kernel modulation, and global reconfiguration triggers.
    """

    def __init__(
        self,
        w_p: float = 0.4,
        w_o: float = 0.2,
        w_c: float = 0.1,
        w_T: float = 0.15,
        w_K: float = 0.15
    ) -> None:
        values = [require_finite_real(v, name) for name, v in (
            ("w_p", w_p), ("w_o", w_o), ("w_c", w_c), ("w_T", w_T), ("w_K", w_K)
        )]
        if any(v < 0.0 for v in values):
            raise ValueError("holistic weights must be non-negative.")
        self.w_p, self.w_o, self.w_c, self.w_T, self.w_K = values
        self.drift_history: list[float] = []

    def compute_H_field(
        self,
        p_avg: float,
        o_avg: float,
        c_avg: float,
        T_avg: float,
        K_avg: float
    ) -> float:
        """
        Decomposes and computes the supervisory field:
        H(t) = w_p * p - w_o * o - w_c * c - w_T * T + w_K * K
        """
        p_avg, o_avg, c_avg, T_avg, K_avg = (
            require_finite_real(v, n) for n, v in (
                ("p_avg", p_avg), ("o_avg", o_avg), ("c_avg", c_avg),
                ("T_avg", T_avg), ("K_avg", K_avg)
            )
        )
        result = self.w_p * p_avg - self.w_o * o_avg - self.w_c * c_avg - self.w_T * T_avg + self.w_K * K_avg
        if not math.isfinite(result):
            raise ValueError("H field must remain finite.")
        return result

    def get_coherence_score(self, H_val: float) -> float:
        """
        Calculates a normalized coherence score C(t) in [0, 1].
        """
        H_val = require_finite_real(H_val, "H_val")
        if H_val >= 0.0:
            return 1.0 / (1.0 + math.exp(-H_val))
        exp_value = math.exp(H_val)
        return exp_value / (1.0 + exp_value)

    def evaluate_stability_band(self, score: float) -> str:
        """
        Returns the stability band (green / yellow / red).
        """
        if score >= 0.7:
            return "green"
        elif score >= 0.4:
            return "yellow"
        else:
            return "red"

    def detect_drift(self, score: float, window: int = 5, threshold: float = 0.15) -> bool:
        """
        Detects significant downward drift in coherence.
        """
        self.drift_history.append(score)
        if len(self.drift_history) < window:
            return False
        recent = self.drift_history[-window:]
        return bool(recent[0] - recent[-1] > threshold)

    def detect_overload(self, T_avg: float, max_threshold: float = 0.8) -> bool:
        """
        Detects average tension overload.
        """
        return T_avg > max_threshold

    def detect_policy_tension(self, p_avg: float, pi_avg: float, threshold: float = 0.5) -> bool:
        """
        Detects tension/conflict between residual deformation and policy priority.
        """
        return (p_avg * pi_avg) > threshold

    def modulate_kernel(self, score: float) -> float:
        """
        Computes a kernel modulation scale factor based on coherence.
        """
        return max(0.1, min(1.0, score * 1.2))

    def check_reconfiguration_trigger(self, score: float, drift: bool, overload: bool) -> bool:
        """
        Triggers global reconfiguration if stability falls into red or critical conditions occur.
        """
        return score < 0.3 or (drift and overload)
