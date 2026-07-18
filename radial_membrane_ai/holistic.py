"""
Holistic Governor Field module.

This module implements the global coherence field H_hol(t) as described in Feeney (2026).
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.boundary import BoundaryGeometry
    from radial_membrane_ai.facet import FacetVector


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
    w_G = 0.7
    w_C = 0.3

    geom_sum = 0.0
    for facet in facets:
        a = facet.activation
        c = facet.capacity
        p = facet.residual
        pi = facet.policy_priority * global_policy_weight

        facet_term = (a * c) * (1.0 - p**2) * pi
        geom_sum += facet_term

    geom_avg = geom_sum / 12.0 if len(facets) > 0 else 0.0
    coherence_avg = float(np.mean(Q_matrix)) if Q_matrix.size > 0 else 0.0

    return w_G * geom_avg + w_C * coherence_avg
