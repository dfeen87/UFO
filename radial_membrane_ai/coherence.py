# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Channel Coherence Ratio module.

This module implements the channel coherence ratio based on shared closure boundaries
along the shortest path between behavioral strings as described in Feeney (2026).
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.boundary import BoundaryGeometry

from radial_membrane_ai.admissibility import angular_decomposition
from radial_membrane_ai.projection import closure_ratio


def closure_coherence(
    membrane: RadialMembrane,
    boundary: BoundaryGeometry,
    i: int,
    j: int,
    samples: int = 32
) -> float:
    """
    Computes the coherence ratio Q_ij(t) between string i and string j.
    The ratio integrates 1 - min(1, i(t, theta)) along the shortest path
    on S^1 between string i and string j.

    Args:
        membrane: The RadialMembrane instance.
        boundary: The BoundaryGeometry instance.
        i: 1-based index of source string.
        j: 1-based index of target string.
        samples: Number of subdivision points along the shortest path.

    Returns:
        Coherence ratio Q_ij(t) in [0, 1].
    """
    if i < 1 or i > 12 or j < 1 or j > 12:
        raise ValueError("String indices must be in [1, 12].")

    theta_i = membrane.strings[i - 1].theta
    theta_j = membrane.strings[j - 1].theta

    # Shortest path on S^1
    diff = (theta_j - theta_i + math.pi) % (2.0 * math.pi) - math.pi

    if samples <= 1:
        angles = [theta_i, theta_j]
    else:
        angles = [theta_i + (diff * float(k) / float(samples - 1)) for k in range(samples)]

    total_coherence = 0.0

    for theta in angles:
        # Extract orthogonal legs
        a, b = angular_decomposition(membrane, theta, samples=64)
        c = boundary.get_radius(theta)

        # Compute local closure ratio
        i_ratio = closure_ratio(a, b, c)

        # Map to [0, 1] coherence: 1 - min(1.0, i_ratio)
        local_coher = 1.0 - min(1.0, i_ratio)
        total_coherence += local_coher

    return total_coherence / len(angles)
