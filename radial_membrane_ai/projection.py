"""
Pythagorean Projection Engine module.

This module implements the mathematical projection of membrane state components
to the admissible capacity boundary as described in Feeney (2026).
"""

from __future__ import annotations
import math
from typing import Any

def closure_ratio(a: float, b: float, c: float) -> float:
    """
    Computes the admissibility (closure) ratio:
    i(t, theta) = (a(t, theta)^2 + b(t, theta)^2) / c(t, theta)^2

    Args:
        a: Orthogonal contribution a(t, theta).
        b: Orthogonal contribution b(t, theta).
        c: Dynamic admissible capacity c(t, theta).

    Returns:
        The admissibility ratio i(t, theta).
    """
    if c <= 1e-9:
        # Avoid division by zero by returning a very large ratio
        return float("inf")
    return (a**2 + b**2) / (c**2)


def admissibility_test(i: float) -> bool:
    """
    Checks if the admissibility ratio satisfies the safety boundary condition.

    Args:
        i: Admissibility ratio.

    Returns:
        True if admissible (i <= 1.0), False otherwise.
    """
    return i <= 1.0 + 1e-9


def project_to_admissible(
    a: float,
    b: float,
    c: float,
    metric: str = "euclidean",
    weights: dict[str, float] | None = None
) -> tuple[float, float]:
    """
    Projects the orthogonal legs (a, b) onto the admissible capacity boundary
    defined by c under the specified metric if the boundary is violated.

    Args:
        a: Orthogonal leg a.
        b: Orthogonal leg b.
        c: Dynamic capacity limit c.
        metric: One of 'euclidean', 'weighted', 'angular', or 'radial'.
        weights: Optional dictionary for 'weighted' metric containing 'w_a' and 'w_b'.

    Returns:
        Tuple of projected (a_proj, b_proj).
    """
    if c <= 1e-9:
        return 0.0, 0.0

    i = closure_ratio(a, b, c)
    if admissibility_test(i):
        return a, b

    norm = math.sqrt(a**2 + b**2)
    metric_lower = metric.lower()

    if metric_lower == "euclidean" or metric_lower == "radial" or metric_lower == "angular":
        # Euclidean, Radial, and Angular projection scale (a, b) to lie on the boundary circle of radius c
        scale = c / norm
        return a * scale, b * scale

    elif metric_lower == "weighted":
        if weights is None:
            weights = {"w_a": 1.0, "w_b": 1.0}
        w_a = weights.get("w_a", 1.0)
        w_b = weights.get("w_b", 1.0)

        if w_a <= 1e-9:
            w_a = 1e-9
        if w_b <= 1e-9:
            w_b = 1e-9

        # Minimize w_a * (a_proj - a)^2 + w_b * (b_proj - b)^2 subject to a_proj^2 + b_proj^2 = c^2
        # Using bisection search for Lagrange multiplier lambda
        # f(lambda) = (w_a * a / (w_a + lambda))^2 + (w_b * b / (w_b + lambda))^2 - c^2 = 0
        # for lambda > -min(w_a, w_b).
        min_w = min(w_a, w_b)
        low = -min_w + 1e-9

        # Estimate high bound
        high = 1.0

        def f(lam: float) -> float:
            return (w_a * a / (w_a + lam))**2 + (w_b * b / (w_b + lam))**2 - c**2

        # Expand high if needed to bracket the root
        for _ in range(50):
            if f(high) < 0:
                break
            high *= 2.0

        # Bisection loop
        for _ in range(100):
            mid = 0.5 * (low + high)
            f_mid = f(mid)
            if abs(f_mid) < 1e-12:
                break
            if f_mid > 0:
                low = mid
            else:
                high = mid

        lam = 0.5 * (low + high)
        a_proj = w_a * a / (w_a + lam)
        b_proj = w_b * b / (w_b + lam)
        return a_proj, b_proj

    else:
        # Fallback to Euclidean
        scale = c / norm
        return a * scale, b * scale


def residual_deformation(a: float, b: float, a_proj: float, b_proj: float) -> tuple[float, float]:
    """
    Computes the residual deformation vector:
    p(t, theta) = A(t, theta) - II_G(A(t, theta))

    Args:
        a: Original orthogonal leg a.
        b: Original orthogonal leg b.
        a_proj: Projected orthogonal leg a.
        b_proj: Projected orthogonal leg b.

    Returns:
        A tuple of (p_a, p_b) representing the residual vector.
    """
    return a - a_proj, b - b_proj
