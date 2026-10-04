# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Deformable Bidirectional Membrane Geometry module.

This module implements the deformable boundary, boundary radius function,
and approximate geometric diagnostics (curvature, asymmetry, tangent)
described in Section 6 of Feeney (2025), updated with the cost-aware and
closure-aware contraction and expansion described in Feeney (2026).
"""

from __future__ import annotations
import math
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.channels import channel_coherence
from radial_membrane_ai.admissibility import angular_decomposition
from radial_membrane_ai.projection import closure_ratio
from radial_membrane_ai.exceptions import GeometryValidationError, ValidationError
from radial_membrane_ai.numeric import require_finite_real


class BoundaryGeometry:
    """
    Deformable Boundary Geometry model on a polar surface.

    Ref: Section 6 of Feeney (2025) & (2026).
    Defines the boundary radius R(theta, t) as a function of angle.
    Allows dynamic stretching under load and collapse under cost pressure.
    """

    def __init__(
        self,
        base_radius: float = 1.0,
        basis_sigma: float = math.pi / 6,
        min_radius_ratio: float = 0.1,
        max_radius_ratio: float = 3.0
    ) -> None:
        """
        Initializes the BoundaryGeometry.

        Args:
            base_radius: Neutral radius R_0.
            basis_sigma: Standard deviation for interpolating deviations across angles.
            min_radius_ratio: Minimum fraction of base_radius permitted (prevents zero/negative radius).
            max_radius_ratio: Maximum multiplier of base_radius permitted.
        """
        base_radius = require_finite_real(base_radius, "base_radius")
        basis_sigma = require_finite_real(basis_sigma, "basis_sigma")
        min_radius_ratio = require_finite_real(min_radius_ratio, "min_radius_ratio")
        max_radius_ratio = require_finite_real(max_radius_ratio, "max_radius_ratio")
        if base_radius <= 0.0:
            raise GeometryValidationError("base_radius must be strictly positive.")
        if basis_sigma <= 0.0:
            raise GeometryValidationError("basis_sigma must be strictly positive.")
        if min_radius_ratio <= 0.0 or max_radius_ratio <= 0.0:
            raise GeometryValidationError("radius ratios must be strictly positive.")
        if min_radius_ratio > max_radius_ratio:
            raise GeometryValidationError("min_radius_ratio cannot exceed max_radius_ratio.")
        self.base_radius = base_radius
        self.basis_sigma = basis_sigma
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self._custom_radius_scale: float = 1.0

        # Store radius deviations for each of the 12 string indices (1 to 12)
        self.radius_deviation: dict[int, float] = {i: 0.0 for i in range(1, 13)}

    def get_radius(self, theta: float) -> float:
        """
        Computes the boundary radius R(theta, t) at any given angle.

        Uses smooth interpolation of string deviations using Gaussian basis functions.
        R(theta, t) = base_radius + sum_{i=1}^{12} phi_i(theta) * deviation_i

        Args:
            theta: Angle in radians.

        Returns:
            The local boundary radius R(theta, t), clamped to [min_radius, max_radius].
        """
        theta = require_finite_real(theta, "theta")
        scale = require_finite_real(self._custom_radius_scale, "custom radius scale")
        if scale <= 0.0:
            raise GeometryValidationError("custom radius scale must be strictly positive.")
        if set(self.radius_deviation) != set(range(1, 13)) or any(
            require_finite_real(value, f"radius_deviation[{index}]") is None
            for index, value in self.radius_deviation.items()
        ):
            raise GeometryValidationError("radius deviations must define twelve finite facets.")
        # Clean angle to [0, 2*pi)
        theta = theta % (2.0 * math.pi)

        total_deviation = 0.0
        total_weight = 0.0

        for i in range(1, 13):
            # Angular position of string i
            theta_i = (2.0 * math.pi * i) / 12.0
            diff = abs(theta - theta_i) % (2.0 * math.pi)
            dist = min(diff, 2.0 * math.pi - diff)

            # Weight using Gaussian bump
            phi = math.exp(-(dist ** 2) / (2.0 * (self.basis_sigma ** 2)))
            total_deviation += phi * self.radius_deviation[i]
            total_weight += phi

        if total_weight > 0.0:
            avg_deviation = total_deviation / total_weight
        else:
            avg_deviation = 0.0

        r = self.base_radius + avg_deviation

        # Clamp radius to stay within physical bounds
        min_r = self.base_radius * self.min_radius_ratio
        max_r = self.base_radius * self.max_radius_ratio
        result = max(min_r, min(max_r, r)) * scale
        if not math.isfinite(result):
            raise GeometryValidationError("boundary radius calculation must remain finite.")
        return result

    def update_boundary(
        self,
        membrane: RadialMembrane,
        task_value: float,
        learning_rate: float = 0.05,
        cost_suppression_weight: float = 0.5,
        coherence_samples: int = 16
    ) -> None:
        """
        Updates the boundary deviations based on activation, depth (radius), cost,
        coherence, and local closure ratios.

        Stretch increases deviation where:
        - Activation and depth (string.radius) are high
        - Task value is high

        Collapse decreases deviation where:
        - Cost is high
        - Coherence is low
        - Governor suppression is strong (high cost, low task value)
        - Local closure i(t, theta) > 1.0 (triggers closure-aware contraction)

        Args:
            membrane: The active RadialMembrane.
            task_value: Task value / reward scalar.
            learning_rate: Scaling rate of deformation.
            cost_suppression_weight: Weight of cost-based suppression.
            coherence_samples: Samples for computing channel coherence.
        """
        task_value = require_finite_real(task_value, "task_value")
        learning_rate = require_finite_real(learning_rate, "learning_rate")
        suppression = require_finite_real(cost_suppression_weight, "cost_suppression_weight")
        if learning_rate < 0.0:
            raise ValidationError("learning_rate must be non-negative.")
        if suppression < 0.0:
            raise ValidationError("cost_suppression_weight must be non-negative.")
        if isinstance(coherence_samples, bool) or not isinstance(coherence_samples, int) or coherence_samples > 100_000:
            raise ValidationError("coherence_samples must be a practical positive integer.")
        if coherence_samples <= 0:
            raise ValidationError("coherence_samples must be a practical positive integer.")
        if len(membrane.strings) != 12:
            raise GeometryValidationError("boundary updates require exactly twelve strings.")

        # 1. Compute every candidate from authoritative state before committing.
        coherences = {}
        for i in range(1, 13):
            c_sum = 0.0
            for j in range(1, 13):
                if i != j:
                    c_sum += channel_coherence(membrane, i, j, samples=coherence_samples)
            coherences[i] = c_sum / 11.0

        candidates = dict(self.radius_deviation)
        for s in membrane.strings:
            idx = s.index
            # High activation, high depth (s.radius), high task value promotes expansion
            stretch_pressure = task_value * s.activation * s.radius

            # High cost, low coherence, strong suppression (cost > task_value) promotes collapse
            coherence_i = coherences[idx]
            incoherence = max(0.0, 1.0 - coherence_i)
            collapse_pressure = suppression * s.cost * incoherence

            # If cost exceeds task value under low task value, add extra collapse pressure
            if s.cost > task_value and task_value < 0.3:
                collapse_pressure += (s.cost - task_value)

            # Closure-ratio awareness: if local closure at the string's angle theta is violated (>1.0),
            # trigger an additional immediate local boundary contraction/suppression proportional to the violation.
            a_theta, b_theta = angular_decomposition(membrane, s.theta, samples=64)
            c_theta = self.get_radius(s.theta)
            i_ratio = closure_ratio(a_theta, b_theta, c_theta)
            if i_ratio > 1.0:
                collapse_pressure += 2.0 * (i_ratio - 1.0)

            delta_deviation = learning_rate * (stretch_pressure - collapse_pressure)

            new_dev = self.radius_deviation[idx] + delta_deviation

            # Bound individual string's local radius deviation before storing
            min_dev = self.base_radius * (self.min_radius_ratio - 1.0)
            max_dev = self.base_radius * (self.max_radius_ratio - 1.0)
            if not math.isfinite(new_dev):
                raise GeometryValidationError("candidate boundary deformation must remain finite.")
            candidates[idx] = max(min_dev, min(max_dev, new_dev))

        if set(candidates) != set(range(1, 13)) or any(not math.isfinite(v) for v in candidates.values()):
            raise GeometryValidationError("candidate boundary geometry is invalid.")
        self.radius_deviation = candidates

    def curvature(self, theta: float, delta: float = 0.01) -> float:
        """
        Computes curvature of the boundary radius w.r.t. angle using second-order finite differences.

        d^2R / dtheta^2 = (R(theta + delta) - 2 * R(theta) + R(theta - delta)) / delta^2

        Args:
            theta: Angle in radians.
            delta: Small change in radians.

        Returns:
            Curvature value.
        """
        theta = require_finite_real(theta, "theta")
        delta = require_finite_real(delta, "delta")
        if delta <= 0.0:
            raise GeometryValidationError("delta must be strictly positive.")
        r_plus = self.get_radius(theta + delta)
        r_mid = self.get_radius(theta)
        r_minus = self.get_radius(theta - delta)

        return (r_plus - 2.0 * r_mid + r_minus) / (delta ** 2)

    def asymmetry(self, theta: float) -> float:
        """
        Computes asymmetry across opposite angles.

        Asymmetry = R(theta) - R(theta + pi)

        Args:
            theta: Angle in radians.

        Returns:
            Difference in radius.
        """
        theta = require_finite_real(theta, "theta")
        r_theta = self.get_radius(theta)
        r_opposite = self.get_radius(theta + math.pi)
        return r_theta - r_opposite

    def tangent(self, theta: float, delta: float = 0.01) -> float:
        """
        Computes derivative of the radius w.r.t. angle (tangent direction factor)
        using central differences.

        dR / dtheta = (R(theta + delta) - R(theta - delta)) / (2 * delta)

        Args:
            theta: Angle in radians.
            delta: Small change in radians.

        Returns:
            First derivative of radius.
        """
        theta = require_finite_real(theta, "theta")
        delta = require_finite_real(delta, "delta")
        if delta <= 0.0:
            raise GeometryValidationError("delta must be strictly positive.")
        r_plus = self.get_radius(theta + delta)
        r_minus = self.get_radius(theta - delta)
        return (r_plus - r_minus) / (2.0 * delta)
