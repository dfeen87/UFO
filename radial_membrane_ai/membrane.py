"""
Radial Identity Membrane and Behavioral Strings module.

This module implements the mathematical and geometric model of the radial identity
membrane described in Section 3 of Feeney (2025). It defines the BehavioralString
and RadialMembrane classes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class BehavioralString:
    """
    A controllable dimension of behavior on the radial identity membrane.

    Ref: Section 3.1 of Feeney (2025).

    Attributes:
        name: Name of the behavioral dimension.
        index: 1-based index of the string (1 to 12).
        theta: Phase position theta_i = 2 * pi * i / 12.
        activation: Activation coefficient a_i(t) in [0, 1].
        radius: Reasoning radius/depth r_i(t).
        cost: Local cost c_i(t).
        tension: Dynamic tension tau_i(t).
        stiffness: Dynamic stiffness K_i(t).
    """
    name: str
    index: int
    theta: float
    activation: float
    radius: float
    cost: float
    tension: float
    stiffness: float


class RadialMembrane:
    """
    Radial Identity Membrane model.

    Ref: Section 3.2 of Feeney (2025).
    Holds 12 BehavioralStrings and computes the membrane activation field.
    """

    def __init__(self, c_baseline: float = 0.1, basis_sigma: float = math.pi / 6) -> None:
        """
        Initializes the RadialMembrane with 12 standard behavioral strings.

        Args:
            c_baseline: Configurable baseline cost for each string.
            basis_sigma: Standard deviation for the Gaussian basis functions.
        """
        self.c_baseline = c_baseline
        self.basis_sigma = basis_sigma

        # Define names of the 12 behavioral strings grouped by quadrants:
        # Analytical: depth, precision, technical detail, structural rigor
        # Contextual: context sensitivity, transparency
        # Generative: initiative, exploration, creativity
        # Interpersonal: tone, emotional warmth, conciseness
        names = [
            "depth", "precision", "technical_detail", "structural_rigor",
            "context_sensitivity", "transparency",
            "initiative", "exploration", "creativity",
            "tone", "emotional_warmth", "conciseness"
        ]

        self.strings: list[BehavioralString] = []
        for idx, name in enumerate(names, start=1):
            theta = (2.0 * math.pi * idx) / 12.0
            string = BehavioralString(
                name=name,
                index=idx,
                theta=theta,
                activation=0.0,
                radius=0.0,
                cost=c_baseline,
                tension=0.0,
                stiffness=0.0
            )
            self.strings.append(string)

        # Quadrant mapping
        self._quadrant_map = {
            "analytical": {"depth", "precision", "technical_detail", "structural_rigor"},
            "contextual": {"context_sensitivity", "transparency"},
            "generative": {"initiative", "exploration", "creativity"},
            "interpersonal": {"tone", "emotional_warmth", "conciseness"}
        }

    def get_activation_vector(self) -> np.ndarray:
        """
        Returns the coefficient vector a(t) = [a_1(t), ..., a_12(t)] as a numpy array.
        """
        return np.array([s.activation for s in self.strings], dtype=np.float64)

    def update_activation(self, delta: np.ndarray) -> None:
        """
        Implements a(t+1) = a(t) + delta_a_gov(t).
        Clamps the activations to a non-negative range, e.g. [0.0, 1.0].

        Args:
            delta: numpy array of shape (12,) containing delta activation values.
        """
        if len(delta) != 12:
            raise ValueError("Delta vector must have length 12.")

        for idx, s in enumerate(self.strings):
            new_activation = s.activation + float(delta[idx])
            # Clamp activation between 0.0 and 1.0
            s.activation = max(0.0, min(1.0, new_activation))

    @staticmethod
    def angular_distance(theta1: float, theta2: float) -> float:
        """
        Computes the shortest angular distance on S^1 between theta1 and theta2.
        """
        diff = abs(theta1 - theta2) % (2.0 * math.pi)
        return min(diff, 2.0 * math.pi - diff)

    def field_value(self, theta: float, basis_type: str = "gaussian") -> float:
        """
        Computes the membrane activation field value Phi(theta, t) at a given angle.

        Phi(theta, t) = sum_{i=1}^{12} phi_i(theta) * a_i(t)

        Args:
            theta: Angle in radians.
            basis_type: Type of basis function, either "gaussian" or "cosine".

        Returns:
            The computed field value.
        """
        total = 0.0
        for s in self.strings:
            dist = self.angular_distance(theta, s.theta)
            if basis_type == "gaussian":
                # Gaussian bump: phi_i(theta) = exp(-dist^2 / (2 * sigma^2))
                phi = math.exp(-(dist ** 2) / (2.0 * (self.basis_sigma ** 2)))
            elif basis_type == "cosine":
                # Cosine bump: phi_i(theta) = (1 + cos(theta - theta_i)) / 2 or max(0, cos(theta - theta_i))
                # Let's use (1 + cos(dist)) / 2 normalized, or max(0, cos(dist))
                phi = max(0.0, math.cos(dist))
            else:
                raise ValueError("basis_type must be either 'gaussian' or 'cosine'")

            total += phi * s.activation
        return total

    def get_quadrant_activation(self, quadrant_name: str) -> float:
        """
        Returns the average activation across behavioral strings in the specified quadrant.

        Args:
            quadrant_name: One of 'analytical', 'contextual', 'generative', 'interpersonal'
                           (case-insensitive).
        """
        q_name_lower = quadrant_name.lower()
        if q_name_lower not in self._quadrant_map:
            raise ValueError(
                f"Unknown quadrant '{quadrant_name}'. Must be one of "
                f"{list(self._quadrant_map.keys())}"
            )

        target_names = self._quadrant_map[q_name_lower]
        target_strings = [s for s in self.strings if s.name in target_names]

        if not target_strings:
            return 0.0

        return sum(s.activation for s in target_strings) / len(target_strings)
