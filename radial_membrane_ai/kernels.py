# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
V-Channel Activation & Kernel Regimes module.

Implements baseline kernel K(t), structured persistence kernel K_HLV(t),
and the V-Channel-paper form of the compute-aware kernel
K_eff(t, theta) = A(0) * u(t, theta) * K_HLV(t),
regime selectors, angular phase alignment, and coherence ratio.

``A(0)`` is an explicitly supplied reference activation.  This surface does
not silently implement the later Unified Math notation ``A(theta)``; callers
that possess angle-local activation must supply that value deliberately.
"""

from __future__ import annotations
import math
from typing import Tuple, List


class KernelRegimeManager:
    """
    Manages structured persistence kernels, baseline kernels, compute-aware kernels,
    and regime switching.
    """

    def __init__(
        self,
        initial_K: float = 1.0,
        initial_K_HLV: float = 1.0,
        alpha_K: float = 0.1,
        gamma_K: float = 0.2,
        beta_HLV: float = 0.05,
        delta_HLV: float = 0.1
    ) -> None:
        self.K_val = initial_K
        self.K_HLV_val = initial_K_HLV
        self.alpha_K = alpha_K
        self.gamma_K = gamma_K
        self.beta_HLV = beta_HLV
        self.delta_HLV = delta_HLV
        self.regimes = [0.0, 0.0, 0.0]  # Regime Selector U = (U_1, U_2, U_3)

    def step_kernels(self, input_excitation: float, v_channel_activity: float) -> Tuple[float, float]:
        """
        Updates baseline kernel K(t) and structured persistence kernel K_HLV(t).
        """
        self.K_val = (1.0 - self.alpha_K) * self.K_val + self.gamma_K * input_excitation
        self.K_val = max(0.0, self.K_val)

        self.K_HLV_val = self.K_HLV_val * math.exp(-self.beta_HLV) + self.delta_HLV * v_channel_activity
        self.K_HLV_val = max(0.0, self.K_HLV_val)

        return self.K_val, self.K_HLV_val

    def compute_effective_kernel(
        self,
        theta: float,
        A_0: float = 1.0,
        u_t_theta: float = 1.0
    ) -> float:
        """
        Computes the V-Channel-paper compute-aware effective kernel:
        K_eff(t, theta) = A(0) * u(t, theta) * K_HLV(t)

        ``theta`` identifies the sampled location for the modulation supplied
        by the caller; it is not itself used to synthesize ``A(theta)``.
        """
        return A_0 * u_t_theta * self.K_HLV_val

    def select_regime(self, load: float, quality: float) -> Tuple[float, float, float]:
        """
        Determines the regime selector vector U = (U_1, U_2, U_3) based on load and quality.
        U_1: Low load / high quality
        U_2: Medium load / balanced quality
        U_3: High load / compromised quality (reconfiguration)
        """
        if load < 0.3 and quality > 0.7:
            self.regimes = [1.0, 0.0, 0.0]
        elif load < 0.7 and quality > 0.4:
            self.regimes = [0.0, 1.0, 0.0]
        else:
            self.regimes = [0.0, 0.0, 1.0]
        return (self.regimes[0], self.regimes[1], self.regimes[2])

    @staticmethod
    def angular_phase_alignment(theta_1: float, theta_2: float) -> float:
        """
        Computes angular phase alignment: (1 + cos(theta_1 - theta_2)) / 2.0
        """
        return (1.0 + math.cos(theta_1 - theta_2)) / 2.0

    @staticmethod
    def coherence_ratio(coherences: List[float]) -> float:
        """
        Calculates coherence ratio Q_ij style average across a sequence.
        """
        return sum(coherences) / len(coherences) if coherences else 0.0
