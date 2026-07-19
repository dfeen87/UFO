"""
Temporal Governance Layer for the U.F.O. architecture.

This module models membrane behavior over time, introducing hysteresis,
temporal tension accumulation, long-range SAO promotion, time-weighted admissibility,
and drift/decay/recovery dynamics.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane


class TemporalMembraneState:
    """
    Tracks and models temporal state over time for a given RadialMembrane.

    Attributes:
        short_horizon: Max history size for short-term tracking (5 ticks).
        long_horizon: Max history size for long-term tracking (20 ticks).
        prior_curvature_history: Curvature values over short/long horizons.
        prior_tension_history: Tension values over short/long horizons.
        prior_admissibility_history: Closure ratios / admissibility flags.
        accumulated_tension: Mathematically accumulated temporal tension.
        consecutive_admissible_ticks: Number of consecutive ticks the membrane was admissible.
        green_ticks_count: Number of consecutive ticks where the effective energy was in green band.
    """

    def __init__(self, short_horizon: int = 5, long_horizon: int = 20) -> None:
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon

        # Double-ended queues to keep history size bounded
        self.curvature_history: deque[float] = deque(maxlen=long_horizon)
        self.tension_history: deque[float] = deque(maxlen=long_horizon)
        self.admissibility_history: deque[float] = deque(maxlen=long_horizon)  # stores max closure ratio

        self.accumulated_tension: float = 0.0
        self.consecutive_admissible_ticks: int = 0
        self.green_ticks_count: int = 0

        # Saved priors for hysteresis
        self.prior_curvature: float = 0.0
        self.prior_tension: float = 0.0

    @property
    def short_tension_history(self) -> List[float]:
        """Returns the list of the most recent short_horizon tension values."""
        hist = list(self.tension_history)
        return hist[-self.short_horizon:] if len(hist) >= self.short_horizon else hist

    @property
    def short_admissibility_history(self) -> List[float]:
        """Returns the list of the most recent short_horizon closure ratios."""
        hist = list(self.admissibility_history)
        return hist[-self.short_horizon:] if len(hist) >= self.short_horizon else hist

    def update_tick_history(self, max_curvature: float, max_tension: float, max_closure_ratio: float) -> None:
        """
        Updates queues with the current tick metrics.
        """
        self.prior_curvature = max_curvature
        self.prior_tension = max_tension

        self.curvature_history.append(max_curvature)
        self.tension_history.append(max_tension)
        self.admissibility_history.append(max_closure_ratio)

        if max_closure_ratio <= 1.0 + 1e-9:
            self.consecutive_admissible_ticks += 1
        else:
            self.consecutive_admissible_ticks = 0

    def compute_exponential_decay_averages(self, eta: float = 0.7) -> Tuple[float, float]:
        """
        Computes the exponentially decaying averages of closure ratio (i) and tension (T).

        Formula:
            avg(t) = eta * avg(t-1) + (1 - eta) * val(t)
        """
        if not self.admissibility_history:
            return 0.0, 0.0

        avg_i = 0.0
        avg_t = 0.0

        # We compute starting from the oldest to newest to reflect chronological exponential decay
        for idx, (i_val, t_val) in enumerate(zip(self.admissibility_history, self.tension_history)):
            if idx == 0:
                avg_i = i_val
                avg_t = t_val
            else:
                avg_i = eta * avg_i + (1.0 - eta) * i_val
                avg_t = eta * avg_t + (1.0 - eta) * t_val

        return avg_i, avg_t

    def get_normalized_tension_history(self) -> float:
        """
        Returns a normalized tension history metric H_T scaled to [0.0, 1.0].
        Clamped to 1.0 if accumulated or average tension is very high.
        """
        if not self.tension_history:
            return 0.0
        # Average over the long-horizon window, divided by a scaling factor
        avg_tension = float(np.mean(self.tension_history))
        return min(1.0, max(0.0, avg_tension / 2.5))

    def update_tension_accumulation(
        self,
        v_channel_load: float,
        cost_taxonomy_contrib: float,
        sao_promotions_intensity: float,
        mem_writes_norm: float,
        gamma: float = 0.8,
        w_v: float = 0.4,
        w_c: float = 0.3,
        w_s: float = 0.2,
        w_m: float = 0.1
    ) -> float:
        """
        Updates and returns the mathematically accumulated temporal tension:
        T_acc(t+1) = gamma * T_acc(t) + w_v * L_v + w_c * C + w_s * P_sao + w_m * W_m
        """
        stimulus = (
            w_v * v_channel_load +
            w_c * cost_taxonomy_contrib +
            w_s * sao_promotions_intensity +
            w_m * mem_writes_norm
        )
        self.accumulated_tension = gamma * self.accumulated_tension + stimulus
        return self.accumulated_tension

    def update_recovery_dynamics(self, is_green: bool, mu: float = 0.1) -> float:
        """
        Tracks green band ticks and returns active relaxation scaling (or mu adjustment).
        """
        if is_green:
            self.green_ticks_count += 1
        else:
            self.green_ticks_count = 0

        # Returns 0.15 (faster relaxation) if green ticks >= 10, else 0.1 (nominal)
        return 0.15 if self.green_ticks_count >= 10 else mu

    def decay_tension(self, low_load: bool, rho: float = 0.7) -> None:
        """
        Enforces tension decay when workload/load drops:
        T(t+1) = rho * T(t)
        """
        if low_load and self.tension_history:
            # Decay both accumulated tension and last history items
            self.accumulated_tension *= rho
            if len(self.tension_history) > 0:
                self.prior_tension *= rho
                # Decay queue elements
                updated_queue = deque([v * rho for v in self.tension_history], maxlen=self.long_horizon)
                self.tension_history = updated_queue

    def apply_curvature_drift(
        self,
        boundary: Any,
        mu: float = 0.1
    ) -> None:
        """
        Applies natural drift of membrane boundary curvature over time back toward neutral baseline.
        We smoothly relax each boundary string's deviation back towards zero:
            deviation(t+1) = deviation(t) + mu * (0.0 - deviation(t))
        """
        for i in range(1, 13):
            dev = boundary.radius_deviation[i]
            boundary.radius_deviation[i] = dev + mu * (0.0 - dev)
