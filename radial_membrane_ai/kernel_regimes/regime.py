# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Kernel Regime classes and definitions for the U.F.O. Kernel Regime Expansion Layer.
"""

from __future__ import annotations
import random
from enum import Enum
from typing import Any, Callable, Dict, Optional


class KernelRegimeType(Enum):
    BALANCED = "balanced"
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    HIGH_CURVATURE = "high_curvature"
    ADVERSARIAL = "adversarial"
    MULTI_PHASE = "multi_phase"


class KernelRegime:
    """
    Defines a governed behavioral regime with specific rules that act as plug-in physics modules.
    Each rule is a callable with signature:
        rule(agent, cluster, engine) -> Any
    """

    def __init__(
        self,
        name: str,
        regime_type: KernelRegimeType,
        closure_ratio_rule: Callable[[Any, Any, Any], Any],
        tension_rule: Callable[[Any, Any, Any], Any],
        capacity_rule: Callable[[Any, Any, Any], Any],
        sao_rule: Callable[[Any, Any, Any], Any],
        stability_rule: Callable[[Any, Any, Any], Any],
        temporal_rule: Callable[[Any, Any, Any], Any],
        policy_rule: Callable[[Any, Any, Any], Any]
    ) -> None:
        self.name = name
        self.regime_type = regime_type
        self.closure_ratio_rule = closure_ratio_rule
        self.tension_rule = tension_rule
        self.capacity_rule = capacity_rule
        self.sao_rule = sao_rule
        self.stability_rule = stability_rule
        self.temporal_rule = temporal_rule
        self.policy_rule = policy_rule


# --- Helper Rule Implementations ---

def standard_policy_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {"aligned": True}


# 0. BALANCED (LEGACY/DEFAULT) REGIME RULES

def balanced_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {
        "admissible": True,
        "clamped_activations": None,
        "violation": False,
        "rollback_type": None
    }


def balanced_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    return 0.0


def balanced_capacity_rule(agent: Any, cluster: Any, engine: Any) -> None:
    return None


def balanced_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {
        "allowed_ranges": {"short", "mid", "long"},
        "coherence_required": 0.0
    }


def balanced_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {
        "valid": True,
        "band": "green",
        "rollback": False,
        "quarantine": False,
        "violation_severity": "none"
    }


def balanced_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {
        "disable_hysteresis": False
    }


# 1. DETERMINISTIC REGIME RULES

def deterministic_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    """
    Strict closure ratio: i(t) <= 1.0.
    If i(t) > 1.0 for any string, clamp activation to boundary, log violation,
    and trigger soft rollback (restore previous activations only).
    """
    violation = False
    clamped_activations = None

    # We evaluate for the active agent
    active_agent = agent if agent is not None else engine
    if active_agent is not None and hasattr(active_agent, "membrane"):
        membrane = active_agent.membrane
        boundary = active_agent.boundary

        # Calculate closure ratios using angular decomposition
        from radial_membrane_ai.admissibility import angular_decomposition
        from radial_membrane_ai.projection import closure_ratio

        cl_ratios = []
        for s in membrane.strings:
            a_theta, b_theta = angular_decomposition(membrane, s.theta, samples=32)
            c_theta = boundary.get_radius(s.theta)
            cl_ratios.append(closure_ratio(a_theta, b_theta, c_theta))

        if any(v > 1.0 for v in cl_ratios):
            violation = True
            # Clamp activation to boundary
            clamped_activations = []
            for idx, s in enumerate(membrane.strings):
                # Clamp s.activation to max possible based on boundary capacity and current angular decomposition
                a_theta, b_theta = angular_decomposition(membrane, s.theta, samples=32)
                c_theta = boundary.get_radius(s.theta)
                # If a_theta + b_theta > c_theta, scale activation down
                if (a_theta + b_theta) > c_theta and (a_theta + b_theta) > 1e-9:
                    scale = c_theta / (a_theta + b_theta)
                    clamped_activations.append(max(0.0, min(1.0, s.activation * scale)))
                else:
                    clamped_activations.append(s.activation)

    return {
        "admissible": not violation,
        "clamped_activations": clamped_activations,
        "violation": violation,
        "rollback_type": "soft" if violation else None
    }


def deterministic_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    # No noise added to tension
    return 0.0


def deterministic_capacity_rule(agent: Any, cluster: Any, engine: Any) -> None:
    # Standard dynamic capacity boundary
    return None


def deterministic_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # only short-range SAO allowed
    return {
        "allowed_ranges": {"short"},
        "coherence_required": 0.0
    }


def deterministic_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    """
    Deterministic stability: must be green.
    """
    # Look at the last band determined by the engine
    active_engine = engine
    if active_engine is not None and hasattr(active_engine, "band_history") and active_engine.band_history:
        current_band = active_engine.band_history[-1]
    elif cluster is not None and hasattr(cluster, "stability_band"):
        current_band = cluster.stability_band
    else:
        current_band = "green"

    violation = (current_band != "green")
    return {
        "valid": not violation,
        "band": current_band,
        "rollback": violation,
        "quarantine": violation and (current_band == "red"),
        "violation_severity": "high" if current_band == "red" else "medium" if violation else "none"
    }


def deterministic_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Bypasses hysteresis: A_new = A_instant (meaning alpha/damping is 0.0)
    return {
        "disable_hysteresis": True
    }


# 2. STOCHASTIC REGIME RULES

def stochastic_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Standard admissibility rules
    return {
        "admissible": True,
        "clamped_activations": None,
        "violation": False,
        "rollback_type": None
    }


def stochastic_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    # Tension accumulation includes stochastic noise: epsilon ~ N(0, 0.05)
    return random.normalvariate(0.0, 0.05)


def stochastic_capacity_rule(agent: Any, cluster: Any, engine: Any) -> None:
    return None


def stochastic_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # SAO promotion requires coherence >= 0.75. Allows short and mid-range, blocks long-range.
    active_agent = agent if agent is not None else engine
    coherence = 1.0
    if active_agent is not None and hasattr(active_agent, "compute_local_coherence"):
        coherence = active_agent.compute_local_coherence()
    elif cluster is not None and hasattr(cluster, "compute_cluster_coherence"):
        coherence = cluster.compute_cluster_coherence()

    coherence_ok = (coherence >= 0.75)
    return {
        "allowed_ranges": {"short", "mid"} if coherence_ok else set(),
        "coherence_required": 0.75
    }


def stochastic_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # green or yellow allowed. Red is a violation (triggers rollback + quarantine)
    active_engine = engine
    if active_engine is not None and hasattr(active_engine, "band_history") and active_engine.band_history:
        current_band = active_engine.band_history[-1]
    elif cluster is not None and hasattr(cluster, "stability_band"):
        current_band = cluster.stability_band
    else:
        current_band = "green"

    violation = (current_band == "red")
    return {
        "valid": not violation,
        "band": current_band,
        "rollback": violation,
        "quarantine": violation,  # Red band triggers quarantine
        "violation_severity": "high" if violation else "none"
    }


def stochastic_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Controlled randomness in activation updates: A_new = clamp(A_instant + epsilon), epsilon ~ N(0, 0.05)
    return {
        "activation_noise": lambda: random.normalvariate(0.0, 0.05)
    }


# 3. HIGH-CURVATURE REGIME RULES

def high_curvature_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    return {
        "admissible": True,
        "clamped_activations": None,
        "violation": False,
        "rollback_type": None
    }


def high_curvature_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    # Faster tension accumulation: multiply new tension by 2.0 (handled in engine tick)
    return 0.0


def high_curvature_capacity_rule(agent: Any, cluster: Any, engine: Any) -> Optional[float]:
    # Capacity boundary shrinks: c_eff = c * (1 - 0.4 * T_bar)
    # T_bar = temporal tension average over last 10 ticks.
    active_agent = agent if agent is not None else engine
    t_state = None
    if active_agent is not None:
        if hasattr(active_agent, "membrane") and hasattr(active_agent.membrane, "temporal_state"):
            t_state = active_agent.membrane.temporal_state
        elif hasattr(active_agent, "temporal_state"):
            t_state = active_agent.temporal_state

    if t_state is not None and len(t_state.tension_history) > 0:
        # Get last 10 ticks tension average
        hist = list(t_state.tension_history)
        last_10 = hist[-10:] if len(hist) >= 10 else hist
        t_bar = float(sum(last_10) / len(last_10))
        return 1.0 - 0.4 * t_bar
    return 1.0


def high_curvature_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # only long-range SAO allowed. Blocks short/mid.
    return {
        "allowed_ranges": {"long"},
        "coherence_required": 0.0
    }


def high_curvature_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # yellow allowed, red forbidden
    active_engine = engine
    if active_engine is not None and hasattr(active_engine, "band_history") and active_engine.band_history:
        current_band = active_engine.band_history[-1]
    elif cluster is not None and hasattr(cluster, "stability_band"):
        current_band = cluster.stability_band
    else:
        current_band = "green"

    violation = (current_band == "red")
    return {
        "valid": not violation,
        "band": current_band,
        "rollback": violation,
        "quarantine": violation,
        "violation_severity": "high" if violation else "none"
    }


def high_curvature_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Curvature amplified: K_new = 1.5 * K (multiply curvature by 1.5)
    # Tension accumulates faster: multiply new tension by 2.0
    return {
        "curvature_amplifier": 1.5,
        "tension_accumulation_multiplier": 2.0
    }


# 4. ADVERSARIAL REGIME RULES

def adversarial_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Closure ratio constraints loosen slightly: i(t) <= 1.15
    return {
        "admissible_threshold": 1.15,
        "admissible": True,
        "clamped_activations": None,
        "violation": False,
        "rollback_type": None
    }


def adversarial_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    return 0.0


def adversarial_capacity_rule(agent: Any, cluster: Any, engine: Any) -> None:
    return None


def adversarial_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # SAO restricted to cluster level only:
    # agent -> cluster allowed, cluster -> global blocked, global SAO blocked.
    return {
        "allowed_ranges": {"short", "mid", "long"},  # standard internal
        "restrict_to_cluster": True,
        "coherence_required": 0.0
    }


def adversarial_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # yellow allowed, red triggers rollback.
    # Stability bands more sensitive: thresholds multiplied by 0.8
    active_engine = engine
    if active_engine is not None and hasattr(active_engine, "band_history") and active_engine.band_history:
        current_band = active_engine.band_history[-1]
    elif cluster is not None and hasattr(cluster, "stability_band"):
        current_band = cluster.stability_band
    else:
        current_band = "green"

    violation = (current_band == "red")
    return {
        "valid": not violation,
        "band": current_band,
        "rollback": violation,
        "quarantine": violation,
        "violation_severity": "high" if violation else "none",
        "threshold_multiplier": 0.8
    }


def adversarial_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Bounded uniform adversarial noise: delta ~ U(-0.05, 0.05)
    return {
        "activation_noise": lambda: random.uniform(-0.05, 0.05)
    }


# 5. MULTI-PHASE REGIME RULES

def multiphase_closure_ratio_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Depends on current phase (deterministic or stochastic)
    phase = "deterministic"
    # Find active manager
    manager = None
    if engine is not None and hasattr(engine, "regime_manager"):
        manager = engine.regime_manager
    elif cluster is not None and hasattr(cluster, "regime_manager"):
        manager = cluster.regime_manager

    if manager is not None:
        phase = manager.get_active_multiphase_phase(agent)

    if phase == "deterministic":
        return deterministic_closure_ratio_rule(agent, cluster, engine)
    else:
        return stochastic_closure_ratio_rule(agent, cluster, engine)


def multiphase_tension_rule(agent: Any, cluster: Any, engine: Any) -> float:
    # Depends on current phase
    phase = "deterministic"
    manager = None
    if engine is not None and hasattr(engine, "regime_manager"):
        manager = engine.regime_manager
    elif cluster is not None and hasattr(cluster, "regime_manager"):
        manager = cluster.regime_manager

    if manager is not None:
        phase = manager.get_active_multiphase_phase(agent)

    if phase == "deterministic":
        return deterministic_tension_rule(agent, cluster, engine)
    else:
        return stochastic_tension_rule(agent, cluster, engine)


def multiphase_capacity_rule(agent: Any, cluster: Any, engine: Any) -> None:
    return None


def multiphase_sao_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # SAO allowed only at phase boundary ticks: tick 10, 20, 30, etc. (boundary of the 10-tick phase window)
    active_engine = engine
    is_boundary = False
    if active_engine is not None and hasattr(active_engine, "tick_count"):
        # Since we alternate 10 ticks deterministic, 10 ticks stochastic, phase boundary is at tick_count % 10 == 0
        is_boundary = (active_engine.tick_count > 0 and active_engine.tick_count % 10 == 0)
    elif active_engine is not None and hasattr(active_engine, "temporal_state"):
        consec = active_engine.temporal_state.consecutive_admissible_ticks
        is_boundary = (consec > 0 and consec % 10 == 0)

    return {
        "allowed_ranges": {"short", "mid", "long"} if is_boundary else set(),
        "coherence_required": 0.0,
        "only_at_phase_boundary": True,
        "is_boundary": is_boundary
    }


def multiphase_stability_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Depends on phase
    phase = "deterministic"
    manager = None
    if engine is not None and hasattr(engine, "regime_manager"):
        manager = engine.regime_manager
    elif cluster is not None and hasattr(cluster, "regime_manager"):
        manager = cluster.regime_manager

    if manager is not None:
        phase = manager.get_active_multiphase_phase(agent)

    if phase == "deterministic":
        return deterministic_stability_rule(agent, cluster, engine)
    else:
        return stochastic_stability_rule(agent, cluster, engine)


def multiphase_temporal_rule(agent: Any, cluster: Any, engine: Any) -> Dict[str, Any]:
    # Track phase ticks. Alternates: deterministic for 10 ticks, then stochastic for 10 ticks.
    # Phase transition triggered by T_bar > 0.8
    # Returns temporal configuration
    phase = "deterministic"
    manager = None
    if engine is not None and hasattr(engine, "regime_manager"):
        manager = engine.regime_manager
    elif cluster is not None and hasattr(cluster, "regime_manager"):
        manager = cluster.regime_manager

    if manager is not None:
        phase = manager.get_active_multiphase_phase(agent)

    if phase == "deterministic":
        return {
            "disable_hysteresis": True,
            "multiphase": True,
            "phase": phase
        }
    else:
        return {
            "activation_noise": lambda: random.normalvariate(0.0, 0.05),
            "multiphase": True,
            "phase": phase
        }


# --- REGIME DICTIONARY ---

BALANCED_REGIME = KernelRegime(
    name="Balanced",
    regime_type=KernelRegimeType.BALANCED,
    closure_ratio_rule=balanced_closure_ratio_rule,
    tension_rule=balanced_tension_rule,
    capacity_rule=balanced_capacity_rule,
    sao_rule=balanced_sao_rule,
    stability_rule=balanced_stability_rule,
    temporal_rule=balanced_temporal_rule,
    policy_rule=standard_policy_rule
)

DETERMINISTIC_REGIME = KernelRegime(
    name="Deterministic",
    regime_type=KernelRegimeType.DETERMINISTIC,
    closure_ratio_rule=deterministic_closure_ratio_rule,
    tension_rule=deterministic_tension_rule,
    capacity_rule=deterministic_capacity_rule,
    sao_rule=deterministic_sao_rule,
    stability_rule=deterministic_stability_rule,
    temporal_rule=deterministic_temporal_rule,
    policy_rule=standard_policy_rule
)

STOCHASTIC_REGIME = KernelRegime(
    name="Stochastic",
    regime_type=KernelRegimeType.STOCHASTIC,
    closure_ratio_rule=stochastic_closure_ratio_rule,
    tension_rule=stochastic_tension_rule,
    capacity_rule=stochastic_capacity_rule,
    sao_rule=stochastic_sao_rule,
    stability_rule=stochastic_stability_rule,
    temporal_rule=stochastic_temporal_rule,
    policy_rule=standard_policy_rule
)

HIGH_CURVATURE_REGIME = KernelRegime(
    name="High-Curvature",
    regime_type=KernelRegimeType.HIGH_CURVATURE,
    closure_ratio_rule=high_curvature_closure_ratio_rule,
    tension_rule=high_curvature_tension_rule,
    capacity_rule=high_curvature_capacity_rule,
    sao_rule=high_curvature_sao_rule,
    stability_rule=high_curvature_stability_rule,
    temporal_rule=high_curvature_temporal_rule,
    policy_rule=standard_policy_rule
)

ADVERSARIAL_REGIME = KernelRegime(
    name="Adversarial",
    regime_type=KernelRegimeType.ADVERSARIAL,
    closure_ratio_rule=adversarial_closure_ratio_rule,
    tension_rule=adversarial_tension_rule,
    capacity_rule=adversarial_capacity_rule,
    sao_rule=adversarial_sao_rule,
    stability_rule=adversarial_stability_rule,
    temporal_rule=adversarial_temporal_rule,
    policy_rule=standard_policy_rule
)

MULTI_PHASE_REGIME = KernelRegime(
    name="Multi-Phase",
    regime_type=KernelRegimeType.MULTI_PHASE,
    closure_ratio_rule=multiphase_closure_ratio_rule,
    tension_rule=multiphase_tension_rule,
    capacity_rule=multiphase_capacity_rule,
    sao_rule=multiphase_sao_rule,
    stability_rule=multiphase_stability_rule,
    temporal_rule=multiphase_temporal_rule,
    policy_rule=standard_policy_rule
)

REGIMES: Dict[KernelRegimeType, KernelRegime] = {
    KernelRegimeType.BALANCED: BALANCED_REGIME,
    KernelRegimeType.DETERMINISTIC: DETERMINISTIC_REGIME,
    KernelRegimeType.STOCHASTIC: STOCHASTIC_REGIME,
    KernelRegimeType.HIGH_CURVATURE: HIGH_CURVATURE_REGIME,
    KernelRegimeType.ADVERSARIAL: ADVERSARIAL_REGIME,
    KernelRegimeType.MULTI_PHASE: MULTI_PHASE_REGIME
}
