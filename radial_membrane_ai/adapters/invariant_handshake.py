# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Invariant Handshake Adapter Module.

Implements the geometric stability framework detailed in 'The Invariant Handshake:
Stabilizing AI-Legacy Interoperability Through Geometric Normalization' by Don M. Feeney Jr.

Evaluates closure condition:
    i = (a^2 + b^2) / c^2

where:
    a = scalar or tensor stress contribution
    b = secondary stress contribution
    c = shared channel capacity (structural bound)

When both legacy binary execution and AI tensor flow satisfy i approx 1 under
shared capacity c, cross-era communication is admissible and stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

from radial_membrane_ai.exceptions import GovernanceError


@dataclass
class LegacyInput:
    """
    Legacy compute input (scalar / discrete binary stress terms).

    Geometric analogue: triangular invariant regime.
    """

    a: float  # Legacy signal strength / scalar activation
    b: float  # Legacy compute cost / effort
    c: float  # Shared channel capacity bound


@dataclass
class AIInput:
    """
    AI compute input (tensor / high-bandwidth continuous activation).

    Geometric analogue: circular brim regime.
    """

    a: float  # AI Lyapunov energy / stability measure
    b: float  # Tensor activation norm
    c: float  # Shared channel capacity bound
    conversion: float = 0.0  # Cost of tensor -> binary representation translation


@dataclass
class HandshakeStatus:
    """
    Brim-level diagnostic output for the Invariant Handshake.
    """

    iLegacy: float
    iAI: float
    legacyStable: bool
    aiStable: bool
    handshakeAllowed: bool
    mode: str = "strict"
    details: Dict[str, Any] = field(default_factory=dict)


def tensor_stress(ai_input: AIInput) -> float:
    """
    Calculates total tensor-side stress including representation conversion cost.
    """
    return ai_input.b + ai_input.conversion


def invariant(a: float, b: float, c: float) -> float:
    """
    Executable form of the geometric invariant:
        i = (a^2 + b^2) / c^2

    If c == 0, degenerate geometry collapses to 0.0.
    """
    den = c * c
    if den == 0.0:
        return 0.0
    return (a * a + b * b) / den


def is_stable(i: float, tol: float = 0.2) -> bool:
    """
    Checks if invariant 'i' lies within stability band [1 - tol, 1 + tol].
    """
    lower = 1.0 - tol
    upper = 1.0 + tol
    return lower < i < upper


def normalize_to_unity(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Contraction operator: Delta AG -> Delta v collapse toward the invariant surface.
    Scales (a, b) such that (a^2 + b^2) / c^2 -> 1.0 if i > 0.
    """
    i_val = invariant(a, b, c)
    if i_val == 0.0:
        return a, b
    scale = 1.0 / math.sqrt(i_val)
    return a * scale, b * scale


def handshake(
    legacy: LegacyInput,
    ai: AIInput,
    tol: float = 0.2,
    mode: str = "strict",
) -> HandshakeStatus:
    """
    Evaluates circle-triangle admissibility handshake between Legacy system and AI tensor flow.

    Modes:
        - "strict": Raises GovernanceError if handshakeAllowed is False.
        - "soft": Normalizes inputs and retries handshake once before flagging quarantine.
        - "simulation": Logs status and permits continuation marked as legacy-unsafe.
    """
    # 1. Raw Invariant Evaluation
    i_legacy_raw = invariant(legacy.a, legacy.b, legacy.c)
    i_ai_raw = invariant(ai.a, tensor_stress(ai), ai.c)

    legacy_raw_stable = is_stable(i_legacy_raw, tol)
    ai_raw_stable = is_stable(i_ai_raw, tol)

    details: Dict[str, Any] = {
        "legacy_raw": {
            "a": legacy.a, "b": legacy.b, "c": legacy.c, "i": i_legacy_raw, "stable": legacy_raw_stable
        },
        "ai_raw": {
            "a": ai.a, "b": ai.b, "c": ai.c, "conversion": ai.conversion, "i": i_ai_raw, "stable": ai_raw_stable
        },
        "contracted": False,
        "legacy_normalized": {"a": legacy.a, "b": legacy.b},
        "ai_normalized": {"a": ai.a, "b": ai.b},
        "quarantined": False,
        "legacy_unsafe": False,
    }

    # If raw inputs are already stable, handshake is allowed directly
    if legacy_raw_stable and ai_raw_stable:
        return HandshakeStatus(
            iLegacy=i_legacy_raw,
            iAI=i_ai_raw,
            legacyStable=True,
            aiStable=True,
            handshakeAllowed=True,
            mode=mode,
            details=details,
        )

    # 2. Geometric Contraction Pass (Delta AG -> Delta v)
    details["contracted"] = True
    l_a, l_b = normalize_to_unity(legacy.a, legacy.b, legacy.c)
    a_a, a_b = normalize_to_unity(ai.a, ai.b, ai.c)

    details["legacy_normalized"] = {"a": l_a, "b": l_b}
    details["ai_normalized"] = {"a": a_a, "b": a_b}

    i_legacy_norm = invariant(l_a, l_b, legacy.c)
    i_ai_norm = invariant(a_a, a_b + ai.conversion, ai.c)

    legacy_norm_stable = is_stable(i_legacy_norm, tol)
    ai_norm_stable = is_stable(i_ai_norm, tol)
    allowed = legacy_norm_stable and ai_norm_stable

    if not allowed:
        if mode == "strict":
            msg = (
                f"Invariant Handshake Rejected [mode={mode}]: "
                f"Legacy invariant={i_legacy_norm:.4f} (raw={i_legacy_raw:.4f}, stable={legacy_norm_stable}), "
                f"AI invariant={i_ai_norm:.4f} (raw={i_ai_raw:.4f}, stable={ai_norm_stable}), tol={tol}."
            )
            raise GovernanceError(msg)
        elif mode == "soft":
            details["quarantined"] = True
        elif mode == "simulation":
            details["legacy_unsafe"] = True

    return HandshakeStatus(
        iLegacy=i_legacy_norm,
        iAI=i_ai_norm,
        legacyStable=legacy_norm_stable,
        aiStable=ai_norm_stable,
        handshakeAllowed=allowed,
        mode=mode,
        details=details,
    )


def map_ufo_state_to_handshake_inputs(
    string_activations: Union[Sequence[float], np.ndarray],
    compute_cost: float,
    lyapunov_energy: float,
    v_channel_pressure: float,
    conversion_cost: float = 0.0,
) -> Tuple[LegacyInput, AIInput]:
    """
    Maps UFO internal dynamic state variables into LegacyInput and AIInput.

    Canonical Mapping:
        LegacyInput (scalar):
            a = string activation L2 norm (signal magnitude)
            b = compute cost
            c = v_channel_pressure + 1e-9 (shared channel capacity)

        AIInput (tensor):
            a = lyapunov energy
            b = string activation L2 norm (tensor norm)
            c = v_channel_pressure + 1e-9 (shared channel capacity)
            conversion = conversion_cost
    """
    activations_arr = np.array(string_activations, dtype=float)
    act_norm = float(np.linalg.norm(activations_arr))

    leg_a = max(0.001, act_norm)
    leg_b = max(0.001, float(compute_cost))

    ai_a = max(0.001, float(lyapunov_energy))
    ai_b = max(0.001, act_norm)

    c_capacity = max(1e-9, float(v_channel_pressure))

    legacy_in = LegacyInput(a=leg_a, b=leg_b, c=c_capacity)
    ai_in = AIInput(a=ai_a, b=ai_b, c=c_capacity, conversion=float(conversion_cost))

    return legacy_in, ai_in
