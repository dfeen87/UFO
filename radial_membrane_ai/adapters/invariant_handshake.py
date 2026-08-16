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
    l_a, l_b = normalize_to_unity(legacy.a, legacy.b, legacy.c)
    a_a, a_b = normalize_to_unity(ai.a, ai.b, ai.c)

    i_legacy = invariant(l_a, l_b, legacy.c)
    i_ai = invariant(a_a, a_b + ai.conversion, ai.c)

    legacy_stable = is_stable(i_legacy, tol)
    ai_stable = is_stable(i_ai, tol)
    allowed = legacy_stable and ai_stable

    details: Dict[str, Any] = {
        "legacy_raw": {"a": legacy.a, "b": legacy.b, "c": legacy.c},
        "ai_raw": {"a": ai.a, "b": ai.b, "c": ai.c, "conversion": ai.conversion},
        "legacy_normalized": {"a": l_a, "b": l_b},
        "ai_normalized": {"a": a_a, "b": a_b},
        "retry_attempted": False,
        "quarantined": False,
        "legacy_unsafe": False,
    }

    if not allowed:
        if mode == "strict":
            msg = (
                f"Invariant Handshake Rejected [mode={mode}]: "
                f"Legacy invariant={i_legacy:.4f} (stable={legacy_stable}), "
                f"AI invariant={i_ai:.4f} (stable={ai_stable}), tol={tol}."
            )
            raise GovernanceError(msg)
        elif mode == "soft":
            details["retry_attempted"] = True
            # Secondary contraction pass: contract total strain (including conversion) onto capacity manifold
            l_a2, l_b2 = normalize_to_unity(l_a, l_b, legacy.c)
            a_a2, a_b2 = normalize_to_unity(a_a, a_b + ai.conversion, ai.c)

            i_legacy2 = invariant(l_a2, l_b2, legacy.c)
            i_ai2 = invariant(a_a2, a_b2, ai.c)
            retry_allowed = is_stable(i_legacy2, tol) and is_stable(i_ai2, tol)

            if retry_allowed:
                allowed = True
                i_legacy, i_ai = i_legacy2, i_ai2
                legacy_stable, ai_stable = True, True
            else:
                details["quarantined"] = True
        elif mode == "simulation":
            details["legacy_unsafe"] = True

    return HandshakeStatus(
        iLegacy=i_legacy,
        iAI=i_ai,
        legacyStable=legacy_stable,
        aiStable=ai_stable,
        handshakeAllowed=allowed,
        mode=mode,
        details=details,
    )


def map_ufo_state_to_handshake_inputs(
    string_activations: Union[Sequence[float], np.ndarray],
    compute_cost: float,
    lyapunov_energy: float,
    v_channel_pressure: float,
    conversion_cost: float = 0.05,
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

    # Shared capacity bound, ensuring non-zero denominator
    c_capacity = max(1e-9, float(v_channel_pressure))

    legacy_in = LegacyInput(
        a=max(0.001, act_norm),
        b=max(0.001, float(compute_cost)),
        c=c_capacity,
    )

    ai_in = AIInput(
        a=max(0.001, float(lyapunov_energy)),
        b=max(0.001, act_norm),
        c=c_capacity,
        conversion=float(conversion_cost),
    )

    return legacy_in, ai_in
