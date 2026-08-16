# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Unit tests for the Invariant Handshake Legacy Hardware Adapter module.
"""

import pytest
import numpy as np

from radial_membrane_ai.adapters.invariant_handshake import (
    LegacyInput,
    AIInput,
    invariant,
    is_stable,
    normalize_to_unity,
    tensor_stress,
    handshake,
    map_ufo_state_to_handshake_inputs,
)
from radial_membrane_ai.exceptions import GovernanceError


def test_invariant_and_stability() -> None:
    # Degenerate geometry (c=0) returns 0.0
    assert invariant(1.0, 2.0, 0.0) == 0.0

    # Right triangle condition (3^2 + 4^2) / 5^2 = 25/25 = 1.0
    assert invariant(3.0, 4.0, 5.0) == pytest.approx(1.0)

    # Stability band checks
    assert is_stable(1.0, tol=0.2) is True
    assert is_stable(1.1, tol=0.2) is True
    assert is_stable(0.9, tol=0.2) is True
    assert is_stable(1.3, tol=0.2) is False
    assert is_stable(0.7, tol=0.2) is False


def test_normalize_to_unity() -> None:
    # Degenerate
    a_deg, b_deg = normalize_to_unity(0.0, 0.0, 0.0)
    assert a_deg == 0.0 and b_deg == 0.0

    # Normalization scales to invariant = 1.0
    a, b = normalize_to_unity(2.0, 4.0, 1.0)
    i_norm = invariant(a, b, 1.0)
    assert i_norm == pytest.approx(1.0)


def test_tensor_stress() -> None:
    ai = AIInput(a=1.0, b=2.0, c=1.0, conversion=0.5)
    assert tensor_stress(ai) == 2.5


def test_handshake_strict_mode_pass_and_fail() -> None:
    # Stable input ((0.6^2 + 0.8^2) / 1.0 = 1.0)
    legacy = LegacyInput(a=0.6, b=0.8, c=1.0)
    ai = AIInput(a=0.6, b=0.8, c=1.0, conversion=0.0)

    status = handshake(legacy, ai, tol=0.2, mode="strict")
    assert status.handshakeAllowed is True
    assert status.legacyStable is True
    assert status.aiStable is True

    # Unstable raw legacy input is contracted onto capacity surface
    legacy_unstable = LegacyInput(a=5.0, b=10.0, c=0.5)
    status_contracted = handshake(legacy_unstable, ai, tol=0.2, mode="strict")
    assert status_contracted.handshakeAllowed is True
    assert status_contracted.details["contracted"] is True

    # High conversion cost causes failure in strict mode
    ai_unstable = AIInput(a=0.6, b=0.8, c=1.0, conversion=5.0)
    with pytest.raises(GovernanceError) as exc_info:
        handshake(legacy, ai_unstable, tol=0.2, mode="strict")
    assert "Invariant Handshake Rejected" in str(exc_info.value)

    # Degenerate geometry raises GovernanceError in strict mode
    legacy_degen = LegacyInput(a=0.0, b=0.0, c=0.0)
    with pytest.raises(GovernanceError) as exc_info:
        handshake(legacy_degen, ai, tol=0.2, mode="strict")
    assert "Invariant Handshake Rejected" in str(exc_info.value)


def test_handshake_soft_mode() -> None:
    legacy = LegacyInput(a=0.6, b=0.8, c=1.0)
    ai = AIInput(a=0.6, b=0.8, c=1.0, conversion=0.0)

    status = handshake(legacy, ai, tol=0.2, mode="soft")
    assert status.handshakeAllowed is True

    # Raw unstable input recovers via soft mode contraction pass
    legacy_unstable = LegacyInput(a=2.0, b=3.0, c=1.0)
    status_soft = handshake(legacy_unstable, ai, tol=0.2, mode="soft")
    assert status_soft.handshakeAllowed is True
    assert status_soft.details["contracted"] is True

    # Degenerate geometry fails soft mode retry and triggers quarantine
    legacy_degen = LegacyInput(a=0.0, b=0.0, c=0.0)
    status_quarantine = handshake(legacy_degen, ai, tol=0.2, mode="soft")
    assert status_quarantine.handshakeAllowed is False
    assert status_quarantine.details["quarantined"] is True


def test_handshake_simulation_mode() -> None:
    legacy = LegacyInput(a=0.6, b=0.8, c=1.0)
    ai_unstable = AIInput(a=0.6, b=0.8, c=1.0, conversion=5.0)

    status = handshake(legacy, ai_unstable, tol=0.2, mode="simulation")
    assert status.handshakeAllowed is False
    assert status.details["legacy_unsafe"] is True


def test_map_ufo_state_to_handshake_inputs() -> None:
    activations = [0.1, 0.2, 0.3, 0.4]
    cost = 0.5
    lyapunov = 0.8
    v_press = 1.2
    conversion = 0.05

    leg, ai = map_ufo_state_to_handshake_inputs(
        string_activations=activations,
        compute_cost=cost,
        lyapunov_energy=lyapunov,
        v_channel_pressure=v_press,
        conversion_cost=conversion,
    )

    expected_norm = float(np.linalg.norm(activations))
    assert leg.a == pytest.approx(expected_norm)
    assert leg.b == 0.5
    assert leg.c == 1.2

    assert ai.a == 0.8
    assert ai.b == pytest.approx(expected_norm)
    assert ai.c == 1.2
    assert ai.conversion == 0.05
