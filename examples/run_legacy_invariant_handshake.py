#!/usr/bin/env python3
# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Example: Running the Invariant Handshake Adapter for Legacy Hardware Interoperability.

This example demonstrates:
1. Evaluating circle-triangle admissibility between Legacy binary compute and AI tensor flow.
2. Normalization via Delta AG -> Delta v contraction.
3. Mapping UFO internal state (Lyapunov energy, activations, compute cost, channel pressure) into (a, b, c).
4. Running strict, soft, and simulation governance modes.
"""

import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from radial_membrane_ai.adapters import (  # noqa: E402
    AIInput,
    LegacyInput,
    handshake,
    map_ufo_state_to_handshake_inputs,
)
from radial_membrane_ai.admissibility import AdmissibilityGate  # noqa: E402
from radial_membrane_ai.exceptions import GovernanceError  # noqa: E402
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("🛸 UFO ARCHITECTURE — INVARIANT HANDSHAKE LEGACY ADAPTER DEMO 🛸")
    print("=" * 70)

    # 1. Canonical Handshake Test
    print("\n1. Direct Invariant Handshake Evaluation (a=0.4, b=0.9, c=1.0)")
    legacy_in = LegacyInput(a=0.4, b=0.9, c=1.0)
    ai_in = AIInput(a=0.4, b=0.9, c=1.0, conversion=0.0)

    status = handshake(legacy_in, ai_in, tol=0.2, mode="strict")
    print(f"   Legacy Invariant (i): {status.iLegacy:.4f} (Stable: {status.legacyStable})")
    print(f"   AI Invariant (i):     {status.iAI:.4f} (Stable: {status.aiStable})")
    print(f"   Handshake Allowed:    {status.handshakeAllowed}")

    # 2. Mapping UFO Engine State
    print("\n2. Mapping UFO Runtime State to Invariant Handshake Inputs")
    activations = [0.2, 0.5, 0.8, 0.3, 0.1, 0.4, 0.6, 0.2, 0.9, 0.3, 0.1, 0.2]
    compute_cost = 0.45
    lyapunov_energy = 0.95
    v_channel_pressure = 1.2
    conversion_cost = 0.05

    leg_mapped, ai_mapped = map_ufo_state_to_handshake_inputs(
        string_activations=activations,
        compute_cost=compute_cost,
        lyapunov_energy=lyapunov_energy,
        v_channel_pressure=v_channel_pressure,
        conversion_cost=conversion_cost,
    )

    print(f"   Mapped LegacyInput: {leg_mapped}")
    print(f"   Mapped AIInput:     {ai_mapped}")

    gate = AdmissibilityGate(tol=0.2, legacy_mode="strict")
    hs_result = gate.check_legacy_handshake(
        string_activations=activations,
        compute_cost=compute_cost,
        lyapunov_energy=lyapunov_energy,
        v_channel_pressure=v_channel_pressure,
        conversion_cost=conversion_cost,
    )
    print(f"   Gate Check Result Allowed: {hs_result.handshakeAllowed}")

    # 3. Engine Integration Check with Legacy Handshake Enabled
    print("\n3. Governed Single-Agent Simulation with Legacy Hardware Handshake")
    engine = SingleAgentEngine(
        seed=0,
        enable_legacy_handshake=True,
        legacy_handshake_mode="soft",
    )

    print("   Running tick with soft enforcement mode...")
    band = engine.tick(
        task_value=0.8,
        excitation=np.ones(12) * 0.1,
    )
    print(f"   Tick complete. Resulting stability band: {band}")

    # 4. Strict Mode Unstable Rejection
    print("\n4. Testing Strict Mode Governance Enforcement under Excessive Conversion Cost")
    unstable_legacy = LegacyInput(a=0.4, b=0.9, c=1.0)
    unstable_ai = AIInput(a=0.4, b=0.9, c=1.0, conversion=5.0)  # Excessive conversion cost

    try:
        handshake(unstable_legacy, unstable_ai, tol=0.2, mode="strict")
        print("   FAILED: Handshake should have raised GovernanceError!")
    except GovernanceError as e:
        print(f"   SUCCESS: GovernanceError caught as expected:\n   {e}")

    print("\n" + "=" * 70)
    print("🛸 INVARIANT HANDSHAKE DEMONSTRATION COMPLETE 🛸")
    print("=" * 70)


if __name__ == "__main__":
    main()
