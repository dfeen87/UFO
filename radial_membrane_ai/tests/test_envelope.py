# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Unit and integration tests for Bounded Compute Envelope (Brim).
"""

from __future__ import annotations
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.envelope import BrimEnvelope, FalsificationStack, ClaimStateLedger


def test_brim_envelope_evaluation() -> None:
    membrane = RadialMembrane()
    boundary = BoundaryGeometry(base_radius=0.05)

    stack = FalsificationStack()
    ledger = ClaimStateLedger()
    envelope = BrimEnvelope(energy_threshold=0.1, falsification_stack=stack, claim_ledger=ledger)

    # 1. Test schema freeze
    envelope.freeze_schema("schema1", {"param1": 42})
    assert envelope.frozen_schemas["schema1"]["param1"] == 42

    # 2. Test default evaluation (admit/rest)
    # Average activation is 0.0 -> returns 'rest'
    verdict, meta = envelope.evaluate_envelope(membrane, boundary)
    assert verdict == "rest"

    # Excite membrane slightly to get non-rest
    for s in membrane.strings:
        s.activation = 0.2

    verdict, meta = envelope.evaluate_envelope(membrane, boundary)
    assert verdict == "admit"
    assert meta["brim_energy"] >= 0.0

    # Excite to extreme level asymmetric to trigger constraint violations or blocking
    # We set only index 0 to high activation, making the field highly asymmetric
    membrane.strings[0].activation = 1.0
    membrane.strings[0].radius = 2.0

    # Force heavy envelope violation
    boundary.base_radius = 0.01
    for k in boundary.radius_deviation:
        boundary.radius_deviation[k] = -0.009

    verdict, meta = envelope.evaluate_envelope(membrane, boundary)
    assert verdict in ("block", "re-project", "constrain")
    assert stack.is_violated() is True

    # Pop violation
    popped = stack.pop_violation()
    assert popped is not None
    assert popped["type"] == "boundary_violation"
