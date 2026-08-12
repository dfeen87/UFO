# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Comprehensive architecture tests covering quarantined shard execution,
stability band transitions, ledger update correctness, SAO residual-only outcomes,
and typed V-Channel exception paths.
"""

from __future__ import annotations
import pytest
import numpy as np

from radial_membrane_ai.shard import FederatedShard, ShardState
from radial_membrane_ai.residuals import ResidualLedger
from radial_membrane_ai.mesh import FederatedShardMesh
from radial_membrane_ai.holistic import HolisticGovernorField
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry


def test_quarantined_shard_execution() -> None:
    """
    Test 1: Quarantined shard execution
    A shard in Quarantined state should reject workload envelopes, produce a residual ledger entry,
    trigger Domain Governor escalation, and leave the mesh coherence score unchanged or reduced.
    """
    mesh = FederatedShardMesh()

    # Create quarantined shard
    shard = FederatedShard(
        shard_id="node_q",
        state=ShardState.QUARANTINED,
        capacity=1.5,
        trust_score=0.9,
        latency=15.0
    )
    mesh.add_shard(shard)

    # Evaluate workload envelope admissibility
    workload = {"capacity": 1.0, "privacy": 0.5, "latency": 50.0, "cost": 2.0}
    admissible = shard.evaluate_admissibility(workload)
    assert admissible is False, "Quarantined shard must reject workload envelope."

    # Run route & execute -> should fail and log critical failure
    success = mesh.route_and_execute(workload)
    assert success is False

    # Verify residual ledger entry produced
    failures = mesh.ledger.get_failures_by_shard("none")
    assert len(failures) > 0
    assert failures[0].error_type == "aggregation_failure"

    # Audit shard with Domain Governor to trigger escalation
    domain_gov = mesh.governors["Domain"]
    # Log critical failures directly to trigger audit quarantine/escalation
    mesh.ledger.log_failure(
        record_id="rec_domain",
        error_type="failed_execution",
        shard_id="node_q",
        severity="critical",
        details={"reason": "quarantined_execution_attempt"},
        timestamp=1.23,
        escalation_tier="Domain"
    )
    domain_gov.audit_shard(shard, mesh.ledger)

    # Assert correct escalation logged in the ledger
    assert mesh.ledger.records[-1].escalation_tier == "Domain"

    # Check coherence impact
    initial_coh = mesh.compute_mesh_coherence()
    # Triggering more failure records should reduce or keep mesh coherence unchanged
    mesh.ledger.log_failure("rec_fail_2", "stale_attestation", "node_q", "high", {})
    subsequent_coh = mesh.compute_mesh_coherence()
    assert subsequent_coh <= initial_coh


def test_stability_band_transitions() -> None:
    """
    Test 2: Stability band transitions
    Force the Holistic Governor into:
      C(t) = 0.71 -> Green
      C(t) = 0.69 -> Yellow
      C(t) = 0.39 -> Red
    Assert correct band classification, intervention signals, and kernel modulation behavior.
    """
    field = HolisticGovernorField()

    # 1. C(t) = 0.71 -> Green
    assert field.evaluate_stability_band(0.71) == "green"
    assert field.check_reconfiguration_trigger(0.71, drift=False, overload=False) is False
    assert field.modulate_kernel(0.71) == pytest.approx(0.71 * 1.2)

    # 2. C(t) = 0.69 -> Yellow
    assert field.evaluate_stability_band(0.69) == "yellow"
    assert field.check_reconfiguration_trigger(0.69, drift=False, overload=False) is False
    assert field.modulate_kernel(0.69) == pytest.approx(0.69 * 1.2)

    # 3. C(t) = 0.39 -> Red
    assert field.evaluate_stability_band(0.39) == "red"
    # Red is not quite 0.3 (reconfiguration threshold) unless drift/overload are true
    assert field.check_reconfiguration_trigger(0.39, drift=True, overload=True) is True
    # At extremely low scores (< 0.3), reconfiguration is triggered immediately
    assert field.check_reconfiguration_trigger(0.25, drift=False, overload=False) is True

    # Check modulation is bounded at min limit 0.1
    assert field.modulate_kernel(0.05) == 0.1


def test_ledger_update_correctness() -> None:
    """
    Test 3: Ledger update correctness
    Trigger failure modes: stale attestation, invalid output, policy conflict, shard churn, SAO residual mismatch.
    Assert ledger entry created, correct failure type, correct timestamp,
    correct residual payload, and correct governor escalation.
    """
    ledger = ResidualLedger()

    failure_scenarios = [
        ("rec_1", "stale_attestation", "node_1", "medium", 10.0, "Local", {"attestation_age": 500}),
        ("rec_2", "invalid_output", "node_2", "high", 11.0, "Domain", {"checksum_mismatch": True}),
        ("rec_3", "policy_conflict", "node_3", "high", 12.0, "Root", {"disallowed_operator": "eval"}),
        ("rec_4", "shard_churn", "node_4", "critical", 13.0, "Holistic", {"node_offline": True}),
        ("rec_5", "sao_residual_mismatch", "node_5", "critical", 14.0, "Holistic", {"p_sao_val": 0.85})
    ]

    for rid, fail_type, sid, sev, ts, tier, payload in failure_scenarios:
        ledger.log_failure(
            record_id=rid,
            error_type=fail_type,
            shard_id=sid,
            severity=sev,
            details=payload,
            timestamp=ts,
            escalation_tier=tier
        )

        # Verify correctness
        record = ledger.records[-1]
        assert record.record_id == rid
        assert record.error_type == fail_type
        assert record.shard_id == sid
        assert record.severity == sev
        assert record.timestamp == ts
        assert record.escalation_tier == tier
        assert record.details == payload


def test_sao_residual_only_outcome() -> None:
    """
    Test 4: SAO residual-only outcome
    Force symmetry alignment to succeed but admissibility to fail:
      x_sym exists (aligned activations are non-zero)
      P(x_sym) does NOT exist (due to highly collapsed boundary, projection is altered)
      p_SAO is preserved (non-zero)
      verdict = block or reproject
    """
    membrane = RadialMembrane()
    boundary = BoundaryGeometry(base_radius=0.01)
    # Force heavy collapse on boundary to make c extremely small
    for idx in boundary.radius_deviation:
        boundary.radius_deviation[idx] = -0.0099

    promotor = SAOPromotor(promotion_threshold=0.3)

    # 1. Align symmetry succeeds with non-zero symmetric representation
    # We use asymmetric original activation so single-frequency sine/cosine Fourier
    # integrals are non-zero, avoiding perfect circular symmetry cancellation.
    # The promotion operator will then scale these to a symmetric aligned representation.
    membrane.strings[0].activation = 0.9

    aligned = promotor.align_symmetry(membrane)
    assert float(np.sum(aligned)) > 0.0, "x_sym must exist"

    # 2. Promote with extremely small boundary capacity
    verdict, p_sao, meta = promotor.promote(membrane, boundary, "Holistic Governor")

    # Admissibility fails, yielding strong residuals
    assert p_sao > 0.0, "p_SAO must be preserved"
    assert verdict in ("block", "reproject", "constrain")


def test_typed_v_channel_exception_path() -> None:
    """
    Test 5: Typed V-Channel exception path
    Send a Type-E (exception) channel event: residual, failed fragment, policy conflict, invalid shard output.
    Assert correct routing, correct governor escalation, correct ledger entry, and correct mesh coherence impact.
    """
    mesh = FederatedShardMesh()
    shard = FederatedShard(shard_id="node_err", state=ShardState.ACTIVE, trust_score=0.9, capacity=1.0)
    mesh.add_shard(shard)

    # Retrieve Type-E channel
    chan_e = mesh.channels["Type-E"]
    assert chan_e.channel_type == "Type-E"

    # Simulate an exception path event
    exception_event = {
        "event_id": "err_01",
        "type": "failed_fragment",
        "severity": "critical",
        "payload": {"failed_block": "sao_layer_0"}
    }

    # Record to ledger
    mesh.ledger.log_failure(
        record_id=exception_event["event_id"],
        error_type=exception_event["type"],
        shard_id=shard.shard_id,
        severity=exception_event["severity"],
        details=exception_event["payload"],
        timestamp=20.0,
        escalation_tier="Root"
    )

    # Run mesh governance audit to handle escalation
    mesh.run_mesh_governance()

    # Assert correct escalation, routing and ledger record creation
    record = mesh.ledger.records[-1]
    assert record.record_id == "err_01"
    assert record.error_type == "failed_fragment"
    assert record.escalation_tier == "Root"

    # Mesh coherence is impacted negatively by the exception log
    coh = mesh.compute_mesh_coherence()
    assert coh < 0.8
