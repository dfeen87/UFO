"""BEDROCK III falsification tests for final paper-supported invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radial_membrane_ai.admissibility import apply_lyapunov_dissipation
from radial_membrane_ai.adapters.invariant_handshake import AIInput, tensor_stress
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.channels import (
    activation_weight, channel_coherence, inverse_cost_weight,
    max_field_activation, phase_alignment, update_radius_along_channel,
)
from radial_membrane_ai.exceptions import GeometryValidationError, ValidationError
from radial_membrane_ai.facet import FacetVector, TensionState
from radial_membrane_ai.governor import Governor
from radial_membrane_ai.holistic import HolisticGovernorField, compute_holistic_field
from radial_membrane_ai.lde import lde_encode, lde_reconstruct
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.mesh import FederatedShardMesh
from radial_membrane_ai.routing import FederatedRoutingChannel
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.shard import FederatedShard, ShardState


EXTREME = 10**10000


@pytest.mark.parametrize(
    "capacity", [EXTREME, -EXTREME, math.nan, math.inf, -math.inf, True, "1"],
    ids=["huge-positive", "huge-negative", "nan", "positive-inf", "negative-inf", "bool", "string"],
)
def test_unrepresentable_capacity_survives_projection_only_as_untrusted_evidence(capacity: object) -> None:
    channel = FederatedRoutingChannel("hostile-capacity")
    shard = FederatedShard("ordinary", capacity=2.0)
    projected = channel.project_into_shard_geometry({"capacity": capacity}, shard)
    assert projected["capacity"] is capacity or projected["capacity"] == capacity
    assert shard.evaluate_admissibility(projected) is False
    assert channel.route_workload(projected, [shard]) is None


def test_unrepresentable_shard_capacity_cannot_clamp_hostile_evidence_into_admission() -> None:
    channel = FederatedRoutingChannel("hostile-shard")
    shard = FederatedShard("hostile", capacity=EXTREME)
    projected = channel.project_into_shard_geometry({"capacity": 1.0}, shard)
    assert projected["capacity"] == 1.0
    assert shard.evaluate_admissibility(projected) is False
    assert channel.route_workload(projected, [shard]) is None


def test_unrepresentable_integer_does_not_crash_adjacent_handshake_boundary() -> None:
    assert tensor_stress(AIInput(a=1.0, b=EXTREME, c=1.0, conversion=EXTREME)) == 0.0


@pytest.mark.parametrize("capacity", [2, 2.0])
def test_finite_capacity_projection_remains_composable(capacity: float) -> None:
    channel = FederatedRoutingChannel("finite")
    shard = FederatedShard("finite", capacity=1.5)
    projected = channel.project_into_shard_geometry({"capacity": capacity}, shard)
    assert projected["capacity"] == 1.5
    assert shard.evaluate_admissibility(projected)
    assert channel.route_workload(projected, [shard]) is shard


@pytest.mark.parametrize(
    ("layer", "ticks", "tension", "expected"),
    [("semantic memory", 5, 0.1, "constrain"), ("identity core", 5, 0.1, "block")],
)
def test_temporal_downgrade_never_commits_sao_evidence(
    layer: str, ticks: int, tension: float, expected: str
) -> None:
    membrane = RadialMembrane()
    for string in membrane.strings:
        string.activation = 0.8
    for _ in range(ticks):
        membrane.temporal_state.update_tick_history(0.0, tension, 0.1)
    promotor = SAOPromotor(0.5)
    for _ in range(3):
        verdict, _, metadata = promotor.promote(membrane, BoundaryGeometry(10.0), layer)
        assert verdict == expected
        assert metadata["temporal_gate_verdict"] == expected
    assert promotor.promotion_ledger == []


def test_final_authorized_sao_verdict_commits_exactly_once() -> None:
    membrane = RadialMembrane()
    for string in membrane.strings:
        string.activation = 0.8
    for _ in range(20):
        membrane.temporal_state.update_tick_history(0.0, 0.1, 0.1)
    promotor = SAOPromotor(0.5)
    verdict, p_sao, metadata = promotor.promote(membrane, BoundaryGeometry(10.0), "semantic memory")
    assert verdict == metadata["temporal_gate_verdict"] == "ascend"
    assert promotor.promotion_ledger == [{
        "layer": "semantic memory", "activation": metadata["collapse_activation"],
        "p_sao": p_sao, "verdict": verdict,
    }]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_radius": math.nan}, {"base_radius": True}, {"base_radius": 0.0},
        {"basis_sigma": math.inf}, {"basis_sigma": 0.0},
        {"min_radius_ratio": 0.0}, {"max_radius_ratio": -1.0},
        {"min_radius_ratio": 2.0, "max_radius_ratio": 1.0},
    ],
)
def test_boundary_constructor_rejects_impossible_geometry(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BoundaryGeometry(**kwargs)  # type: ignore[arg-type]


def test_boundary_update_is_atomic_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    membrane = RadialMembrane()
    for string in membrane.strings:
        string.activation = 1.0
        string.radius = 1.0
    boundary = BoundaryGeometry(min_radius_ratio=0.5, max_radius_ratio=1.5)
    boundary.update_boundary(membrane, task_value=1e200, learning_rate=1e-200)
    assert all(0.5 <= boundary.get_radius(s.theta) <= 1.5 for s in membrane.strings)
    assert all(math.isfinite(boundary.tangent(s.theta)) and math.isfinite(boundary.curvature(s.theta))
               for s in membrane.strings)

    prior = dict(boundary.radius_deviation)
    calls = 0

    def fail_late(*args: object, **kwargs: object) -> float:
        nonlocal calls
        calls += 1
        if calls > 100:
            raise ValidationError("invalid coherence sample")
        return 0.5

    monkeypatch.setattr("radial_membrane_ai.boundary.channel_coherence", fail_late)
    with pytest.raises(ValidationError):
        boundary.update_boundary(membrane, task_value=0.5)
    assert boundary.radius_deviation == prior


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1"])
def test_channel_public_helpers_reject_malformed_numeric_evidence(bad: object) -> None:
    membrane = RadialMembrane()
    with pytest.raises(ValueError):
        phase_alignment(bad, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        activation_weight(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        inverse_cost_weight(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        channel_coherence(membrane, 1, 2, epsilon=bad)  # type: ignore[arg-type]


def test_v_channel_bounds_monotonicity_and_custom_mapping_contract() -> None:
    membrane = RadialMembrane()
    source, target = membrane.strings[:2]
    source.radius = 2.0
    target.activation = 0.5
    assert 0.0 <= phase_alignment(source.theta, target.theta) <= 1.0
    assert inverse_cost_weight(0.0) >= inverse_cost_weight(1.0) >= inverse_cost_weight(10.0)
    assert update_radius_along_channel(source, target, 0.25) <= 0.25
    with pytest.raises(ValueError):
        update_radius_along_channel(source, target, 1.0, h_func=lambda _: math.nan)
    with pytest.raises(ValueError):
        max_field_activation(membrane, samples=0)


@pytest.mark.parametrize("state", [ShardState.QUARANTINED, ShardState.REVOKED, ShardState.EXPIRED])
def test_terminal_shard_cannot_add_positive_mesh_coherence_evidence(state: ShardState) -> None:
    mesh = FederatedShardMesh()
    mesh.add_shard(FederatedShard("eligible", state=ShardState.ACTIVE, trust_score=0.6,
                                  quality_score=0.6, cost_factor=2.0, latency=50.0))
    baseline = mesh.compute_mesh_coherence()
    mesh.add_shard(FederatedShard("stale", state=state, trust_score=1.0, policy_compliance=1.0,
                                  quality_score=1.0, cost_factor=0.0, latency=0.0))
    assert mesh.compute_mesh_coherence() <= baseline


def test_holistic_metric_is_validated_but_remains_diagnostic() -> None:
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()
    facets = [FacetVector(str(i), TensionState.ADMISSIBLE, 0.5, 1.0, 0.0, 1.0)
              for i in range(12)]
    before = membrane.get_activation_vector().copy()
    observed = compute_holistic_field(membrane, boundary, facets, np.eye(12))
    assert math.isfinite(observed)
    assert np.array_equal(before, membrane.get_activation_vector())
    field = HolisticGovernorField()
    assert field.get_coherence_score(-1e300) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        compute_holistic_field(membrane, boundary, facets[:-1], np.eye(12))


def test_restricted_lyapunov_style_dissipation_does_not_increase_energy() -> None:
    membrane = RadialMembrane()
    governor = Governor()
    for string in membrane.strings:
        string.activation = 0.9
        string.radius = 2.0
        string.cost = 1.0
        string.tension = 1.0
    before = governor.compute_lyapunov_energy(membrane)
    apply_lyapunov_dissipation(membrane, governor, target_energy=0.0, dissipation_rate=0.25)
    after = governor.compute_lyapunov_energy(membrane)
    assert after <= before
    assert all(string.activation == pytest.approx(0.675) for string in membrane.strings)
    with pytest.raises(ValidationError):
        apply_lyapunov_dissipation(membrane, governor, target_energy=0.0, dissipation_rate=math.nan)


@pytest.mark.parametrize(
    "text",
    ["naïve Καλημέρα 漢字", "emoji: 🛸🚀", "e\u0301 != é", "a\r\nb\rc\nd", "\t  \t", "!?…—"],
)
def test_lde_full_mode_preserves_raw_unicode_and_line_boundaries(text: str) -> None:
    assert lde_reconstruct(lde_encode(text)) == text
