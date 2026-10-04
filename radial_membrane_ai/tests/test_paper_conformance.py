"""Falsification-oriented regressions for published research contracts."""

from __future__ import annotations

import math

import pytest

from radial_membrane_ai.lde import LDEConfig, lde_encode, lde_reconstruct
from radial_membrane_ai.exceptions import ReconstructionError
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.routing import FederatedRoutingChannel
from radial_membrane_ai.shard import FederatedShard, ShardState


EXPECTED_FACET_BASIS = (
    "depth", "precision", "technical_detail", "structural_rigor",
    "context_sensitivity", "transparency", "initiative", "exploration",
    "creativity", "tone", "emotional_warmth", "conciseness",
)


def test_twelve_facet_basis_has_stable_identity_phase_and_order() -> None:
    membrane = RadialMembrane()
    assert tuple(string.name for string in membrane.strings) == EXPECTED_FACET_BASIS
    assert tuple(string.index for string in membrane.strings) == tuple(range(1, 13))
    for index, string in enumerate(membrane.strings, 1):
        assert string.theta == pytest.approx(2.0 * math.pi * index / 12.0)


def test_projection_admission_and_routing_compose_without_trusting_marker() -> None:
    channel = FederatedRoutingChannel("paper-contract")
    shard = FederatedShard("valid", capacity=2.0)
    workload = {"capacity": 1.5, "privacy": 0.5, "latency": 100.0, "cost": 5.0}

    projected = channel.project_into_shard_geometry(workload, shard)

    assert projected == {**workload, "routed": True}
    assert shard.evaluate_admissibility(projected) is True
    assert channel.route_workload(projected, [shard]) is shard


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "1", True])
def test_projected_malformed_numeric_evidence_fails_closed(bad: object) -> None:
    channel = FederatedRoutingChannel("paper-contract")
    shard = FederatedShard("valid")
    projected = channel.project_into_shard_geometry({"capacity": bad}, shard)
    assert shard.evaluate_admissibility(projected) is False
    assert channel.route_workload(projected, [shard]) is None


@pytest.mark.parametrize("metadata", [False, 1, "true", None, {}])
def test_forged_or_malformed_routing_metadata_cannot_improve_admission(metadata: object) -> None:
    shard = FederatedShard("too-small", capacity=0.25)
    evidence = {"capacity": 0.5, "privacy": 0.5, "latency": 100.0, "cost": 5.0}
    forged = {**evidence, "routed": metadata}
    assert shard.evaluate_admissibility(forged) is False


def test_unknown_projection_field_remains_fail_closed() -> None:
    channel = FederatedRoutingChannel("paper-contract")
    shard = FederatedShard("valid")
    projected = channel.project_into_shard_geometry({"unknown": 1.0}, shard)
    assert shard.evaluate_admissibility(projected) is False


@pytest.mark.parametrize(
    "state", [ShardState.QUARANTINED, ShardState.REVOKED, ShardState.EXPIRED]
)
def test_terminal_governance_overrides_valid_projected_evidence(state: ShardState) -> None:
    channel = FederatedRoutingChannel("paper-contract")
    shard = FederatedShard("unavailable", state=state)
    projected = channel.project_into_shard_geometry({"capacity": 0.5}, shard)
    assert shard.evaluate_admissibility(projected) is False
    assert channel.route_workload(projected, [shard]) is None


@pytest.mark.parametrize(
    "text",
    [
        "", "aaaaAA", "Mixed CASE!?", "two  spaces", "tabs\tand\nlines",
        "repeat::repeat::repeat", "naïve café U0001f680", "...___111___...",
    ],
)
def test_lde_full_reconstruction_round_trips_adversarial_text(text: str) -> None:
    assert lde_reconstruct(lde_encode(text)) == text


def test_lde_compressed_geometry_is_explicitly_not_lossless() -> None:
    state = lde_encode("Geometry != identity", LDEConfig(rho="compressed"))
    with pytest.raises(ReconstructionError, match="not lossless"):
        lde_reconstruct(state)


def test_lde_geometry_is_dimensionally_consistent_and_finite() -> None:
    state = lde_encode("adversarial repetition!!! aaa BBB\n")
    geometry = state.boundary
    assert len(geometry.radius_map) == len(geometry.tangent_map) == len(geometry.curvature_map) == 100
    assert all(math.isfinite(value) for values in (
        geometry.radius_map, geometry.tangent_map, geometry.curvature_map,
    ) for value in values)
    assert math.isfinite(geometry.asymmetry)
    assert len(state.coherence_matrix) == 26 * 26
