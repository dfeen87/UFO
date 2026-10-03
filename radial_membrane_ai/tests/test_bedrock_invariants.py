"""Regression tests for BEDROCK numerical and trust-boundary invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.governor import GovernorConfig
from radial_membrane_ai.membrane import BehavioralString, RadialMembrane
from radial_membrane_ai.shard import FederatedShard
from ufo_cli import main


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_behavioral_string_rejects_non_finite_state(invalid: float) -> None:
    with pytest.raises(ValidationError):
        BehavioralString("invalid", 1, 0.0, invalid, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_membrane_activation_update_is_atomic_for_invalid_delta(invalid: float) -> None:
    membrane = RadialMembrane()
    before = membrane.get_activation_vector().copy()
    delta = np.full(12, 0.1)
    delta[6] = invalid

    with pytest.raises(ValidationError):
        membrane.update_activation(delta)

    np.testing.assert_array_equal(membrane.get_activation_vector(), before)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_governor_config_rejects_non_finite_coefficients(invalid: float) -> None:
    with pytest.raises(ValidationError):
        GovernorConfig(w_d=invalid)


@pytest.mark.parametrize(
    "field",
    ["capacity", "privacy_level", "latency", "cost_factor", "trust_score", "policy_compliance"],
)
def test_shard_with_non_finite_evidence_fails_closed(field: str) -> None:
    shard = FederatedShard(shard_id="untrusted")
    setattr(shard, field, math.nan)

    assert shard.evaluate_admissibility({}) is False


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "not-a-number", None])
def test_shard_rejects_malformed_workload_evidence(value: object) -> None:
    shard = FederatedShard(shard_id="trusted")

    assert shard.evaluate_admissibility({"capacity": value}) is False


def test_shard_rejects_unknown_or_negative_workload_requirements() -> None:
    shard = FederatedShard(shard_id="trusted")

    assert shard.evaluate_admissibility({"unknown": 1.0}) is False
    assert shard.evaluate_admissibility({"capacity": -0.1}) is False


def test_shard_rejects_out_of_domain_internal_evidence() -> None:
    shard = FederatedShard(shard_id="invalid", trust_score=1.1)

    assert shard.evaluate_admissibility({}) is False


def test_cli_reports_authoritative_package_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["ufo", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "UFO 4.0.0"
