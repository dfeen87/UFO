"""
Extra test module to ensure 100% test coverage across all lines.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from unittest.mock import patch

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.projection import project_to_admissible
from radial_membrane_ai.governor import Governor
from radial_membrane_ai.admissibility import (
    global_closure_aggregation,
    local_closure_test_at_angle
)
from radial_membrane_ai.envelope import BrimEnvelope, ClaimStateLedger, FalsificationStack
from radial_membrane_ai.mesh import FederatedShardMesh, MeshGovernor, ShardState, FederatedShard
from radial_membrane_ai.routing import FederatedRoutingChannel
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.simulation import RainbowSimulation


def test_final_coverage_gaps() -> None:
    """
    Cover the remaining missing lines from the coverage report.
    """
    # 1. boundary.py: update_boundary clamps, i_ratio > 1.0, asymmetry, tangent
    membrane = RadialMembrane()
    membrane.strings[0].activation = 1.0
    membrane.strings[0].radius = 1.0
    boundary = BoundaryGeometry()

    # Trigger i_ratio > 1.0 in update_boundary (set all deviations to make capacity small)
    boundary.radius_deviation = {i: -0.9 for i in range(1, 13)}
    boundary.update_boundary(membrane, task_value=0.9)

    # Huge stretch to hit max_dev limit
    boundary.update_boundary(membrane, task_value=1000.0, learning_rate=1.0)
    # Huge collapse to hit min_dev limit
    membrane.strings[0].cost = 1000.0
    boundary.update_boundary(membrane, task_value=0.0, learning_rate=1.0)

    # get_radius total_weight <= 0.0
    boundary.basis_sigma = 1e-12
    boundary.get_radius(0.5)
    boundary.basis_sigma = math.pi / 6

    # asymmetry and tangent
    assert boundary.asymmetry(1.0) == (boundary.get_radius(1.0) - boundary.get_radius(1.0 + math.pi))
    assert boundary.tangent(1.0, delta=0.01) == ((boundary.get_radius(1.01) - boundary.get_radius(0.99)) / 0.02)

    # 2. governor.py: extra clamp logic, is_stable cases
    gov = Governor()
    # Trigger c_i > task_value and task_value < 0.3 clamp
    membrane_gov = RadialMembrane()
    for s in membrane_gov.strings:
        s.radius = 10.0
    gov.update_membrane(membrane_gov, task_value=0.1)

    # is_stable when energy history is small
    gov.energy_history = [1.0]
    assert gov.is_stable() is True
    gov.energy_history = [1.0, 2.0]
    assert gov.is_stable(window=1) is True

    # is_stable trending upward (returns False)
    gov.energy_history = [1.0, 2.0, 3.0]
    assert gov.is_stable() is False

    # 3. membrane.py: quadrant average with empty target strings
    membrane_empty = RadialMembrane()
    membrane_empty._quadrant_map["analytical"] = {"non_existent"}
    assert membrane_empty.get_quadrant_activation("analytical") == 0.0

    # get_membrane_field
    field_val = membrane_empty.get_membrane_field(0.0, 1.0)
    assert field_val >= 0.0

    # 4. projection.py: weighted with None weights and edge cases
    # Weighted Projection with None weights
    a_p, b_p = project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights=None)
    assert math.isclose(a_p**2 + b_p**2, 4.0)

    # Bisection search precision
    assert len(project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights={"w_a": 1.0, "w_b": 1.0})) == 2

    # Fallback to Euclidean
    a_f, b_f = project_to_admissible(3.0, 4.0, 2.0, metric="invalid_metric")
    assert math.isclose(a_f**2 + b_f**2, 4.0)

    # Force line 112 in projection.py (high *= 2.0)
    a_high, b_high = project_to_admissible(1.0, 1.0, 0.5, metric="weighted", weights={"w_a": 0.0001, "w_b": 1000.0})
    assert math.isclose(a_high**2 + b_high**2, 0.25)

    # 5. admissibility.py global aggregation non-admissible trigger
    membrane_non = RadialMembrane()
    membrane_non.strings[0].activation = 10.0
    boundary_non = BoundaryGeometry(base_radius=0.01)
    for k in boundary_non.radius_deviation:
        boundary_non.radius_deviation[k] = -0.009
    agg = global_closure_aggregation(membrane_non, boundary_non, samples=12)
    assert agg["is_admissible"] is False

    # 6. envelope.py ClaimStateLedger & FalsificationStack corner cases
    ledger = ClaimStateLedger()
    ledger.record_claim("c1", "assert", "pending", {})
    ledger.update_status("c1", "verified")
    assert ledger.claims[0]["status"] == "verified"

    stack = FalsificationStack()
    assert stack.pop_violation() is None

    # evaluate_envelope corner cases: block/reproject/constrain
    envelope_test = BrimEnvelope(energy_threshold=0.02)
    membrane_env = RadialMembrane()
    membrane_env.strings[0].activation = 0.6
    boundary_env = BoundaryGeometry(base_radius=0.15)

    verdict, meta = envelope_test.evaluate_envelope(membrane_env, boundary_env)
    assert verdict == "re-project"

    envelope_test_constrain = BrimEnvelope(energy_threshold=1.0)
    verdict_c, meta_c = envelope_test_constrain.evaluate_envelope(membrane_env, boundary_env)
    assert verdict_c == "constrain"

    # 7. mesh.py governor and route/execute exceptions
    with pytest.raises(ValueError):
        MeshGovernor("g1", "InvalidTier")

    mesh_test = FederatedShardMesh()
    assert mesh_test.compute_mesh_coherence() == 1.0

    # Quarantined shard execution rejection
    shard_q = FederatedShard(shard_id="node_q", state=ShardState.QUARANTINED)
    mesh_test.add_shard(shard_q)
    # Workload that maps to it
    workload = {"capacity": 0.1, "privacy": 0.1, "latency": 100.0, "cost": 5.0}
    # Force Type-W channel to select it
    mesh_test.channels["Type-W"].route_workload = lambda w, s: shard_q
    assert mesh_test.route_and_execute(workload) is False

    # Low trust shard execution rejection
    shard_lt = FederatedShard(shard_id="node_lt", state=ShardState.ACTIVE, trust_score=0.2)
    mesh_test.channels["Type-W"].route_workload = lambda w, s: shard_lt
    assert mesh_test.route_and_execute(workload) is False

    # 8. routing.py invalid channel type
    with pytest.raises(ValueError):
        FederatedRoutingChannel("c1", "InvalidType")

    # 9. saopromotion.py invalid target layer & admit verdict
    promotor = SAOPromotor(promotion_threshold=10.0) # threshold extremely high so we hit "admit"
    membrane_sao = RadialMembrane()
    # Ensure activations are asymmetric to have non-zero field but tiny so we don't violate capacity
    membrane_sao.strings[0].activation = 0.01
    boundary_sao = BoundaryGeometry(base_radius=1.5)

    # Target layer invalid exception
    with pytest.raises(ValueError):
        promotor.promote(membrane_sao, boundary_sao, "InvalidLayer")

    verdict_sao, p_sao, meta_sao = promotor.promote(membrane_sao, boundary_sao, "Holistic Governor")
    assert verdict_sao == "admit"

    # SAO promotion with reproject and constrain verdicts
    membrane_constrain = RadialMembrane()
    membrane_constrain.strings[0].activation = 0.8
    boundary_constrain = BoundaryGeometry(base_radius=0.14)
    verdict_constrain, _, _ = promotor.promote(membrane_constrain, boundary_constrain, "Holistic Governor")
    assert verdict_constrain == "constrain"

    membrane_reproject = RadialMembrane()
    membrane_reproject.strings[0].activation = 3.5
    boundary_reproject = BoundaryGeometry(base_radius=1.0)
    with patch('radial_membrane_ai.saopromotion.closure_ratio', return_value=0.5):
        with patch('radial_membrane_ai.saopromotion.project_to_admissible', return_value=(0.0, 0.0)):
            verdict_rep, _, _ = promotor.promote(membrane_reproject, boundary_reproject, "Holistic Governor")
            assert verdict_rep == "reproject"

    # 10. shard.py admissibility reject paths
    shard_rej = FederatedShard(
        shard_id="node_rej",
        consent_granted=False,
        capacity=1.0,
        privacy_level=1.0,
        latency=10.0,
        cost_factor=1.0,
        trust_score=1.0,
        policy_compliance=1.0
    )
    assert shard_rej.evaluate_admissibility({"capacity": 0.5}) is False

    shard_rej.consent_granted = True
    assert shard_rej.evaluate_admissibility({"capacity": 2.0}) is False
    assert shard_rej.evaluate_admissibility({"privacy": 2.0}) is False
    assert shard_rej.evaluate_admissibility({"latency": 5.0}) is False
    assert shard_rej.evaluate_admissibility({"cost": 0.5}) is False

    shard_rej.trust_score = 0.1
    assert shard_rej.evaluate_admissibility({}) is False

    # 11. simulation.py run step envelope verdicts branching (block, constrain, re-project)
    sim = RainbowSimulation()
    # Excite to block
    sim.membrane.strings[0].activation = 1.0
    sim.boundary.base_radius = 0.001
    for k in sim.boundary.radius_deviation:
        sim.boundary.radius_deviation[k] = -0.0009

    # This run_step will trigger block / constrain path
    sim.run_step("planning")
    assert len(sim.brim_verdicts) > 0

    # Excite to constrain
    sim_c = RainbowSimulation()
    sim_c.brim_envelope.evaluate_envelope = lambda m, b: ("constrain", {"brim_energy": 0.02})
    sim_c.run_step("planning")
    assert "constrain" in sim_c.brim_verdicts

    # Excite to re-project
    sim_rep = RainbowSimulation()
    sim_rep.brim_envelope.evaluate_envelope = lambda m, b: ("re-project", {"brim_energy": 0.02})
    sim_rep.run_step("planning")
    assert "re-project" in sim_rep.brim_verdicts
