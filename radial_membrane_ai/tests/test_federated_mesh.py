"""
Unit and integration tests for Federated Shard Mesh, routing, and residuals.
"""

from __future__ import annotations
import pytest
from radial_membrane_ai.shard import FederatedShard, ShardState
from radial_membrane_ai.residuals import ResidualLedger
from radial_membrane_ai.routing import FederatedRoutingChannel
from radial_membrane_ai.mesh import FederatedShardMesh, MeshGovernor


def test_federated_mesh_and_routing() -> None:
    mesh = FederatedShardMesh()

    # 1. Create and add shards
    shard_1 = FederatedShard(shard_id="node_1", state=ShardState.IDLE, trust_score=0.9, capacity=2.0)
    shard_2 = FederatedShard(shard_id="node_2", state=ShardState.IDLE, trust_score=0.4, capacity=1.0)
    mesh.add_shard(shard_1)
    mesh.add_shard(shard_2)

    # 2. Test V-Channel route and execution
    workload = {"capacity": 1.5, "privacy": 0.5, "latency": 100.0, "cost": 5.0}

    # Node 1 is admissible, Node 2 is not (capacity 1.0 < workload 1.5)
    success = mesh.route_and_execute(workload)
    assert success is True
    assert shard_1.state == ShardState.ACTIVE

    # Evaluate routing projection
    routing_chan = FederatedRoutingChannel("test_chan", "Type-W")
    proj_workload = routing_chan.project_into_shard_geometry(workload, shard_2)
    assert proj_workload["capacity"] == 1.0

    # 3. Test exceptions and residuals logging
    # Heavy workload exceeding all capacity
    heavy_workload = {"capacity": 5.0, "privacy": 0.5, "latency": 10.0, "cost": 1.0}
    success_heavy = mesh.route_and_execute(heavy_workload)
    assert success_heavy is False

    failures = mesh.ledger.get_failures_by_shard("none")
    assert len(failures) > 0
    assert failures[0].error_type == "aggregation_failure"

    # 4. Mesh Coherence Score calculation
    coh = mesh.compute_mesh_coherence()
    assert 0.0 <= coh <= 1.0

    # 5. Audit and quarantine
    mesh.ledger.log_failure("f1", "failed_execution", "node_1", "critical", {})
    mesh.ledger.log_failure("f2", "failed_execution", "node_1", "critical", {})
    mesh.run_mesh_governance()
    assert shard_1.state == ShardState.QUARANTINED
