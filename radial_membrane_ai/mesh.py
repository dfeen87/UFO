# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Federated Shard Mesh and Governors module.

Implements Shard Mesh and Governors:
- Local Shard Governor
- Domain Governor
- Root Mesh Governor
- Holistic Mesh Governor

And calculates the global mesh coherence score:
C_mesh(t) = normalize(
    w_q * Q_route + w_t * T_trust + w_e * E_economic -
    w_r * R_residual - w_p * P_policy - w_l * L_latency - w_f * F_failure
)
"""

from __future__ import annotations
import math
from typing import List, Dict, Any
from radial_membrane_ai.shard import FederatedShard, ShardState
from radial_membrane_ai.residuals import ResidualLedger
from radial_membrane_ai.routing import FederatedRoutingChannel
from radial_membrane_ai.numeric import finite_real, require_finite_real


class MeshGovernor:
    """
    Governor governing a segment or root of the federated shard mesh.
    """

    def __init__(self, gov_id: str, tier: str = "Local") -> None:
        self.gov_id = gov_id
        valid_tiers = {"Local", "Domain", "Root", "Holistic"}
        if tier not in valid_tiers:
            raise ValueError(f"Invalid Governor tier: {tier}")
        self.tier = tier

    def audit_shard(self, shard: FederatedShard, ledger: ResidualLedger) -> bool:
        """
        Audits shard, quarantining it if failures exceed policy limits.
        """
        failures = ledger.get_failures_by_shard(shard.shard_id)
        critical_failures = []
        for f in failures:
            sev = f.severity
            if isinstance(sev, str):
                if sev.lower() in ("high", "critical"):
                    critical_failures.append(f)
            else:
                if any(s.lower() in ("high", "critical") for s in sev):
                    critical_failures.append(f)

        if len(critical_failures) >= 2 or len(failures) >= 5:
            shard.state = ShardState.QUARANTINED
            return False
        return True


class FederatedShardMesh:
    """
    Unified Federated Mesh managing shards, multi-tier governors, V-channels,
    residual tracking, and global mesh coherence.
    """

    def __init__(self) -> None:
        self.shards: List[FederatedShard] = []
        self.governors: Dict[str, MeshGovernor] = {
            "Local": MeshGovernor("local_gov", "Local"),
            "Domain": MeshGovernor("domain_gov", "Domain"),
            "Root": MeshGovernor("root_gov", "Root"),
            "Holistic": MeshGovernor("holistic_gov", "Holistic")
        }
        self.ledger = ResidualLedger()
        self.channels: Dict[str, FederatedRoutingChannel] = {
            "Type-W": FederatedRoutingChannel("chan_w", "Type-W"),
            "Type-R": FederatedRoutingChannel("chan_r", "Type-R"),
            "Type-T": FederatedRoutingChannel("chan_t", "Type-T"),
            "Type-A": FederatedRoutingChannel("chan_a", "Type-A"),
            "Type-E": FederatedRoutingChannel("chan_e", "Type-E")
        }

    def add_shard(self, shard: FederatedShard) -> None:
        self.shards.append(shard)

    def route_and_execute(self, workload: Dict[str, Any]) -> bool:
        """
        Executes routing via channels, logs exceptions/residuals if failure occurs.
        """
        chan_w = self.channels["Type-W"]
        target_shard = chan_w.route_workload(workload, self.shards)

        if target_shard is None:
            # Exception: log residual failure
            self.ledger.log_failure(
                record_id=f"fail_route_{len(self.ledger.records)}",
                error_type="aggregation_failure",
                shard_id="none",
                severity="high",
                details={"workload": workload}
            )
            return False

        # Attempt execute simulation
        if target_shard.state == ShardState.QUARANTINED:
            self.ledger.log_failure(
                record_id=f"fail_exec_{len(self.ledger.records)}",
                error_type="failed_execution",
                shard_id=target_shard.shard_id,
                severity="critical",
                details={"reason": "shard_quarantined"}
            )
            return False

        # Simulate execution success based on trust score
        if target_shard.trust_score < 0.6:
            self.ledger.log_failure(
                record_id=f"fail_exec_{len(self.ledger.records)}",
                error_type="invalid_output",
                shard_id=target_shard.shard_id,
                severity="medium",
                details={"reason": "low_trust"}
            )
            return False

        return True

    def run_mesh_governance(self) -> None:
        """
        Runs multi-tier audit over all shards.
        """
        for shard in self.shards:
            self.governors["Local"].audit_shard(shard, self.ledger)
            self.governors["Domain"].audit_shard(shard, self.ledger)
            self.governors["Root"].audit_shard(shard, self.ledger)
            self.governors["Holistic"].audit_shard(shard, self.ledger)

    def compute_mesh_coherence(
        self,
        w_q: float = 0.2,
        w_t: float = 0.2,
        w_e: float = 0.15,
        w_r: float = 0.15,
        w_p: float = 0.1,
        w_l: float = 0.1,
        w_f: float = 0.1
    ) -> float:
        """
        Computes the global mesh coherence score:
        C_mesh(t) = normalize(
            w_q * Q_route + w_t * T_trust + w_e * E_economic -
            w_r * R_residual - w_p * P_policy - w_l * L_latency - w_f * F_failure
        )
        """
        weights = (w_q, w_t, w_e, w_r, w_p, w_l, w_f)
        converted_weights = tuple(require_finite_real(value, "mesh weight") for value in weights)
        if any(value < 0.0 for value in converted_weights):
            raise ValueError("mesh weights must be non-negative.")
        w_q, w_t, w_e, w_r, w_p, w_l, w_f = converted_weights
        if not self.shards:
            return 1.0

        for shard in self.shards:
            evidence = (
                shard.quality_score, shard.trust_score, shard.cost_factor,
                shard.policy_compliance, shard.latency,
            )
            converted = tuple(finite_real(value) for value in evidence)
            if any(value is None for value in converted):
                raise ValueError("mesh shard evidence must be finite and representable.")
            quality, trust, cost, policy, latency = converted
            if (quality is None or trust is None or cost is None
                    or policy is None or latency is None):  # mypy narrowing
                raise ValueError("mesh shard evidence must be finite and representable.")
            if not (0.0 <= quality <= 1.0 and 0.0 <= trust <= 1.0
                    and cost >= 0.0 and 0.0 <= policy <= 1.0 and latency >= 0.0):
                raise ValueError("mesh shard evidence is outside its model domain.")

        # Terminal shards cannot contribute positive authorization/coherence
        # evidence.  They remain represented by failure, residual and policy
        # penalties so quarantine cannot improve the aggregate.
        terminal = {ShardState.QUARANTINED, ShardState.REVOKED, ShardState.EXPIRED}
        eligible_shards = [s for s in self.shards if s.state not in terminal]
        active_shards = [s for s in self.shards if s.state == ShardState.ACTIVE]
        q_route = sum(s.quality_score for s in active_shards) / len(active_shards) if active_shards else 0.5
        t_trust = (sum(s.trust_score for s in eligible_shards) / len(eligible_shards)
                   if eligible_shards else 0.0)
        e_economic = (sum(1.0 / (s.cost_factor + 1e-6) for s in eligible_shards)
                      / len(eligible_shards) if eligible_shards else 0.0)

        r_residual = self.ledger.get_total_severity_index() / 100.0
        # Keep terminal non-compliance as retained negative evidence without
        # letting favorable stale policy dilute the eligible mesh denominator.
        p_policy = sum(1.0 - s.policy_compliance for s in self.shards) / max(len(eligible_shards), 1)
        l_latency = (sum(s.latency for s in eligible_shards) / len(eligible_shards) / 1000.0
                     if eligible_shards else 0.0)

        terminal_count = len([s for s in self.shards if s.state in terminal])
        f_failure = terminal_count / len(self.shards)

        # Raw formula evaluation
        raw_val = (
            w_q * q_route +
            w_t * t_trust +
            w_e * e_economic -
            w_r * r_residual -
            w_p * p_policy -
            w_l * l_latency -
            w_f * f_failure
        )

        # Min-max sigmoid styled normalization to [0, 1]
        if not math.isfinite(raw_val):
            raise ValueError("mesh coherence evidence must remain finite.")
        if raw_val >= 0.0:
            return float(1.0 / (1.0 + math.exp(-raw_val)))
        exp_value = math.exp(raw_val)
        return float(exp_value / (1.0 + exp_value))
