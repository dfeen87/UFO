"""
Residuals Ledger module for Shards and Mesh.

Tracks failed execution, partial completion, stale attestation, latency timeouts,
invalid outputs, policy conflicts, shard churn, and aggregation failures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ResidualRecord:
    record_id: str
    error_type: str
    shard_id: str
    severity: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    escalation_tier: str = "none"


class ResidualLedger:
    """
    Ledger to record failures, partial completions, timeouts, and exceptions in the mesh.
    """

    def __init__(self) -> None:
        self.records: List[ResidualRecord] = []

    def log_failure(
        self,
        record_id: str,
        error_type: str,
        shard_id: str,
        severity: str,
        details: Dict[str, Any],
        timestamp: float = 0.0,
        escalation_tier: str = "none"
    ) -> None:
        """
        Logs a failure or exception in the federated shard execution mesh.
        """
        self.records.append(
            ResidualRecord(
                record_id=record_id,
                error_type=error_type,
                shard_id=shard_id,
                severity=severity,
                details=details,
                timestamp=timestamp,
                escalation_tier=escalation_tier
            )
        )

    def get_failures_by_shard(self, shard_id: str) -> List[ResidualRecord]:
        return [r for r in self.records if r.shard_id == shard_id]

    def get_total_severity_index(self) -> float:
        """
        Calculates an overall index of residual issues.
        """
        severity_weights = {"low": 1.0, "medium": 3.0, "high": 5.0, "critical": 10.0}
        return sum(severity_weights.get(r.severity.lower(), 1.0) for r in self.records)
