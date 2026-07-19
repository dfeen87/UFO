"""
Adds serialization and file export capability to the ResidualLedger.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any

from radial_membrane_ai.residuals import ResidualLedger


def ledger_to_json(ledger: ResidualLedger, filepath: str) -> None:
    """
    Exports all records inside a ResidualLedger to a JSON file.
    Creates necessary directories if they do not exist.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    data = []
    for r in ledger.records:
        data.append({
            "record_id": r.record_id,
            "error_type": r.error_type,
            "shard_id": r.shard_id,
            "severity": r.severity,
            "details": r.details,
            "timestamp": r.timestamp,
            "escalation_tier": r.escalation_tier
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def export_simulation_results_to_json(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    Serializes simulation results dictionary containing numpy arrays or objects
    to a JSON file safely.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Convert complex objects and numpy arrays to serializable types
    def default_serializer(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    import numpy as np  # locally import to satisfy default_serializer

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, default=default_serializer, indent=4)
