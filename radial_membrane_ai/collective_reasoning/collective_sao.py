"""
Collective Symmetric Ascension Operator (SAO) for multi-agent collective reasoning.
"""

from __future__ import annotations
import time
from typing import Any, Tuple, Optional, Dict
from radial_membrane_ai.semantic_memory.core import MemoryRecord, MeshSemanticMemory
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.residuals import ResidualRecord


def collective_sao_promote(
    level: str,  # "local" (agent -> cluster), "cluster" (cluster -> mesh), "global" (mesh -> global store)
    source_entity: Any,
    target_entity: Any,
    key: str,
    global_envelope: PolicyEnvelope,
    coherence_score_val: float,
    stability_bands: Dict[str, str],
    current_ticks: int,
    ledger: Optional[Any] = None
) -> Tuple[bool, float, Optional[MemoryRecord]]:
    """
    Executes a Symmetric Ascension Operator (SAO) promotion across agents/clusters.

    Returns:
        A tuple of (success_flag, p_sao, promoted_record).
    """
    p_sao = max(0.0, 1.0 - coherence_score_val)

    # 1. Level-specific Gating and Thresholds Checks
    if level == "local":
        # Local SAO (agent -> cluster)
        # Requires short-range gate: ticks >= 5, tension < 2.0 (green/yellow, no red)
        t_state = getattr(source_entity.membrane, "temporal_state", None) if hasattr(source_entity, "membrane") else None
        ticks = t_state.consecutive_admissible_ticks if t_state else current_ticks
        avg_tension = t_state.accumulated_tension if t_state else 0.0

        if ticks < 5:
            return False, p_sao, None
        if avg_tension >= 2.0:
            return False, p_sao, None

        # Coherence check
        if coherence_score_val < 0.7:
            return False, p_sao, None

        # Stability compliance: no red band for agent's cluster or global
        if stability_bands.get("global") == "red" or stability_bands.get(getattr(target_entity, "cluster_id", "")) == "red":
            return False, p_sao, None

        # Envelope check
        local_env = getattr(source_entity, "policy_envelope", None) or PolicyEnvelope()
        aligned_envelope = global_envelope.intersect(local_env)
        # Check compliance
        trust = getattr(source_entity.shard, "trust_score", 1.0) if hasattr(source_entity, "shard") else 1.0
        compliance = getattr(source_entity.shard, "policy_compliance", 1.0) if hasattr(source_entity, "shard") else 1.0
        cost_band = 0 if compliance >= 0.8 else (1 if compliance >= 0.5 else 2)
        if not aligned_envelope.validate_compliance(
            current_trust=trust,
            current_cost_band=cost_band,
            current_stability_band=stability_bands.get("global", "green"),
            current_ticks=ticks,
            agent_ids={source_entity.agent_id} if hasattr(source_entity, "agent_id") else None
        ):
            return False, p_sao, None

    elif level == "cluster":
        # Cluster SAO (cluster -> global mesh)
        # Requires mid-range gate: ticks >= 20, cluster tension predominantly green (< 1.0)
        c_t_state = getattr(source_entity.membrane, "temporal_state", None) if hasattr(source_entity, "membrane") else None
        ticks = c_t_state.consecutive_admissible_ticks if c_t_state else current_ticks
        avg_tension = source_entity.tension_metric if hasattr(source_entity, "tension_metric") else 0.0

        if ticks < 20:
            return False, p_sao, None
        if avg_tension >= 1.0:
            return False, p_sao, None

        # Coherence check
        if coherence_score_val < 0.7:
            return False, p_sao, None

        # Stability check: no red band allowed. Yellow is tolerated.
        if stability_bands.get("global") == "red" or stability_bands.get(source_entity.cluster_id) == "red":
            return False, p_sao, None

        # Envelope check: cluster envelope + global envelope intersected and satisfied
        cluster_env = getattr(source_entity, "policy_envelope", None) or PolicyEnvelope()
        aligned_envelope = global_envelope.intersect(cluster_env)
        # Check compliance
        trust = 1.0
        cost_band = 0
        if not aligned_envelope.validate_compliance(
            current_trust=trust,
            current_cost_band=cost_band,
            current_stability_band=stability_bands.get("global", "green"),
            current_ticks=ticks,
            cluster_ids={source_entity.cluster_id} if hasattr(source_entity, "cluster_id") else None
        ):
            return False, p_sao, None

    elif level == "global":
        # Global SAO (mesh -> global semantic memory)
        # Requires long-range gate: ticks >= 50, global tension predominantly green (< 1.0), no red
        if current_ticks < 50:
            return False, p_sao, None

        # Stability check: require GREEN at global level
        if stability_bands.get("global", "green") != "green":
            return False, p_sao, None

        # Coherence check
        if coherence_score_val < 0.8:
            return False, p_sao, None

        # Envelope check: global envelope satisfied
        if not global_envelope.validate_compliance(
            current_trust=1.0,
            current_cost_band=0,
            current_stability_band="green",
            current_ticks=current_ticks
        ):
            return False, p_sao, None

    else:
        raise ValueError(f"Unknown level: {level}")

    # 2. Extract or Create Memory Record to Promote
    # Determine the memory record to promote
    record_to_promote: Optional[MemoryRecord] = None

    if level == "local":
        source_mem = getattr(source_entity, "semantic_memory", None)
        if source_mem and hasattr(source_mem, "local_store") and key in source_mem.local_store:
            record_to_promote = source_mem.local_store[key]
    elif level == "cluster":
        source_mem = getattr(source_entity, "semantic_memory", None)
        if source_mem and hasattr(source_mem, "global_store") and key in source_mem.global_store:
            record_to_promote = source_mem.global_store[key]
    elif level == "global":
        # From mesh to global persistent memory
        if hasattr(source_entity, "global_store") and key in source_entity.global_store:
            record_to_promote = source_entity.global_store[key]

    if record_to_promote is None:
        # Create a dummy or summary record if none exists to support promotions of activations/patterns
        record_to_promote = MemoryRecord(
            key=key,
            value=f"Summary of {level} SAO at {time.time()}",
            tags={"promoted", level},
            created_at=time.time(),
            updated_at=time.time(),
            origin_agent_id=getattr(source_entity, "agent_id", getattr(source_entity, "cluster_id", "mesh")),
            policy_envelope=global_envelope
        )

    # 3. Handle Overwrites and Residual Separation
    target_store = getattr(target_entity, "global_store", None)
    prev_residual: Optional[ResidualRecord] = None
    if target_store is not None and key in target_store:
        prev_shared = target_store[key]
        res_id = f"sao_overwrite_{level}_{key}_{int(time.time() * 1000)}"
        prev_residual = ResidualRecord(
            record_id=res_id,
            error_type="collective_memory_overwrite_residual",
            shard_id=getattr(source_entity, "agent_id", getattr(source_entity, "cluster_id", "mesh")),
            severity="medium",
            details={
                "key": key,
                "overwritten_value": prev_shared.value,
                "p_sao": p_sao
            },
            timestamp=time.time(),
            escalation_tier="none"
        )
        if ledger is not None:
            ledger.log_failure(
                record_id=prev_residual.record_id,
                error_type=prev_residual.error_type,
                shard_id=prev_residual.shard_id,
                severity=prev_residual.severity,
                details=prev_residual.details,
                timestamp=prev_residual.timestamp,
                escalation_tier=prev_residual.escalation_tier
            )

    # 4. Perform Promotion
    promoted_rec = MemoryRecord(
        key=key,
        value=record_to_promote.value,
        tags=record_to_promote.tags.union({"promoted", level}),
        created_at=record_to_promote.created_at,
        updated_at=time.time(),
        origin_agent_id=getattr(source_entity, "agent_id", getattr(source_entity, "cluster_id", "mesh")),
        policy_envelope=record_to_promote.policy_envelope,
        residual_state=prev_residual
    )

    if target_store is not None:
        target_store[key] = promoted_rec

    # Append promotion history and curvature tension updates if applicable
    target_mem = target_entity
    if hasattr(target_mem, "add_history"):
        target_mem.add_history(
            getattr(source_entity, "agent_id", getattr(source_entity, "cluster_id", "mesh")),
            "promote",
            key,
            "success"
        )
    if hasattr(target_mem, "curvature_state"):
        target_mem.curvature_state.update_on_write(size_factor=0.2)

    return True, p_sao, promoted_rec
