"""
Core data structures and APIs for Policy-Bound Semantic Memory.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Optional, Literal

from radial_membrane_ai.residuals import ResidualRecord, ResidualLedger
from radial_membrane_ai.semantic_memory.policy import MemoryPolicy, AdmissibilityContext, PolicyContext
from radial_membrane_ai.semantic_memory.curvature import MemoryCurvatureState
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope


@dataclass(frozen=True)
class MemoryEvent:
    """
    Log of an event/action occurring in semantic memory.
    """
    event_id: str
    action: Literal["write", "read", "overwrite", "read_rejected", "write_rejected", "promote", "promote_rejected"]
    agent_id: str
    key: str
    timestamp: float
    status: Literal["success", "rejected"]


@dataclass
class MemoryRecord:
    """
    A single policy-bounded semantic memory record.
    """
    key: str
    value: Any
    tags: Set[str]
    created_at: float
    updated_at: float
    origin_agent_id: Optional[str]
    policy_envelope: PolicyEnvelope
    residual_state: Optional[ResidualRecord] = None


@dataclass
class AgentViewConfig:
    """
    Tracks configurations and metadata about an agent's view/access to the mesh.
    """
    agent_id: str
    read_granted: bool = True
    write_granted: bool = True
    last_access: float = 0.0


@dataclass
class MemoryResult:
    """
    Result of a memory write or read attempt.
    """
    success: bool
    record: Optional[MemoryRecord]
    error_message: Optional[str] = None


@dataclass
class PromotionResult:
    """
    Result of a local memory promotion to the shared mesh.
    """
    success: bool
    record: Optional[MemoryRecord]
    p_sao: float = 0.0
    error_message: Optional[str] = None


class AgentSemanticMemory:
    """
    Governs local, policy-bounded semantic memory for a single UFO agent.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.local_store: Dict[str, MemoryRecord] = {}
        self.history: List[MemoryEvent] = []
        self.residuals: List[ResidualRecord] = []
        self.curvature_state = MemoryCurvatureState()

    def add_history(self, action: Any, key: str, status: Any) -> None:
        event_id = f"event_{self.agent_id}_{len(self.history)}_{int(time.time() * 1000)}"
        self.history.append(
            MemoryEvent(
                event_id=event_id,
                action=action,
                agent_id=self.agent_id,
                key=key,
                timestamp=time.time(),
                status=status
            )
        )


class MeshSemanticMemory:
    """
    Governs the shared global semantic memory across the federated shard mesh.
    """

    def __init__(self) -> None:
        self.global_store: Dict[str, MemoryRecord] = {}
        self.agent_views: Dict[str, AgentViewConfig] = {}
        self.history: List[MemoryEvent] = []
        self.curvature_state = MemoryCurvatureState()

    def add_history(self, agent_id: str, action: Any, key: str, status: Any) -> None:
        event_id = f"event_mesh_{len(self.history)}_{int(time.time() * 1000)}"
        self.history.append(
            MemoryEvent(
                event_id=event_id,
                action=action,
                agent_id=agent_id,
                key=key,
                timestamp=time.time(),
                status=status
            )
        )


# =================================================================------------
# Core APIs
# =================================================================------------

def write_memory(
    memory: AgentSemanticMemory,
    key: str,
    value: Any,
    tags: Set[str],
    policy: MemoryPolicy,
    context: AdmissibilityContext,
    global_ledger: Optional[ResidualLedger] = None
) -> MemoryResult:
    """
    Writes a record to local agent semantic memory, performing admissibility checks,
    preserving overwrite residuals, and logging events.
    """
    # 1. Instantiate temporary record for check
    temp_rec = MemoryRecord(
        key=key,
        value=value,
        tags=tags,
        created_at=time.time(),
        updated_at=time.time(),
        origin_agent_id=memory.agent_id,
        policy_envelope=policy
    )

    # 2. Check admissibility via policy
    if not policy.check_write_admissible(temp_rec, context):
        memory.add_history("write_rejected", key, "rejected")
        memory.curvature_state.update_on_conflict(conflict_intensity=0.3)
        # Log to ledger as policy conflict
        if global_ledger is not None:
            global_ledger.log_failure(
                record_id=f"write_rejected_{memory.agent_id}_{int(time.time() * 1000)}",
                error_type="memory_policy_conflict",
                shard_id=memory.agent_id,
                severity="medium",
                details={"key": key, "coherence": context.coherence}
            )
        return MemoryResult(success=False, record=None, error_message="Write rejected by policy.")

    # 3. Check for previous state (destructive overwrite prevention)
    prev_residual: Optional[ResidualRecord] = None
    if key in memory.local_store:
        prev_rec = memory.local_store[key]
        # Formulate residual record representing prior state
        prev_residual = ResidualRecord(
            record_id=f"residual_mem_{memory.agent_id}_{key}_{int(time.time() * 1000)}",
            error_type="memory_overwrite_residual",
            shard_id=memory.agent_id,
            severity="low",
            details={"key": key, "overwritten_value": prev_rec.value, "overwritten_tags": list(prev_rec.tags)},
            timestamp=time.time(),
            escalation_tier="none"
        )
        memory.residuals.append(prev_residual)
        memory.add_history("overwrite", key, "success")
        if global_ledger is not None:
            global_ledger.log_failure(
                record_id=prev_residual.record_id,
                error_type=prev_residual.error_type,
                shard_id=prev_residual.shard_id,
                severity=prev_residual.severity,
                details=prev_residual.details,
                timestamp=prev_residual.timestamp,
                escalation_tier=prev_residual.escalation_tier
            )
    else:
        memory.add_history("write", key, "success")

    # 4. Form final record and store
    final_rec = MemoryRecord(
        key=key,
        value=value,
        tags=tags,
        created_at=memory.local_store[key].created_at if key in memory.local_store else time.time(),
        updated_at=time.time(),
        origin_agent_id=memory.agent_id,
        policy_envelope=policy,
        residual_state=prev_residual
    )
    memory.local_store[key] = final_rec
    memory.curvature_state.update_on_write(size_factor=0.1)

    return MemoryResult(success=True, record=final_rec)


def read_memory(
    memory: AgentSemanticMemory,
    key: str,
    context: PolicyContext
) -> Optional[MemoryRecord]:
    """
    Reads a record from local agent semantic memory, validating read permission.
    """
    if key not in memory.local_store:
        return None

    record = memory.local_store[key]
    if not record.policy_envelope.check_read_allowed(record, memory.agent_id, context):
        memory.add_history("read_rejected", key, "rejected")
        memory.curvature_state.update_on_conflict(conflict_intensity=0.15)
        return None

    memory.add_history("read", key, "success")
    memory.curvature_state.update_on_read(read_intensity=0.02)
    return record


def list_memory(
    memory: AgentSemanticMemory,
    filter_tags: Optional[Set[str]] = None
) -> List[MemoryRecord]:
    """
    Lists local memory records, optionally filtering by tags.
    """
    records = list(memory.local_store.values())
    if filter_tags is None:
        return records
    return [r for r in records if filter_tags.intersection(r.tags)]


def promote_memory(
    mesh_memory: MeshSemanticMemory,
    agent_memory: AgentSemanticMemory,
    key: str,
    context: AdmissibilityContext,
    global_ledger: Optional[ResidualLedger] = None
) -> PromotionResult:
    """
    Promotes a local agent semantic memory record to the shared mesh semantic memory (global store)
    via an SAO-style promotion mechanism.
    Checks admissibility and logs event details.
    """
    if key not in agent_memory.local_store:
        return PromotionResult(success=False, record=None, error_message="Key not found in local memory.")

    local_rec = agent_memory.local_store[key]

    # Evaluate promotion admissibility via record's policy envelope
    if not local_rec.policy_envelope.check_write_admissible(local_rec, context):
        agent_memory.add_history("promote_rejected", key, "rejected")
        mesh_memory.add_history(agent_memory.agent_id, "promote_rejected", key, "rejected")
        mesh_memory.curvature_state.update_on_conflict(conflict_intensity=0.4)

        if global_ledger is not None:
            global_ledger.log_failure(
                record_id=f"promote_rejected_{agent_memory.agent_id}_{int(time.time() * 1000)}",
                error_type="memory_promotion_rejected",
                shard_id=agent_memory.agent_id,
                severity="high",
                details={"key": key, "coherence": context.coherence}
            )
        return PromotionResult(success=False, record=None, error_message="Promotion blocked by policy constraints.")

    # Shared global stores also require tracking previous overwrite residuals
    prev_residual: Optional[ResidualRecord] = None
    if key in mesh_memory.global_store:
        prev_shared = mesh_memory.global_store[key]
        prev_residual = ResidualRecord(
            record_id=f"residual_mesh_{agent_memory.agent_id}_{key}_{int(time.time() * 1000)}",
            error_type="mesh_memory_overwrite_residual",
            shard_id=agent_memory.agent_id,
            severity="medium",
            details={"key": key, "overwritten_value": prev_shared.value, "origin": prev_shared.origin_agent_id},
            timestamp=time.time(),
            escalation_tier="none"
        )
        if global_ledger is not None:
            global_ledger.log_failure(
                record_id=prev_residual.record_id,
                error_type=prev_residual.error_type,
                shard_id=prev_residual.shard_id,
                severity=prev_residual.severity,
                details=prev_residual.details,
                timestamp=prev_residual.timestamp,
                escalation_tier=prev_residual.escalation_tier
            )

    # SAO promotion residual computation:
    # A representation of "unaligned" or lost context during global sharing.
    # Mathematically modeled as an ascension residual: p_SAO.
    # Higher coherence context yields smaller residual p_SAO.
    p_sao = max(0.0, 1.0 - context.coherence)

    promoted_rec = MemoryRecord(
        key=key,
        value=local_rec.value,
        tags=local_rec.tags,
        created_at=mesh_memory.global_store[key].created_at if key in mesh_memory.global_store else time.time(),
        updated_at=time.time(),
        origin_agent_id=agent_memory.agent_id,
        policy_envelope=local_rec.policy_envelope,
        residual_state=prev_residual
    )

    mesh_memory.global_store[key] = promoted_rec
    mesh_memory.add_history(agent_memory.agent_id, "promote", key, "success")
    mesh_memory.curvature_state.update_on_write(size_factor=0.2)

    # Update agent view config last access
    if agent_memory.agent_id not in mesh_memory.agent_views:
        mesh_memory.agent_views[agent_memory.agent_id] = AgentViewConfig(agent_id=agent_memory.agent_id)
    mesh_memory.agent_views[agent_memory.agent_id].last_access = time.time()

    return PromotionResult(success=True, record=promoted_rec, p_sao=p_sao)


def read_shared_memory(
    mesh_memory: MeshSemanticMemory,
    agent_id: str,
    key: str,
    context: PolicyContext
) -> Optional[MemoryRecord]:
    """
    Reads a record from shared global semantic memory, checking permission for the querying agent.
    """
    if key not in mesh_memory.global_store:
        return None

    # Check agent's view permission
    view_cfg = mesh_memory.agent_views.get(agent_id)
    if view_cfg is not None and not view_cfg.read_granted:
        mesh_memory.add_history(agent_id, "read_rejected", key, "rejected")
        mesh_memory.curvature_state.update_on_conflict(conflict_intensity=0.2)
        return None

    record = mesh_memory.global_store[key]
    if not record.policy_envelope.check_read_allowed(record, agent_id, context):
        mesh_memory.add_history(agent_id, "read_rejected", key, "rejected")
        mesh_memory.curvature_state.update_on_conflict(conflict_intensity=0.15)
        return None

    mesh_memory.add_history(agent_id, "read", key, "success")
    mesh_memory.curvature_state.update_on_read(read_intensity=0.03)

    if view_cfg is not None:
        view_cfg.last_access = time.time()

    return record


def list_shared_memory(
    mesh_memory: MeshSemanticMemory,
    filter_tags: Optional[Set[str]] = None
) -> List[MemoryRecord]:
    """
    Lists shared memory records, optionally filtering by tags.
    """
    records = list(mesh_memory.global_store.values())
    if filter_tags is None:
        return records
    return [r for r in records if filter_tags.intersection(r.tags)]
