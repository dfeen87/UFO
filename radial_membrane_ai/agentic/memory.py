"""Fail-closed adapter from governed action evidence to semantic memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from radial_membrane_ai.agentic.contracts import (
    ActionProposal,
    IntentContract,
    MemoryCommitResult,
    MemoryCommitStatus,
    MemoryQualificationResult,
    MemoryQualificationStatus,
    MemoryStageEvidence,
    Observation,
    ProvisionalMemoryCandidate,
    VerificationResult,
    VerificationStatus,
    stable_digest,
)
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.saopromotion import SAOPromotor
from radial_membrane_ai.semantic_memory.core import AgentSemanticMemory, MemoryRecord, write_memory
from radial_membrane_ai.semantic_memory.policy import AdmissibilityContext, MemoryPolicy


def policy_digest(policy: MemoryPolicy) -> str:
    return stable_digest(policy)


def _mutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return value


def semantic_memory_digest(memory: AgentSemanticMemory) -> str:
    """Digest authoritative semantics, excluding diagnostic wall-clock fields."""
    records = {
        key: {
            "value": record.value,
            "tags": sorted(record.tags),
            "origin": record.origin_agent_id,
            "policy": record.policy_envelope,
        }
        for key, record in sorted(memory.local_store.items())
    }
    commits = dict(sorted(getattr(memory, "governed_commits", {}).items()))
    curvature = {
        "curvature": memory.curvature_state.curvature,
        "tension": memory.curvature_state.tension,
        "access_count": memory.curvature_state.access_count,
        "conflict_count": memory.curvature_state.conflict_count,
    }
    return stable_digest({
        "records": records, "governed_commits": commits, "curvature_state": curvature,
    })


@dataclass(frozen=True)
class GovernedMemoryConfig:
    policy: MemoryPolicy
    context: AdmissibilityContext
    tags: frozenset[str] = frozenset({"governed-action-evidence"})


class GovernedMemoryPipeline:
    """Candidate/evaluate/final-verdict/atomic-commit orchestration."""

    def __init__(
        self,
        memory: AgentSemanticMemory,
        sao: SAOPromotor,
        config: GovernedMemoryConfig | None,
        *,
        failure_injector: Callable[[], None] | None = None,
    ) -> None:
        self.memory = memory
        self.sao = sao
        self.config = config
        self.failure_injector = failure_injector
        if not hasattr(memory, "governed_commits"):
            memory.governed_commits = {}  # type: ignore[attr-defined]

    def create_candidate(
        self,
        intent: IntentContract,
        proposal: ActionProposal,
        execution_id: str,
        observation: Observation,
        verification: VerificationResult,
    ) -> ProvisionalMemoryCandidate:
        if self.config is None:
            raise ValidationError("governed memory requires explicit policy and admissibility context.")
        return ProvisionalMemoryCandidate(
            intent.intent_digest, proposal.plan_revision, proposal.proposal_digest,
            execution_id, observation.observation_id, stable_digest(verification),
            f"governed:{proposal.action_id}:{proposal.proposal_digest[:16]}",
            observation.evidence, self.config.tags, policy_digest(self.config.policy),
        )

    def govern(
        self,
        intent: IntentContract,
        proposal: ActionProposal,
        execution_id: str,
        observation: Observation | None,
        verification: VerificationResult,
        membrane: RadialMembrane,
        boundary: BoundaryGeometry,
    ) -> MemoryStageEvidence:
        before = semantic_memory_digest(self.memory)
        empty = MemoryCommitResult(
            MemoryCommitStatus.NOT_ATTEMPTED, None, None, before, ("NO_COMMIT_ATTEMPT",)
        )
        if self.config is None:
            return MemoryStageEvidence(None, None, None, "NO_CANDIDATE", empty,
                                       ("MEMORY_GOVERNANCE_NOT_CONFIGURED",))
        if observation is None or not observation.complete:
            return MemoryStageEvidence(None, None, None, "NO_CANDIDATE", empty,
                                       ("COMPLETE_OBSERVATION_REQUIRED",))
        candidate = self.create_candidate(intent, proposal, execution_id, observation, verification)
        if verification.status is not VerificationStatus.VERIFIED:
            qualification = MemoryQualificationResult(
                MemoryQualificationStatus.REJECTED, candidate.candidate_digest,
                (f"ACTION_VERIFICATION_{verification.status.value}",),
            )
            return MemoryStageEvidence(candidate, qualification, None, "VERIFICATION_REJECTED", empty,
                                       qualification.reason_codes)

        try:
            verdict, p_sao, temporal = self.sao.evaluate(membrane, boundary, "semantic memory")
        except (ValueError, TypeError, ArithmeticError):
            verdict, p_sao, temporal = "unknown", float("nan"), {}
        temporal_digest = stable_digest(temporal) if temporal else None
        if not math.isfinite(p_sao):
            q_status = MemoryQualificationStatus.UNKNOWN
            q_reasons = ("TEMPORAL_EVIDENCE_NON_FINITE",)
        elif verdict == "ascend":
            q_status = MemoryQualificationStatus.QUALIFIED
            q_reasons = ("TEMPORAL_QUALIFIED",)
        elif verdict in {"admit", "constrain", "reproject"}:
            q_status = MemoryQualificationStatus.PROVISIONAL
            q_reasons = (f"TEMPORAL_{verdict.upper()}",)
        elif verdict == "block":
            q_status = MemoryQualificationStatus.REJECTED
            q_reasons = ("TEMPORAL_BLOCKED",)
        else:
            q_status = MemoryQualificationStatus.UNKNOWN
            q_reasons = ("TEMPORAL_EVIDENCE_UNKNOWN",)
        qualification = MemoryQualificationResult(
            q_status, candidate.candidate_digest, q_reasons, temporal_digest,
        )
        if q_status is not MemoryQualificationStatus.QUALIFIED:
            return MemoryStageEvidence(candidate, qualification, None, q_status.value, empty, q_reasons)

        record = MemoryRecord(candidate.candidate_key, candidate.evidence, set(candidate.tags), 0.0, 0.0,
                              self.memory.agent_id, self.config.policy)
        context_values = (
            self.config.context.coherence, self.config.context.cost_factor,
            self.config.context.stability_energy,
        )
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in context_values)
                or not isinstance(self.config.context.quarantined, bool)):
            return MemoryStageEvidence(candidate, qualification, None, "POLICY_EVIDENCE_UNKNOWN", empty,
                                       ("ADMISSIBILITY_CONTEXT_INVALID",))
        admissible = self.config.policy.check_write_admissible(record, self.config.context)
        if not admissible:
            return MemoryStageEvidence(candidate, qualification, False, "POLICY_REJECTED", empty,
                                       ("MEMORY_POLICY_REJECTED",))

        commit = self._atomic_commit(candidate)
        if commit.status in {MemoryCommitStatus.COMMITTED, MemoryCommitStatus.ALREADY_COMMITTED}:
            # Publication is deliberately last: neither SAO nor memory history can
            # claim success until the final verdict and durable commit both exist.
            if commit.status is MemoryCommitStatus.COMMITTED:
                self.sao.commit_evaluation("semantic memory", temporal["collapse_activation"], p_sao, "ascend")
            return MemoryStageEvidence(candidate, qualification, True, "COMMIT_ACCEPTED", commit,
                                       commit.reason_codes)
        return MemoryStageEvidence(candidate, qualification, True, "COMMIT_FAILED", commit,
                                   commit.reason_codes)

    def _atomic_commit(self, candidate: ProvisionalMemoryCandidate) -> MemoryCommitResult:
        commits = self.memory.governed_commits  # type: ignore[attr-defined]
        prior = commits.get(candidate.candidate_digest)
        before_digest = semantic_memory_digest(self.memory)
        if prior is not None:
            if prior != candidate.candidate_key:
                raise ValidationError("candidate replay target mismatch.")
            return MemoryCommitResult(
                MemoryCommitStatus.ALREADY_COMMITTED, candidate.candidate_digest,
                candidate.candidate_key, before_digest, ("EXACT_REPLAY",),
            )
        snapshot = (
            deepcopy(self.memory.local_store), deepcopy(self.memory.history),
            deepcopy(self.memory.residuals), deepcopy(self.memory.curvature_state), deepcopy(commits),
        )
        try:
            assert self.config is not None
            result = write_memory(
                self.memory, candidate.candidate_key, _mutable(candidate.evidence), set(candidate.tags),
                self.config.policy, self.config.context,
            )
            if not result.success:
                raise ValidationError(result.error_message or "semantic memory write failed")
            if self.failure_injector is not None:
                self.failure_injector()
            commits[candidate.candidate_digest] = candidate.candidate_key
            digest = semantic_memory_digest(self.memory)
            return MemoryCommitResult(
                MemoryCommitStatus.COMMITTED, candidate.candidate_digest, candidate.candidate_key,
                digest, ("DURABLE_MEMORY_COMMITTED",),
            )
        except Exception:
            (self.memory.local_store, self.memory.history, self.memory.residuals,
             self.memory.curvature_state, restored) = snapshot
            self.memory.governed_commits = restored  # type: ignore[attr-defined]
            return MemoryCommitResult(
                MemoryCommitStatus.FAILED, candidate.candidate_digest, candidate.candidate_key,
                semantic_memory_digest(self.memory), ("ATOMIC_COMMIT_ROLLED_BACK",),
            )
