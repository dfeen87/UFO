"""Immutable contracts for the v5 single-action governed transaction."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.numeric import finite_real


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string.")
    return value


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    result = finite_real(value)
    if result is None or (nonnegative and result < 0.0):
        raise ValidationError(f"{name} must be a finite non-negative real number.")
    return result


def _digest(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValidationError(f"{name} must be a canonical SHA-256 digest.")
    return result


def _vector(values: Sequence[Any], name: str, size: int = 8) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise ValidationError(f"{name} must contain exactly {size} dimensions.")
    return tuple(_number(value, f"{name}[{index}]") for index, value in enumerate(values))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and finite_real(value) is None:
            raise ValidationError("contract values cannot contain non-finite numbers.")
        return value
    raise ValidationError(f"contract value is not deterministically serializable: {type(value).__name__}.")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=str)
    if isinstance(value, Enum):
        return value.value
    return value


def stable_digest(value: Any) -> str:
    """Return a canonical SHA-256 identity for governed semantic data."""
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SideEffectClass(str, Enum):
    OBSERVATIONAL = "OBSERVATIONAL"
    REVERSIBLE = "REVERSIBLE"
    COMPENSATABLE = "COMPENSATABLE"
    IRREVERSIBLE = "IRREVERSIBLE"


class AdmissionDecision(str, Enum):
    ADMIT = "ADMIT"
    CONSTRAIN = "CONSTRAIN"
    REPROJECT = "REPROJECT"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class ExecutionState(str, Enum):
    NOT_EXECUTED = "NOT_EXECUTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ClosureDecision(str, Enum):
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    CONSTRAIN = "CONSTRAIN"
    REFLECT = "REFLECT"
    ESCALATE = "ESCALATE"
    HALT_SUCCESS = "HALT_SUCCESS"
    HALT_FAILURE = "HALT_FAILURE"


class PlanTransitionType(str, Enum):
    NONE = "NONE"
    VERIFIED_STEP = "VERIFIED_STEP"
    REPLAN = "REPLAN"


class MemoryQualificationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    PROVISIONAL = "PROVISIONAL"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class MemoryCommitStatus(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    COMMITTED = "COMMITTED"
    ALREADY_COMMITTED = "ALREADY_COMMITTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ResourceBudget:
    limits: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", _vector(self.limits, "budget limits"))

    def can_cover(self, cost: Sequence[float]) -> bool:
        candidate = _vector(cost, "predicted cost")
        return all(required <= available for required, available in zip(candidate, self.limits))

    def subtract(self, cost: Sequence[float]) -> "ResourceBudget":
        candidate = _vector(cost, "reserved cost")
        if not self.can_cover(candidate):
            raise ValidationError("resource budget cannot cover reservation.")
        return ResourceBudget(tuple(a - c for a, c in zip(self.limits, candidate)))


@dataclass(frozen=True)
class ExpectedPostcondition:
    verifier: str
    expected: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.verifier, "verifier")
        if not isinstance(self.expected, Mapping):
            raise ValidationError("expected postcondition values must be a mapping.")
        object.__setattr__(self, "expected", _freeze(self.expected))


@dataclass(frozen=True)
class IntentContract:
    intent_id: str
    goal: str
    success_conditions: tuple[ExpectedPostcondition, ...]
    immutable_constraints: frozenset[str]
    delegated_authority: frozenset[str]
    initial_budget: ResourceBudget
    permitted_side_effects: frozenset[SideEffectClass] = field(
        default_factory=lambda: frozenset({SideEffectClass.OBSERVATIONAL})
    )
    intent_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id")
        _text(self.goal, "goal")
        if not self.success_conditions or not all(
            isinstance(value, ExpectedPostcondition) for value in self.success_conditions
        ):
            raise ValidationError("intent requires typed success conditions.")
        object.__setattr__(self, "success_conditions", tuple(self.success_conditions))
        object.__setattr__(self, "immutable_constraints", frozenset(self.immutable_constraints))
        object.__setattr__(self, "delegated_authority", frozenset(self.delegated_authority))
        object.__setattr__(
            self,
            "permitted_side_effects",
            frozenset(SideEffectClass(value) for value in self.permitted_side_effects),
        )
        if not all(isinstance(value, str) and value.strip() for value in self.delegated_authority):
            raise ValidationError("delegated authority must contain capability identifiers.")
        if not all(isinstance(value, str) and value.strip() for value in self.immutable_constraints):
            raise ValidationError("immutable constraints must contain non-empty identifiers.")
        identity = {
            "intent_id": self.intent_id,
            "goal": self.goal,
            "success_conditions": self.success_conditions,
            "immutable_constraints": self.immutable_constraints,
            "delegated_authority": self.delegated_authority,
            "initial_budget": self.initial_budget,
            "permitted_side_effects": self.permitted_side_effects,
        }
        object.__setattr__(self, "intent_digest", stable_digest(identity))


@dataclass(frozen=True)
class ActionProposal:
    intent_id: str
    plan_revision: str
    action_id: str
    tool_name: str
    parameters: Mapping[str, Any]
    expected_postconditions: tuple[ExpectedPostcondition, ...]
    predicted_cost: tuple[float, ...]
    side_effect_class: SideEffectClass
    required_capabilities: frozenset[str]
    provenance: str
    handshake_required: bool = False
    proposal_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in ((self.intent_id, "intent_id"), (self.plan_revision, "plan_revision"),
                            (self.action_id, "action_id"), (self.tool_name, "tool_name"),
                            (self.provenance, "provenance")):
            _text(value, name)
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("proposal parameters must be a mapping.")
        if not self.expected_postconditions:
            raise ValidationError("proposal requires expected postconditions.")
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "expected_postconditions", tuple(self.expected_postconditions))
        object.__setattr__(self, "predicted_cost", _vector(self.predicted_cost, "predicted cost"))
        object.__setattr__(self, "side_effect_class", SideEffectClass(self.side_effect_class))
        object.__setattr__(self, "required_capabilities", frozenset(self.required_capabilities))
        if not self.required_capabilities or not all(isinstance(v, str) and v for v in self.required_capabilities):
            raise ValidationError("proposal requires capability identifiers.")
        if not isinstance(self.handshake_required, bool):
            raise ValidationError("handshake_required must be Boolean.")
        identity = {
            "intent_id": self.intent_id, "plan_revision": self.plan_revision, "action_id": self.action_id,
            "tool_name": self.tool_name, "parameters": self.parameters,
            "postconditions": [(p.verifier, p.expected) for p in self.expected_postconditions],
            "predicted_cost": self.predicted_cost, "side_effect_class": self.side_effect_class,
            "required_capabilities": sorted(self.required_capabilities), "provenance": self.provenance,
            "handshake_required": self.handshake_required,
        }
        object.__setattr__(self, "proposal_digest", stable_digest(identity))


@dataclass(frozen=True)
class ResourceReservation:
    reservation_id: str
    proposal_digest: str
    reserved_cost: tuple[float, ...]
    released: bool = False

    def __post_init__(self) -> None:
        _text(self.reservation_id, "reservation_id")
        _text(self.proposal_digest, "proposal_digest")
        object.__setattr__(self, "reserved_cost", _vector(self.reserved_cost, "reserved cost"))


@dataclass(frozen=True)
class AdmissionVerdict:
    decision: AdmissionDecision
    intent_id: str
    intent_digest: str
    plan_revision: str
    proposal_digest: str
    state_fingerprint: str
    approved_parameters_digest: str
    reservation: ResourceReservation | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", AdmissionDecision(self.decision))
        for value, name in (
            (self.intent_id, "intent_id"),
            (self.intent_digest, "intent_digest"),
            (self.plan_revision, "plan_revision"),
            (self.proposal_digest, "proposal_digest"),
            (self.state_fingerprint, "state_fingerprint"),
            (self.approved_parameters_digest, "approved_parameters_digest"),
        ):
            _text(value, name)
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.decision is AdmissionDecision.ADMIT and self.reservation is None:
            raise ValidationError("ADMIT requires a resource reservation.")


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    proposal_digest: str
    admission_digest: str
    state: ExecutionState
    tool_name: str
    success_reported: bool | None
    output: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    observed_cost: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.execution_id, "execution_id"), (self.proposal_digest, "proposal_digest"),
                            (self.admission_digest, "admission_digest"), (self.tool_name, "tool_name")):
            _text(value, name)
        object.__setattr__(self, "state", ExecutionState(self.state))
        if self.success_reported is not None and not isinstance(self.success_reported, bool):
            raise ValidationError("success_reported must be Boolean or None.")
        object.__setattr__(self, "data", _freeze(self.data))
        if self.observed_cost is not None:
            object.__setattr__(self, "observed_cost", _vector(self.observed_cost, "observed cost"))


@dataclass(frozen=True)
class Observation:
    observation_id: str
    execution_id: str
    evidence: Mapping[str, Any]
    complete: bool

    def __post_init__(self) -> None:
        _text(self.observation_id, "observation_id")
        _text(self.execution_id, "execution_id")
        if not isinstance(self.complete, bool):
            raise ValidationError("observation complete must be Boolean.")
        object.__setattr__(self, "evidence", _freeze(self.evidence))


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    proposal_digest: str
    observation_id: str | None
    checked_postconditions: int
    satisfied_postconditions: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", VerificationStatus(self.status))
        _text(self.proposal_digest, "proposal_digest")
        if isinstance(self.checked_postconditions, bool) or self.checked_postconditions < 0:
            raise ValidationError("checked_postconditions must be non-negative.")
        if self.satisfied_postconditions < 0 or self.satisfied_postconditions > self.checked_postconditions:
            raise ValidationError("satisfied_postconditions is outside its domain.")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))


@dataclass(frozen=True)
class ProvisionalMemoryCandidate:
    """Immutable evidence proposal; construction has no persistence authority."""

    intent_digest: str
    plan_revision: str
    proposal_digest: str
    execution_id: str
    observation_id: str
    verification_digest: str
    candidate_key: str
    evidence: Mapping[str, Any]
    tags: frozenset[str]
    policy_digest: str
    candidate_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.intent_digest, "intent_digest"), (self.plan_revision, "plan_revision"),
            (self.proposal_digest, "proposal_digest"), (self.execution_id, "execution_id"),
            (self.observation_id, "observation_id"),
            (self.verification_digest, "verification_digest"),
            (self.candidate_key, "candidate_key"), (self.policy_digest, "policy_digest"),
        ):
            _text(value, name)
        object.__setattr__(self, "evidence", _freeze(self.evidence))
        object.__setattr__(self, "tags", frozenset(_text(v, "memory tag") for v in self.tags))
        identity = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "candidate_digest"}
        object.__setattr__(self, "candidate_digest", stable_digest(identity))


@dataclass(frozen=True)
class MemoryQualificationResult:
    status: MemoryQualificationStatus
    candidate_digest: str
    reason_codes: tuple[str, ...]
    temporal_evidence_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MemoryQualificationStatus(self.status))
        _text(self.candidate_digest, "candidate_digest")
        if self.temporal_evidence_digest is not None:
            _text(self.temporal_evidence_digest, "temporal_evidence_digest")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))


@dataclass(frozen=True)
class MemoryCommitResult:
    status: MemoryCommitStatus
    candidate_digest: str | None
    target_key: str | None
    memory_state_digest: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MemoryCommitStatus(self.status))
        if self.candidate_digest is not None:
            _text(self.candidate_digest, "candidate_digest")
        if self.target_key is not None:
            _text(self.target_key, "target_key")
        _text(self.memory_state_digest, "memory_state_digest")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        has_identity = self.candidate_digest is not None and self.target_key is not None
        if self.status is MemoryCommitStatus.NOT_ATTEMPTED:
            if self.candidate_digest is not None or self.target_key is not None:
                raise ValidationError("an unattempted memory commit cannot carry commit identity.")
        elif not has_identity:
            raise ValidationError("an attempted memory commit requires candidate and target identity.")


@dataclass(frozen=True)
class MemoryStageEvidence:
    candidate: ProvisionalMemoryCandidate | None
    qualification: MemoryQualificationResult | None
    policy_admissible: bool | None
    final_verdict: str
    commit: MemoryCommitResult
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.final_verdict, "final memory verdict")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.qualification is not None:
            if self.candidate is None or self.qualification.candidate_digest != self.candidate.candidate_digest:
                raise ValidationError("memory qualification candidate mismatch.")
        if self.commit.candidate_digest is not None:
            if self.candidate is None or self.commit.candidate_digest != self.candidate.candidate_digest:
                raise ValidationError("memory commit candidate mismatch.")
            if self.commit.target_key != self.candidate.candidate_key:
                raise ValidationError("memory commit target mismatch.")
        if self.candidate is None:
            if self.qualification is not None or self.policy_admissible is not None:
                raise ValidationError("candidate-free memory evidence cannot be qualified or policy evaluated.")
            if self.final_verdict != "NO_CANDIDATE" or self.commit.status is not MemoryCommitStatus.NOT_ATTEMPTED:
                raise ValidationError("candidate-free memory evidence must be an unattempted NO_CANDIDATE state.")
            return

        if self.qualification is None:
            raise ValidationError("a memory candidate requires a qualification result.")
        status = self.qualification.status
        commit_status = self.commit.status
        valid_states = {
            "VERIFICATION_REJECTED": (MemoryQualificationStatus.REJECTED, None,
                                      {MemoryCommitStatus.NOT_ATTEMPTED}),
            "PROVISIONAL": (MemoryQualificationStatus.PROVISIONAL, None,
                            {MemoryCommitStatus.NOT_ATTEMPTED}),
            "REJECTED": (MemoryQualificationStatus.REJECTED, None,
                         {MemoryCommitStatus.NOT_ATTEMPTED}),
            "UNKNOWN": (MemoryQualificationStatus.UNKNOWN, None,
                        {MemoryCommitStatus.NOT_ATTEMPTED}),
            "POLICY_EVIDENCE_UNKNOWN": (MemoryQualificationStatus.QUALIFIED, None,
                                        {MemoryCommitStatus.NOT_ATTEMPTED}),
            "POLICY_REJECTED": (MemoryQualificationStatus.QUALIFIED, False,
                                {MemoryCommitStatus.NOT_ATTEMPTED}),
            "COMMIT_ACCEPTED": (MemoryQualificationStatus.QUALIFIED, True,
                                {MemoryCommitStatus.COMMITTED, MemoryCommitStatus.ALREADY_COMMITTED}),
            "COMMIT_FAILED": (MemoryQualificationStatus.QUALIFIED, True,
                              {MemoryCommitStatus.FAILED}),
        }
        expected = valid_states.get(self.final_verdict)
        if expected is None:
            raise ValidationError("unknown final memory verdict.")
        expected_qualification, expected_policy, allowed_commits = expected
        if (status is not expected_qualification or self.policy_admissible is not expected_policy
                or commit_status not in allowed_commits):
            raise ValidationError("memory evidence fields encode a contradictory authoritative state.")


@dataclass(frozen=True)
class PlanTransitionEvidence:
    prior_plan_revision: str
    proposal_digest: str
    action_id: str
    transition: PlanTransitionType
    resulting_plan_revision: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in ((self.prior_plan_revision, "prior plan revision"),
                            (self.proposal_digest, "proposal digest"),
                            (self.action_id, "action id"),
                            (self.resulting_plan_revision, "resulting plan revision")):
            _text(value, name)
        object.__setattr__(self, "transition", PlanTransitionType(self.transition))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.transition is PlanTransitionType.NONE and self.prior_plan_revision != self.resulting_plan_revision:
            raise ValidationError("no-mutation plan evidence cannot change revision.")


@dataclass(frozen=True)
class IntentSatisfaction:
    status: VerificationStatus
    intent_digest: str
    observation_id: str | None
    checked_conditions: int
    satisfied_conditions: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", VerificationStatus(self.status))
        _text(self.intent_digest, "intent_digest")
        if isinstance(self.checked_conditions, bool) or self.checked_conditions < 0:
            raise ValidationError("checked_conditions must be non-negative.")
        if self.satisfied_conditions < 0 or self.satisfied_conditions > self.checked_conditions:
            raise ValidationError("satisfied_conditions is outside its domain.")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))


@dataclass(frozen=True)
class AgenticResidual:
    proposal_digest: str
    goal: tuple[str, ...] = ()
    resource: tuple[float, ...] = (0.0,) * 8
    policy_authority: tuple[str, ...] = ()
    stability: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.proposal_digest, "proposal_digest")
        object.__setattr__(self, "goal", tuple(self.goal))
        object.__setattr__(self, "resource", _vector(self.resource, "resource residual"))
        object.__setattr__(self, "policy_authority", tuple(self.policy_authority))
        object.__setattr__(self, "stability", tuple(self.stability))
        object.__setattr__(self, "uncertainty", tuple(self.uncertainty))


@dataclass(frozen=True)
class ReflectionCandidate:
    proposal_digest: str
    activations: tuple[float, ...]
    reason: str

    def __post_init__(self) -> None:
        _text(self.proposal_digest, "proposal_digest")
        _text(self.reason, "reason")
        values = _vector(self.activations, "candidate activations", size=12)
        if any(value > 1.0 for value in values):
            raise ValidationError("candidate activations must be in [0, 1].")
        object.__setattr__(self, "activations", values)


@dataclass(frozen=True)
class ReflectionResult:
    attempted: bool
    committed: bool
    candidate: ReflectionCandidate | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attempted, bool) or not isinstance(self.committed, bool):
            raise ValidationError("reflection status fields must be Boolean.")
        if self.committed and not self.attempted:
            raise ValidationError("committed reflection must have been attempted.")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))


@dataclass(frozen=True)
class ClosureResult:
    decision: ClosureDecision
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ClosureDecision(self.decision))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))


@dataclass(frozen=True)
class ResourceReconciliation:
    reservation: ResourceReservation | None
    predicted_cost: tuple[float, ...]
    observed_cost: tuple[float, ...] | None
    remaining_budget: ResourceBudget
    hard_budget_breach: bool = False
    budget_before: ResourceBudget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicted_cost", _vector(self.predicted_cost, "predicted cost"))
        if self.observed_cost is not None:
            object.__setattr__(self, "observed_cost", _vector(self.observed_cost, "observed cost"))
        if not isinstance(self.hard_budget_breach, bool):
            raise ValidationError("hard_budget_breach must be Boolean.")


@dataclass(frozen=True)
class GovernedFeedbackContext:
    """Immutable reconciled information carried beyond a completed cycle."""

    intent_id: str
    intent_digest: str
    source_cycle_index: int
    source_receipt_digest: str
    resulting_state_fingerprint: str
    residual: AgenticResidual
    memory_state_digest: str
    remaining_budget: ResourceBudget
    plan_transition: PlanTransitionEvidence
    resulting_plan_revision: str
    prior_closure: ClosureResult
    feedback_context_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.intent_id, "intent_id"),
            (self.intent_digest, "intent_digest"),
            (self.source_receipt_digest, "source_receipt_digest"),
            (self.resulting_state_fingerprint, "resulting_state_fingerprint"),
            (self.memory_state_digest, "memory_state_digest"),
            (self.resulting_plan_revision, "resulting_plan_revision"),
        ):
            _text(value, name)
        if (
            isinstance(self.source_cycle_index, bool)
            or not isinstance(self.source_cycle_index, int)
            or self.source_cycle_index < 0
        ):
            raise ValidationError("source_cycle_index must be a non-negative integer.")
        if not isinstance(self.residual, AgenticResidual):
            raise ValidationError("feedback residual must be typed evidence.")
        if not isinstance(self.remaining_budget, ResourceBudget):
            raise ValidationError("feedback remaining budget must be typed evidence.")
        if not isinstance(self.plan_transition, PlanTransitionEvidence):
            raise ValidationError("feedback plan transition must be typed evidence.")
        if not isinstance(self.prior_closure, ClosureResult):
            raise ValidationError("feedback closure must be typed evidence.")
        if self.plan_transition.resulting_plan_revision != self.resulting_plan_revision:
            raise ValidationError("feedback resulting plan revision contradicts its transition.")
        identity = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "feedback_context_digest"
        }
        object.__setattr__(self, "feedback_context_digest", stable_digest(identity))


@dataclass(frozen=True)
class AgenticActionReceipt:
    intent_id: str
    plan_revision: str
    proposal: ActionProposal
    initial_state_fingerprint: str
    initial_memory_state_digest: str
    admission: AdmissionVerdict
    reconciliation: ResourceReconciliation
    execution: ExecutionRecord
    observation: Observation | None
    verification: VerificationResult
    intent_satisfaction: IntentSatisfaction
    residual: AgenticResidual
    reflection: ReflectionResult
    resulting_state_fingerprint: str
    closure: ClosureResult
    durable_memory_attempted: bool = False
    memory: MemoryStageEvidence | None = None
    plan_transition: PlanTransitionEvidence | None = None
    plan_transition_finalized: bool = False
    incoming_feedback_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.durable_memory_attempted, bool):
            raise ValidationError("durable_memory_attempted must be Boolean.")
        if not isinstance(self.plan_transition_finalized, bool):
            raise ValidationError("plan_transition_finalized must be Boolean.")
        if self.plan_transition_finalized and self.plan_transition is None:
            raise ValidationError("a finalized plan transition requires typed evidence.")
        if self.incoming_feedback_digest is not None:
            _text(self.incoming_feedback_digest, "incoming_feedback_digest")
        _text(self.initial_memory_state_digest, "initial_memory_state_digest")
        if self.intent_id != self.proposal.intent_id or self.intent_id != self.admission.intent_id:
            raise ValidationError("receipt intent provenance mismatch.")
        if self.plan_revision != self.proposal.plan_revision or self.plan_revision != self.admission.plan_revision:
            raise ValidationError("receipt plan provenance mismatch.")
        digest = self.proposal.proposal_digest
        if self.admission.proposal_digest != digest or self.execution.proposal_digest != digest:
            raise ValidationError("receipt proposal provenance mismatch.")
        if self.execution.admission_digest != admission_binding_digest(self.admission):
            raise ValidationError("receipt execution admission binding mismatch.")
        if self.admission.state_fingerprint != self.initial_state_fingerprint:
            raise ValidationError("receipt initial state provenance mismatch.")
        if self.observation is not None and self.observation.execution_id != self.execution.execution_id:
            raise ValidationError("receipt observation provenance mismatch.")
        expected_observation = self.observation.observation_id if self.observation else None
        if self.verification.observation_id != expected_observation:
            raise ValidationError("receipt verification observation mismatch.")
        if self.verification.proposal_digest != digest or self.residual.proposal_digest != digest:
            raise ValidationError("receipt verification/residual provenance mismatch.")
        if self.intent_satisfaction.intent_digest != self.admission.intent_digest:
            raise ValidationError("receipt intent satisfaction provenance mismatch.")
        if self.intent_satisfaction.observation_id != expected_observation:
            raise ValidationError("receipt intent observation mismatch.")
        if self.reflection.candidate is not None and self.reflection.candidate.proposal_digest != digest:
            raise ValidationError("receipt reflection provenance mismatch.")
        if self.memory is not None and self.memory.candidate is not None:
            candidate = self.memory.candidate
            if (candidate.intent_digest != self.admission.intent_digest
                    or candidate.plan_revision != self.plan_revision
                    or candidate.proposal_digest != digest
                    or candidate.execution_id != self.execution.execution_id
                    or self.observation is None
                    or candidate.observation_id != self.observation.observation_id
                    or candidate.verification_digest != stable_digest(self.verification)):
                raise ValidationError("receipt memory provenance mismatch.")
        if self.plan_transition is not None:
            transition = self.plan_transition
            if (transition.prior_plan_revision != self.plan_revision
                    or transition.proposal_digest != digest
                    or transition.action_id != self.proposal.action_id):
                raise ValidationError("receipt plan transition provenance mismatch.")

    def canonical(self) -> str:
        """Semantic receipt serialization, intentionally excluding timing metadata."""
        return json.dumps(_plain(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TransitionSignature:
    """Deterministic evidence binding both sides and the path of one governed successor."""

    intent_id: str
    intent_digest: str
    cycle_index: int
    source_state_fingerprint: str
    source_memory_state_digest: str
    source_budget: ResourceBudget
    source_plan_revision: str
    source_feedback_digest: str | None
    proposal_digest: str
    action_id: str
    finalized_receipt_digest: str
    resulting_feedback_digest: str
    resulting_state_fingerprint: str
    resulting_memory_state_digest: str
    resulting_budget: ResourceBudget
    resulting_plan_revision: str
    verification: VerificationResult
    verification_status: VerificationStatus
    verification_digest: str
    residual: AgenticResidual
    residual_digest: str
    memory_outcome: str
    memory_commit_status: MemoryCommitStatus
    memory_evidence: MemoryStageEvidence
    memory_evidence_digest: str
    plan_transition: PlanTransitionEvidence
    plan_transition_digest: str
    closure: ClosureResult
    closure_digest: str
    state_changed: bool
    memory_changed: bool
    plan_changed: bool
    resource_changed: bool
    transition_signature_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.intent_id, "intent_id"), (self.source_plan_revision, "source_plan_revision"),
            (self.action_id, "action_id"), (self.resulting_plan_revision, "resulting_plan_revision"),
            (self.memory_outcome, "memory_outcome"),
        ):
            _text(value, name)
        for value, name in (
            (self.intent_digest, "intent_digest"),
            (self.source_state_fingerprint, "source_state_fingerprint"),
            (self.source_memory_state_digest, "source_memory_state_digest"),
            (self.proposal_digest, "proposal_digest"),
            (self.finalized_receipt_digest, "finalized_receipt_digest"),
            (self.resulting_feedback_digest, "resulting_feedback_digest"),
            (self.resulting_state_fingerprint, "resulting_state_fingerprint"),
            (self.resulting_memory_state_digest, "resulting_memory_state_digest"),
            (self.verification_digest, "verification_digest"),
            (self.residual_digest, "residual_digest"),
            (self.memory_evidence_digest, "memory_evidence_digest"),
            (self.plan_transition_digest, "plan_transition_digest"),
            (self.closure_digest, "closure_digest"),
        ):
            _digest(value, name)
        if self.source_feedback_digest is not None:
            _digest(self.source_feedback_digest, "source_feedback_digest")
        if isinstance(self.cycle_index, bool) or not isinstance(self.cycle_index, int) or self.cycle_index < 0:
            raise ValidationError("cycle_index must be a non-negative integer.")
        if self.cycle_index == 0 and self.source_feedback_digest is not None:
            raise ValidationError("the initial transition cannot claim predecessor feedback.")
        if self.cycle_index > 0 and self.source_feedback_digest is None:
            raise ValidationError("a recurrent transition requires predecessor feedback.")
        if not isinstance(self.source_budget, ResourceBudget) or not isinstance(self.resulting_budget, ResourceBudget):
            raise ValidationError("transition budgets must be typed evidence.")
        object.__setattr__(self, "verification_status", VerificationStatus(self.verification_status))
        object.__setattr__(self, "memory_commit_status", MemoryCommitStatus(self.memory_commit_status))
        if not isinstance(self.verification, VerificationResult):
            raise ValidationError("transition verification must be typed evidence.")
        if (self.verification_status is not self.verification.status
                or self.verification_digest != stable_digest(self.verification)):
            raise ValidationError("transition verification identity mismatch.")
        if not isinstance(self.residual, AgenticResidual):
            raise ValidationError("transition residual must be typed evidence.")
        if self.residual_digest != stable_digest(self.residual):
            raise ValidationError("transition residual identity mismatch.")
        if not isinstance(self.memory_evidence, MemoryStageEvidence):
            raise ValidationError("transition memory outcome must be typed evidence.")
        if (self.memory_outcome != self.memory_evidence.final_verdict
                or self.memory_commit_status is not self.memory_evidence.commit.status
                or self.memory_evidence_digest != stable_digest(self.memory_evidence)):
            raise ValidationError("transition memory outcome identity mismatch.")
        if not isinstance(self.plan_transition, PlanTransitionEvidence):
            raise ValidationError("transition plan evidence must be typed evidence.")
        if self.plan_transition_digest != stable_digest(self.plan_transition):
            raise ValidationError("transition plan identity mismatch.")
        if not isinstance(self.closure, ClosureResult):
            raise ValidationError("transition closure must be typed evidence.")
        if self.closure_digest != stable_digest(self.closure):
            raise ValidationError("transition closure identity mismatch.")
        for name in ("state_changed", "memory_changed", "plan_changed", "resource_changed"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be Boolean.")
        expected_changes = (
            self.source_state_fingerprint != self.resulting_state_fingerprint,
            self.source_memory_state_digest != self.resulting_memory_state_digest,
            self.source_plan_revision != self.resulting_plan_revision,
            self.source_budget != self.resulting_budget,
        )
        if expected_changes != (self.state_changed, self.memory_changed, self.plan_changed, self.resource_changed):
            raise ValidationError("transition change descriptors contradict before/after evidence.")
        identity = {item.name: getattr(self, item.name) for item in fields(self)
                    if item.name != "transition_signature_digest"}
        object.__setattr__(self, "transition_signature_digest", stable_digest(identity))


@dataclass(frozen=True)
class GovernedRunResult:
    """Validated semantic trajectory produced by a bounded governed run."""

    intent_id: str
    intent_digest: str
    receipts: tuple[AgenticActionReceipt, ...]
    cycles_executed: int
    final_remaining_budget: ResourceBudget
    final_closure: ClosureResult | None
    termination_reason: str
    success: bool
    final_plan_revision: str
    feedback_contexts: tuple[GovernedFeedbackContext, ...] = ()
    transition_signatures: tuple[TransitionSignature, ...] = ()

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id")
        _text(self.intent_digest, "intent_digest")
        _text(self.termination_reason, "termination_reason")
        _text(self.final_plan_revision, "final_plan_revision")
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(self, "feedback_contexts", tuple(self.feedback_contexts))
        object.__setattr__(self, "transition_signatures", tuple(self.transition_signatures))
        if (
            isinstance(self.cycles_executed, bool)
            or not isinstance(self.cycles_executed, int)
            or self.cycles_executed < 0
        ):
            raise ValidationError("cycles_executed must be a non-negative integer.")
        if self.cycles_executed != len(self.receipts):
            raise ValidationError("run cycle count does not match its receipts.")
        if len(self.feedback_contexts) != len(self.receipts):
            raise ValidationError("run requires one feedback context per completed receipt.")
        if len(self.transition_signatures) != len(self.receipts):
            raise ValidationError("run requires one transition signature per completed receipt.")
        if not isinstance(self.success, bool):
            raise ValidationError("run success must be Boolean.")
        if self.receipts:
            if self.final_closure != self.receipts[-1].closure:
                raise ValidationError("final closure must be the last receipt closure.")
            if self.final_remaining_budget != self.receipts[-1].reconciliation.remaining_budget:
                raise ValidationError("final run budget does not match the trajectory.")
        expected_success = bool(
            self.final_closure and self.final_closure.decision is ClosureDecision.HALT_SUCCESS
        )
        if self.success is not expected_success:
            raise ValidationError("run success must derive from authoritative HALT_SUCCESS.")
        previous: AgenticActionReceipt | None = None
        for index, (receipt, feedback, signature) in enumerate(zip(
            self.receipts, self.feedback_contexts, self.transition_signatures
        )):
            if receipt.intent_id != self.intent_id or receipt.admission.intent_digest != self.intent_digest:
                raise ValidationError("run receipt intent provenance mismatch.")
            if feedback.intent_id != self.intent_id or feedback.intent_digest != self.intent_digest:
                raise ValidationError("run feedback intent provenance mismatch.")
            if feedback.source_cycle_index != index:
                raise ValidationError("run feedback cycle provenance mismatch.")
            if feedback.source_receipt_digest != stable_digest(receipt):
                raise ValidationError("run feedback receipt provenance mismatch.")
            if receipt.plan_transition is None:
                raise ValidationError("run receipts require plan transition evidence.")
            if (
                feedback.resulting_state_fingerprint != receipt.resulting_state_fingerprint
                or feedback.residual != receipt.residual
                or receipt.memory is None
                or feedback.memory_state_digest != receipt.memory.commit.memory_state_digest
                or feedback.remaining_budget != receipt.reconciliation.remaining_budget
                or feedback.plan_transition != receipt.plan_transition
                or feedback.resulting_plan_revision != receipt.plan_transition.resulting_plan_revision
                or feedback.prior_closure != receipt.closure
            ):
                raise ValidationError("run feedback does not match its finalized receipt.")
            if receipt.reconciliation.budget_before is None:
                raise ValidationError("run receipts require budget-before evidence.")
            if not receipt.plan_transition_finalized:
                raise ValidationError("run receipts require finalized plan transition evidence.")
            if (
                signature.intent_id != self.intent_id
                or signature.intent_digest != self.intent_digest
                or signature.cycle_index != index
                or signature.source_state_fingerprint != receipt.initial_state_fingerprint
                or signature.source_memory_state_digest != receipt.initial_memory_state_digest
                or signature.source_budget != receipt.reconciliation.budget_before
                or signature.source_plan_revision != receipt.plan_revision
                or signature.source_feedback_digest != receipt.incoming_feedback_digest
                or signature.proposal_digest != receipt.proposal.proposal_digest
                or signature.action_id != receipt.proposal.action_id
                or signature.finalized_receipt_digest != stable_digest(receipt)
                or signature.resulting_feedback_digest != feedback.feedback_context_digest
                or signature.resulting_state_fingerprint != feedback.resulting_state_fingerprint
                or signature.resulting_memory_state_digest != feedback.memory_state_digest
                or signature.resulting_budget != feedback.remaining_budget
                or signature.resulting_plan_revision != feedback.resulting_plan_revision
                or signature.verification != receipt.verification
                or signature.residual != receipt.residual
                or signature.memory_evidence != receipt.memory
                or signature.plan_transition != receipt.plan_transition
                or signature.closure != receipt.closure
            ):
                raise ValidationError("run transition signature provenance mismatch.")
            if previous is not None:
                if receipt.incoming_feedback_digest != self.feedback_contexts[index - 1].feedback_context_digest:
                    raise ValidationError("run incoming feedback continuity is broken.")
                if previous.resulting_state_fingerprint != receipt.initial_state_fingerprint:
                    raise ValidationError("run state fingerprint continuity is broken.")
                if previous.reconciliation.remaining_budget != receipt.reconciliation.budget_before:
                    raise ValidationError("run budget continuity is broken.")
                if previous.plan_transition is None:
                    raise ValidationError("run receipts require plan transition evidence.")
                if previous.plan_transition.resulting_plan_revision != receipt.plan_revision:
                    raise ValidationError("run plan revision continuity is broken.")
                prior_signature = self.transition_signatures[index - 1]
                if (
                    signature.source_feedback_digest != prior_signature.resulting_feedback_digest
                    or signature.source_state_fingerprint != prior_signature.resulting_state_fingerprint
                    or signature.source_memory_state_digest != prior_signature.resulting_memory_state_digest
                    or signature.source_budget != prior_signature.resulting_budget
                    or signature.source_plan_revision != prior_signature.resulting_plan_revision
                ):
                    raise ValidationError("run successor signature continuity is broken.")
            elif receipt.incoming_feedback_digest is not None:
                raise ValidationError("the first run receipt cannot have incoming feedback.")
            previous = receipt
        if self.receipts:
            transition = self.receipts[-1].plan_transition
            if transition is None or transition.resulting_plan_revision != self.final_plan_revision:
                raise ValidationError("final plan revision does not match trajectory evidence.")

    def canonical(self) -> str:
        """Deterministic semantic serialization without wall-clock diagnostics."""
        return json.dumps(_plain(self), sort_keys=True, separators=(",", ":"))


def _reason_codes(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not all(isinstance(value, str) and value for value in values):
        raise ValidationError("reason codes must be a sequence of non-empty strings.")
    return tuple(values)


def admission_binding_digest(admission: AdmissionVerdict) -> str:
    """Canonical execution binding for an exact admission verdict."""
    return stable_digest({
        "intent": admission.intent_digest,
        "proposal": admission.proposal_digest,
        "state": admission.state_fingerprint,
        "parameters": admission.approved_parameters_digest,
        "reservation": admission.reservation.reservation_id if admission.reservation else None,
        "decision": admission.decision,
    })
