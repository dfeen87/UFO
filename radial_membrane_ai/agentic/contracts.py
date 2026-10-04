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

    def __post_init__(self) -> None:
        _text(self.intent_id, "intent_id")
        _text(self.goal, "goal")
        if not self.success_conditions or not all(isinstance(v, ExpectedPostcondition) for v in self.success_conditions):
            raise ValidationError("intent requires typed success conditions.")
        object.__setattr__(self, "success_conditions", tuple(self.success_conditions))
        object.__setattr__(self, "immutable_constraints", frozenset(self.immutable_constraints))
        object.__setattr__(self, "delegated_authority", frozenset(self.delegated_authority))
        object.__setattr__(self, "permitted_side_effects", frozenset(SideEffectClass(v) for v in self.permitted_side_effects))
        if not all(isinstance(v, str) and v for v in self.delegated_authority):
            raise ValidationError("delegated authority must contain capability identifiers.")


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
    plan_revision: str
    proposal_digest: str
    state_fingerprint: str
    approved_parameters_digest: str
    reservation: ResourceReservation | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", AdmissionDecision(self.decision))
        for value, name in ((self.intent_id, "intent_id"), (self.plan_revision, "plan_revision"),
                            (self.proposal_digest, "proposal_digest"), (self.state_fingerprint, "state_fingerprint"),
                            (self.approved_parameters_digest, "approved_parameters_digest")):
            _text(value, name)
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
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
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))


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


@dataclass(frozen=True)
class ClosureResult:
    decision: ClosureDecision
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ClosureDecision(self.decision))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))


@dataclass(frozen=True)
class ResourceReconciliation:
    reservation: ResourceReservation | None
    predicted_cost: tuple[float, ...]
    observed_cost: tuple[float, ...] | None
    remaining_budget: ResourceBudget


@dataclass(frozen=True)
class AgenticActionReceipt:
    intent_id: str
    plan_revision: str
    proposal: ActionProposal
    initial_state_fingerprint: str
    admission: AdmissionVerdict
    reconciliation: ResourceReconciliation
    execution: ExecutionRecord
    observation: Observation | None
    verification: VerificationResult
    residual: AgenticResidual
    reflection: ReflectionResult
    resulting_state_fingerprint: str
    closure: ClosureResult
    durable_memory_attempted: bool = False

    def canonical(self) -> str:
        """Semantic receipt serialization, intentionally excluding timing metadata."""
        return json.dumps(_plain(self), sort_keys=True, separators=(",", ":"))
