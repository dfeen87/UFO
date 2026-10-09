"""Prospective action admission and runtime-owned resource governance."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable
from weakref import ReferenceType, ref

from radial_membrane_ai.agentic.contracts import (
    ActionProposal,
    AdmissionDecision,
    AdmissionVerdict,
    IntentContract,
    ResourceBudget,
    ResourceReservation,
    stable_digest,
)
from radial_membrane_ai.exceptions import ValidationError


@dataclass
class _ReservationEntry:
    reservation: ResourceReservation
    state: str = "RESERVED"
    released_receipt: ResourceReservation | None = None


@dataclass
class GovernedBudget:
    """Single-owner in-process ledger; serialized reservations are only evidence."""

    remaining: ResourceBudget
    _initial: ResourceBudget = field(init=False, repr=False, compare=False)
    _spent: tuple[float, ...] = field(default=(0.0,) * 8, init=False, repr=False, compare=False)
    _entries: dict[int, _ReservationEntry] = field(default_factory=dict, init=False, repr=False, compare=False)
    _released: dict[int, _ReservationEntry] = field(default_factory=dict, init=False, repr=False, compare=False)
    _sequence: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.remaining, ResourceBudget):
            raise ValidationError("governed budget requires a typed resource budget.")
        self._initial = self.remaining

    def _entry(self, reservation: ResourceReservation) -> _ReservationEntry:
        entry = self._entries.get(id(reservation))
        if entry is None or entry.reservation is not reservation:
            raise ValidationError("reservation is foreign, copied, or forged.")
        return entry

    def _require_reserved(self, reservation: ResourceReservation) -> None:
        if self._entry(reservation).state != "RESERVED":
            raise ValidationError("reservation is no longer available for execution or release.")

    def _begin_execution(self, reservation: ResourceReservation) -> None:
        self._require_reserved(reservation)
        self._entry(reservation).state = "EXECUTING"

    def _available(self, *, excluding: ResourceReservation | None = None) -> ResourceBudget:
        held = [entry.reservation.reserved_cost for entry in self._entries.values()
                if entry.state in {"RESERVED", "EXECUTING"} and entry.reservation is not excluding]
        limits = []
        for index, initial in enumerate(self._initial.limits):
            try:
                used = math.fsum([self._spent[index], *(cost[index] for cost in held)])
            except OverflowError:
                used = math.inf
            limits.append(max(0.0, initial - used))
        return ResourceBudget(tuple(limits))

    def _settle(self, entry: _ReservationEntry, cost: tuple[float, ...]) -> None:
        # Saturate modeled expenditure on overrun; never refund from caller evidence.
        self._spent = tuple(min(initial, spent + observed)
                            for initial, spent, observed in zip(self._initial.limits, self._spent, cost))
        entry.state = "CONSUMED"
        self.remaining = self._available()

    def reserve(self, proposal: ActionProposal) -> ResourceReservation | None:
        if not self.remaining.can_cover(proposal.predicted_cost):
            return None
        self._sequence += 1
        reservation = ResourceReservation(
            reservation_id=f"reservation-{proposal.proposal_digest}-{self._sequence}",
            proposal_digest=proposal.proposal_digest,
            reserved_cost=proposal.predicted_cost,
        )
        self._entries[id(reservation)] = _ReservationEntry(reservation)
        self.remaining = self._available()
        return reservation

    def release(self, reservation: ResourceReservation) -> ResourceReservation:
        prior = self._released.get(id(reservation))
        if prior is not None and prior.released_receipt is reservation:
            return reservation
        entry = self._entry(reservation)
        if entry.state == "RELEASED":
            assert entry.released_receipt is not None
            return entry.released_receipt
        self._require_reserved(reservation)
        released = ResourceReservation(
            reservation.reservation_id, reservation.proposal_digest, reservation.reserved_cost, released=True
        )
        entry.state = "RELEASED"
        entry.released_receipt = released
        self._released[id(released)] = entry
        self.remaining = self._available()
        return released

    def consume_unknown(self, reservation: ResourceReservation) -> None:
        """Finalize an invoked action without inventing observed cost or a refund."""
        entry = self._entry(reservation)
        if entry.state != "EXECUTING":
            raise ValidationError("unknown consumption requires an executing reservation.")
        self._settle(entry, reservation.reserved_cost)

    def reconcile(self, reservation: ResourceReservation, observed_cost: tuple[float, ...]) -> bool:
        entry = self._entry(reservation)
        if entry.state not in {"RESERVED", "EXECUTING"}:
            raise ValidationError("reservation has already been settled.")
        # Validate every dimension before changing either lifecycle or accounting.
        observed = ResourceBudget(observed_cost).limits
        available = self._available(excluding=reservation).limits
        hard_breach = any(cost > limit for cost, limit in zip(observed, available))
        self._settle(entry, observed)
        return hard_breach


class ProspectiveAgenticAdmission:
    """Conjunctive admission coordinator; diagnostics never grant authority."""

    SUPPORTED_CONSTRAINTS = frozenset({"do-not-expand-authority"})
    # Live, process-local authority is separate from digestable/serializable evidence.
    # Weak verdict references retire unused permits when their receipts are discarded.
    _issued: dict[int, tuple[ReferenceType[AdmissionVerdict], GovernedBudget]] = {}

    def evaluate(
        self,
        intent: IntentContract,
        proposal: ActionProposal,
        state_fingerprint: str,
        budget: GovernedBudget,
        *,
        stability_allowed: bool = True,
        policy_allowed: bool = True,
        handshake: Callable[[], bool] | None = None,
    ) -> AdmissionVerdict:
        reasons: list[str] = []
        if proposal.intent_id != intent.intent_id:
            reasons.append("INTENT_IDENTITY_MISMATCH")
        if not proposal.required_capabilities.issubset(intent.delegated_authority):
            reasons.append("AUTHORITY_CAPABILITY_MISSING")
        unknown_constraints = intent.immutable_constraints - self.SUPPORTED_CONSTRAINTS
        if unknown_constraints:
            reasons.append("UNKNOWN_MANDATORY_CONSTRAINT")
        if (
            "do-not-expand-authority" in intent.immutable_constraints
            and not proposal.required_capabilities.issubset(intent.delegated_authority)
        ):
            reasons.append("IMMUTABLE_AUTHORITY_CONSTRAINT_FAILED")
        if proposal.side_effect_class not in intent.permitted_side_effects:
            reasons.append("SIDE_EFFECT_CLASS_NOT_PERMITTED")
        if not stability_allowed:
            reasons.append("STABILITY_GATE_FAILED")
        if not policy_allowed:
            reasons.append("POLICY_GATE_FAILED")
        if not budget.remaining.can_cover(proposal.predicted_cost):
            reasons.append("BUDGET_INSUFFICIENT")
        if proposal.handshake_required:
            try:
                handshake_passed = handshake is not None and handshake() is True
            except Exception:
                handshake_passed = False
                reasons.append("INVARIANT_HANDSHAKE_EXCEPTION")
            if not handshake_passed:
                reasons.append("INVARIANT_HANDSHAKE_FAILED")

        reservation = None
        decision = AdmissionDecision.BLOCK
        if not reasons:
            reservation = budget.reserve(proposal)
            if reservation is None:
                reasons.append("RESERVATION_FAILED")
            else:
                decision = AdmissionDecision.ADMIT
                reasons.append("ALL_MANDATORY_GATES_PASSED")
        elif reasons == ["STABILITY_GATE_FAILED"]:
            decision = AdmissionDecision.REPROJECT
        elif set(reasons).issubset({"POLICY_GATE_FAILED", "SIDE_EFFECT_CLASS_NOT_PERMITTED"}):
            decision = AdmissionDecision.CONSTRAIN

        verdict = AdmissionVerdict(
            decision=decision,
            intent_id=intent.intent_id,
            intent_digest=intent.intent_digest,
            plan_revision=proposal.plan_revision,
            proposal_digest=proposal.proposal_digest,
            state_fingerprint=state_fingerprint,
            approved_parameters_digest=stable_digest(proposal.parameters),
            reservation=reservation,
            reason_codes=tuple(reasons),
        )
        if decision is AdmissionDecision.ADMIT:
            key = id(verdict)
            self._issued[key] = (ref(verdict, lambda unused: self._issued.pop(key, None)), budget)
        return verdict

    @staticmethod
    def validate_execution_binding(
        intent: IntentContract,
        proposal: ActionProposal,
        verdict: AdmissionVerdict,
        current_state_fingerprint: str,
    ) -> None:
        """Fail closed at the execution boundary against stale or altered authority."""
        mismatched = (
            verdict.decision is not AdmissionDecision.ADMIT
            or verdict.intent_id != intent.intent_id
            or verdict.intent_digest != intent.intent_digest
            or verdict.plan_revision != proposal.plan_revision
            or verdict.proposal_digest != proposal.proposal_digest
            or verdict.approved_parameters_digest != stable_digest(proposal.parameters)
            or verdict.state_fingerprint != current_state_fingerprint
            or verdict.reservation is None
            or verdict.reservation.proposal_digest != proposal.proposal_digest
            or verdict.reservation.released
        )
        if mismatched:
            raise ValidationError("execution rejected: admission binding is missing, stale, or mismatched.")
        issued = ProspectiveAgenticAdmission._issued.get(id(verdict))
        if issued is None or issued[0]() is not verdict:
            raise ValidationError("execution rejected: admission was not issued here or was already claimed.")
        assert verdict.reservation is not None
        issued[1]._require_reserved(verdict.reservation)

    @classmethod
    def claim_execution(
        cls, intent: IntentContract, proposal: ActionProposal, verdict: AdmissionVerdict,
        current_state_fingerprint: str,
    ) -> GovernedBudget:
        """Consume exactly one runtime-issued permit before invoking a tool."""
        cls.validate_execution_binding(intent, proposal, verdict, current_state_fingerprint)
        _, budget = cls._issued.pop(id(verdict))
        assert verdict.reservation is not None
        budget._begin_execution(verdict.reservation)
        return budget
