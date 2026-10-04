"""Prospective action admission and runtime-owned resource governance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from radial_membrane_ai.agentic.contracts import (
    ActionProposal, AdmissionDecision, AdmissionVerdict, IntentContract,
    ResourceBudget, ResourceReservation, stable_digest,
)
from radial_membrane_ai.exceptions import ValidationError


@dataclass
class GovernedBudget:
    """The sole mutable owner of remaining budget and active reservations."""

    remaining: ResourceBudget

    def reserve(self, proposal: ActionProposal) -> ResourceReservation | None:
        if not self.remaining.can_cover(proposal.predicted_cost):
            return None
        reservation = ResourceReservation(
            reservation_id=f"reservation-{proposal.proposal_digest[:16]}",
            proposal_digest=proposal.proposal_digest,
            reserved_cost=proposal.predicted_cost,
        )
        self.remaining = self.remaining.subtract(proposal.predicted_cost)
        return reservation

    def release(self, reservation: ResourceReservation) -> ResourceReservation:
        if reservation.released:
            return reservation
        self.remaining = ResourceBudget(tuple(
            current + reserved for current, reserved in zip(self.remaining.limits, reservation.reserved_cost)
        ))
        return ResourceReservation(
            reservation.reservation_id, reservation.proposal_digest, reservation.reserved_cost, released=True
        )

    def reconcile(self, reservation: ResourceReservation, observed_cost: tuple[float, ...]) -> None:
        # Reservation is already debited. Refund unused dimensions; overruns debit
        # what remains without creating fictitious negative availability.
        adjusted = tuple(max(0.0, current + reserved - observed) for current, reserved, observed in zip(
            self.remaining.limits, reservation.reserved_cost, observed_cost
        ))
        self.remaining = ResourceBudget(adjusted)


class ProspectiveAgenticAdmission:
    """Conjunctive admission coordinator; diagnostics never grant authority."""

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
        if proposal.side_effect_class not in intent.permitted_side_effects:
            reasons.append("SIDE_EFFECT_CLASS_NOT_PERMITTED")
        if not stability_allowed:
            reasons.append("STABILITY_GATE_FAILED")
        if not policy_allowed:
            reasons.append("POLICY_GATE_FAILED")
        if not budget.remaining.can_cover(proposal.predicted_cost):
            reasons.append("BUDGET_INSUFFICIENT")
        if proposal.handshake_required:
            if handshake is None or handshake() is not True:
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

        return AdmissionVerdict(
            decision=decision,
            intent_id=intent.intent_id,
            plan_revision=proposal.plan_revision,
            proposal_digest=proposal.proposal_digest,
            state_fingerprint=state_fingerprint,
            approved_parameters_digest=stable_digest(proposal.parameters),
            reservation=reservation,
            reason_codes=tuple(reasons),
        )

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
