"""Pure construction and validation for governed successor evidence."""

from radial_membrane_ai.agentic.contracts import (
    AgenticActionReceipt,
    GovernedFeedbackContext,
    TransitionSignature,
    stable_digest,
)
from radial_membrane_ai.exceptions import ValidationError


def build_transition_signature(
    receipt: AgenticActionReceipt,
    feedback: GovernedFeedbackContext,
    cycle_index: int,
    *,
    source_feedback: GovernedFeedbackContext | None = None,
    prior_signature: TransitionSignature | None = None,
) -> TransitionSignature:
    """Validate an exact finalized transition and return non-authoritative evidence."""
    if receipt.memory is None or receipt.plan_transition is None or not receipt.plan_transition_finalized:
        raise ValidationError("a transition signature requires a finalized receipt.")
    if receipt.reconciliation.budget_before is None:
        raise ValidationError("a transition signature requires budget-before evidence.")
    if cycle_index == 0:
        if source_feedback is not None or prior_signature is not None or receipt.incoming_feedback_digest is not None:
            raise ValidationError("the initial transition cannot have predecessor evidence.")
    else:
        if source_feedback is None or prior_signature is None:
            raise ValidationError("a recurrent transition requires its immediate predecessor evidence.")
        if source_feedback.source_cycle_index != cycle_index - 1:
            raise ValidationError("successor predecessor cycle is not immediate.")
        if (
            prior_signature.cycle_index != cycle_index - 1
            or prior_signature.resulting_feedback_digest != source_feedback.feedback_context_digest
            or receipt.incoming_feedback_digest != source_feedback.feedback_context_digest
            or receipt.initial_state_fingerprint != source_feedback.resulting_state_fingerprint
            or receipt.initial_memory_state_digest != source_feedback.memory_state_digest
            or receipt.reconciliation.budget_before != source_feedback.remaining_budget
            or receipt.plan_revision != source_feedback.resulting_plan_revision
        ):
            raise ValidationError("successor predecessor provenance is stale or spliced.")

    if (
        feedback.source_cycle_index != cycle_index
        or feedback.source_receipt_digest != stable_digest(receipt)
        or feedback.intent_id != receipt.intent_id
        or feedback.intent_digest != receipt.admission.intent_digest
        or feedback.resulting_state_fingerprint != receipt.resulting_state_fingerprint
        or feedback.residual != receipt.residual
        or feedback.memory_state_digest != receipt.memory.commit.memory_state_digest
        or feedback.remaining_budget != receipt.reconciliation.remaining_budget
        or feedback.plan_transition != receipt.plan_transition
        or feedback.resulting_plan_revision != receipt.plan_transition.resulting_plan_revision
        or feedback.prior_closure != receipt.closure
    ):
        raise ValidationError("resulting feedback is not bound to the finalized receipt.")

    source_budget = receipt.reconciliation.budget_before
    memory = receipt.memory
    return TransitionSignature(
        receipt.intent_id,
        receipt.admission.intent_digest,
        cycle_index,
        receipt.initial_state_fingerprint,
        receipt.initial_memory_state_digest,
        source_budget,
        receipt.plan_revision,
        source_feedback.feedback_context_digest if source_feedback is not None else None,
        receipt.proposal.proposal_digest,
        receipt.proposal.action_id,
        stable_digest(receipt),
        feedback.feedback_context_digest,
        feedback.resulting_state_fingerprint,
        feedback.memory_state_digest,
        feedback.remaining_budget,
        feedback.resulting_plan_revision,
        receipt.verification,
        receipt.verification.status,
        stable_digest(receipt.verification),
        receipt.residual,
        stable_digest(receipt.residual),
        memory.final_verdict,
        memory.commit.status,
        memory,
        stable_digest(memory),
        receipt.plan_transition,
        stable_digest(receipt.plan_transition),
        receipt.closure,
        stable_digest(receipt.closure),
        receipt.initial_state_fingerprint != feedback.resulting_state_fingerprint,
        receipt.initial_memory_state_digest != feedback.memory_state_digest,
        receipt.plan_revision != feedback.resulting_plan_revision,
        source_budget != feedback.remaining_budget,
    )
