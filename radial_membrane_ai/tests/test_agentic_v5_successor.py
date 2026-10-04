"""Falsification tests for the governed successor evidence boundary."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    ClosureDecision,
    ExpectedPostcondition,
    IntentContract,
    Plan,
    PlanStep,
    ResourceBudget,
    SideEffectClass,
    build_transition_signature,
)
from radial_membrane_ai.agentic.contracts import stable_digest
from radial_membrane_ai.exceptions import ValidationError


def intent(intent_id="successor-intent"):
    return IntentContract(
        intent_id,
        "research recurrence",
        (ExpectedPostcondition("fields_equal", {"query": "finish"}),),
        frozenset({"do-not-expand-authority"}),
        frozenset({"memory.read", "search"}),
        ResourceBudget((3.0,) * 8),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def plan(first_query="context"):
    return Plan("research recurrence", [
        PlanStep(1, "context", "MemoryRetrievalTool", {"type": "general", "tag": first_query}),
        PlanStep(2, "finish", "SearchTool", {"query": "finish"}),
    ])


def result(first_query="context"):
    return AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=plan(first_query))


def test_signature_is_deterministic_complete_and_path_sensitive():
    first = result()
    replay = result()
    assert first.transition_signatures == replay.transition_signatures
    assert first.canonical() == replay.canonical()
    assert len(first.transition_signatures) == len(first.receipts) == len(first.feedback_contexts) == 2
    for index, (receipt, feedback, signature) in enumerate(zip(
        first.receipts, first.feedback_contexts, first.transition_signatures
    )):
        assert signature.cycle_index == index
        assert signature.finalized_receipt_digest == stable_digest(receipt)
        assert signature.resulting_feedback_digest == feedback.feedback_context_digest
        assert signature.verification == receipt.verification
        assert signature.residual == receipt.residual
        assert signature.memory_evidence == receipt.memory
        assert signature.plan_transition == receipt.plan_transition
        assert signature.closure == receipt.closure
    assert first.transition_signatures[0].source_feedback_digest is None
    assert (first.transition_signatures[1].source_feedback_digest
            == first.feedback_contexts[0].feedback_context_digest)


def test_initial_source_binds_authoritative_precycle_surfaces():
    run = result()
    receipt = run.receipts[0]
    signature = run.transition_signatures[0]
    assert signature.source_state_fingerprint == receipt.initial_state_fingerprint
    assert signature.source_memory_state_digest == receipt.initial_memory_state_digest
    assert signature.source_budget == receipt.reconciliation.budget_before
    assert signature.source_plan_revision == receipt.plan_revision


def test_recurrent_signature_has_exact_cross_cycle_continuity():
    run = result()
    prior, current = run.transition_signatures
    assert current.source_feedback_digest == prior.resulting_feedback_digest
    assert current.source_state_fingerprint == prior.resulting_state_fingerprint
    assert current.source_memory_state_digest == prior.resulting_memory_state_digest
    assert current.source_budget == prior.resulting_budget
    assert current.source_plan_revision == prior.resulting_plan_revision


def test_terminal_transition_is_auditable_but_not_authority():
    run = result()
    assert run.receipts[-1].closure.decision is ClosureDecision.HALT_SUCCESS
    assert run.transition_signatures[-1].closure == run.receipts[-1].closure
    # Admission is deliberately absent from the evidence contract.
    assert not hasattr(run.transition_signatures[-1], "authorized")
    assert not hasattr(run.transition_signatures[-1], "next_action_allowed")


def test_signature_construction_is_non_mutating():
    run = result()
    engine = AgenticEngine()
    before = engine.state_fingerprint()
    rebuilt = build_transition_signature(run.receipts[0], run.feedback_contexts[0], 0)
    assert rebuilt == run.transition_signatures[0]
    assert engine.state_fingerprint() == before


@pytest.mark.parametrize(
    "forge",
    [
        lambda r: replace(r, initial_state_fingerprint="wrong-state"),
        lambda r: replace(r, initial_memory_state_digest="wrong-memory"),
        lambda r: replace(r, reconciliation=replace(
            r.reconciliation, budget_before=ResourceBudget((99.0,) * 8))),
        lambda r: replace(r, plan_revision="wrong-plan", proposal=replace(r.proposal, plan_revision="wrong-plan"),
                          admission=replace(r.admission, plan_revision="wrong-plan")),
        lambda r: replace(r, incoming_feedback_digest="wrong-feedback"),
    ],
    ids=("state", "memory", "budget", "plan", "feedback"),
)
def test_recurrent_builder_rejects_forged_source_evidence(forge):
    run = result()
    with pytest.raises(ValidationError):
        build_transition_signature(
            forge(run.receipts[1]), run.feedback_contexts[1], 1,
            source_feedback=run.feedback_contexts[0], prior_signature=run.transition_signatures[0],
        )


@pytest.mark.parametrize(
    "forge",
    [
        lambda f: replace(f, source_receipt_digest="wrong-receipt"),
        lambda f: replace(f, resulting_state_fingerprint="wrong-state"),
        lambda f: replace(f, memory_state_digest="wrong-memory"),
        lambda f: replace(f, remaining_budget=ResourceBudget((99.0,) * 8)),
        lambda f: replace(f, residual=replace(f.residual, uncertainty=("forged",))),
        lambda f: replace(f, prior_closure=replace(f.prior_closure, reason_codes=("forged",))),
        lambda f: replace(f, plan_transition=replace(f.plan_transition, reason_codes=("forged",))),
    ],
    ids=("receipt", "state", "memory", "budget", "residual", "closure", "plan"),
)
def test_builder_rejects_forged_result_evidence(forge):
    run = result()
    with pytest.raises(ValidationError):
        build_transition_signature(run.receipts[0], forge(run.feedback_contexts[0]), 0)


def test_builder_rejects_provisional_receipt_and_skipped_predecessor():
    run = result()
    with pytest.raises(ValidationError, match="finalized"):
        build_transition_signature(
            replace(run.receipts[0], plan_transition_finalized=False), run.feedback_contexts[0], 0
        )
    with pytest.raises(ValidationError, match="immediate"):
        build_transition_signature(
            run.receipts[1], run.feedback_contexts[1], 2,
            source_feedback=run.feedback_contexts[0], prior_signature=run.transition_signatures[0],
        )


def test_run_rejects_cross_trajectory_splice_and_forged_outcomes():
    run = result()
    other = result("different-context")
    with pytest.raises(ValidationError):
        replace(run, transition_signatures=(other.transition_signatures[0],) + run.transition_signatures[1:])
    signature = run.transition_signatures[0]
    for forged in (
        replace(signature, verification=replace(signature.verification, reason_codes=("forged",)),
                verification_digest=stable_digest(replace(signature.verification, reason_codes=("forged",)))),
        replace(signature, residual=replace(signature.residual, uncertainty=("forged",)),
                residual_digest=stable_digest(replace(signature.residual, uncertainty=("forged",)))),
        replace(signature, closure=replace(signature.closure, reason_codes=("forged",)),
                closure_digest=stable_digest(replace(signature.closure, reason_codes=("forged",)))),
        replace(signature, plan_transition=replace(signature.plan_transition, reason_codes=("forged",)),
                plan_transition_digest=stable_digest(replace(signature.plan_transition, reason_codes=("forged",)))),
        replace(signature, memory_evidence=replace(signature.memory_evidence, reason_codes=("forged",)),
                memory_evidence_digest=stable_digest(replace(signature.memory_evidence, reason_codes=("forged",)))),
    ):
        with pytest.raises(ValidationError):
            replace(run, transition_signatures=(forged,) + run.transition_signatures[1:])


def test_destination_identity_does_not_collapse_distinct_paths():
    first = result("one")
    second = result("two")
    left = first.transition_signatures[0]
    right = second.transition_signatures[0]
    # Both observational paths leave the same authoritative runtime state, but their actions differ.
    assert left.resulting_state_fingerprint == right.resulting_state_fingerprint
    assert left.proposal_digest != right.proposal_digest
    assert left.transition_signature_digest != right.transition_signature_digest


def test_result_identity_still_changes_signature():
    run = result()
    signature = run.transition_signatures[0]
    different_result = stable_digest("different-result")
    changed = replace(
        signature,
        resulting_state_fingerprint=different_result,
        state_changed=signature.source_state_fingerprint != different_result,
    )
    assert changed.transition_signature_digest != signature.transition_signature_digest


def test_nonfinite_resources_fail_closed():
    run = result()
    with pytest.raises(ValidationError):
        replace(run.transition_signatures[0], source_budget=ResourceBudget((float("nan"),) * 8))
