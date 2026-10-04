"""Falsification tests for governed cross-cycle feedback provenance."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    ClosureDecision,
    ExpectedPostcondition,
    IntentContract,
    Plan,
    PlanStep,
    PlanTransitionEvidence,
    PlanTransitionType,
    ResourceBudget,
    SideEffectClass,
)
from radial_membrane_ai.agentic.admission import GovernedBudget
from radial_membrane_ai.agentic.contracts import (
    MemoryCommitResult,
    MemoryCommitStatus,
    MemoryQualificationResult,
    MemoryQualificationStatus,
    MemoryStageEvidence,
    ProvisionalMemoryCandidate,
    stable_digest,
)
from radial_membrane_ai.exceptions import ValidationError


def intent(*, intent_id="feedback-intent", budget=(3.0,) * 8):
    return IntentContract(
        intent_id,
        "research recurrence",
        (ExpectedPostcondition("fields_equal", {"query": "finish"}),),
        frozenset({"do-not-expand-authority"}),
        frozenset({"memory.read", "search"}),
        ResourceBudget(budget),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def two_step_plan():
    return Plan("research recurrence", [
        PlanStep(1, "context", "MemoryRetrievalTool", {"type": "general", "tag": "context"}),
        PlanStep(2, "finish", "SearchTool", {"query": "finish"}),
    ])


def recurrent_result():
    return AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=two_step_plan())


def test_feedback_is_deterministic_and_binds_finalized_trajectory():
    first = recurrent_result()
    second = recurrent_result()
    assert first.canonical() == second.canonical()
    assert len(first.feedback_contexts) == len(first.receipts) == 2
    assert first.receipts[0].incoming_feedback_digest is None
    assert (
        first.receipts[1].incoming_feedback_digest
        == first.feedback_contexts[0].feedback_context_digest
    )
    for index, (receipt, feedback) in enumerate(zip(first.receipts, first.feedback_contexts)):
        assert feedback.source_cycle_index == index
        assert feedback.source_receipt_digest == stable_digest(receipt)
        assert feedback.resulting_state_fingerprint == receipt.resulting_state_fingerprint
        assert feedback.residual == receipt.residual
        assert feedback.memory_state_digest == receipt.memory.commit.memory_state_digest
        assert feedback.remaining_budget == receipt.reconciliation.remaining_budget
        assert feedback.plan_transition == receipt.plan_transition
        assert feedback.resulting_plan_revision == receipt.plan_transition.resulting_plan_revision
        assert feedback.prior_closure == receipt.closure
    assert first.feedback_contexts[0].resulting_plan_revision != first.receipts[0].plan_revision


def test_replan_feedback_binds_post_replan_revision():
    governed_intent = intent()
    active_plan = Plan("research recurrence", [PlanStep(
        1,
        "contradiction",
        "SearchTool",
        {"query": "actual"},
        expected_postconditions=(ExpectedPostcondition("fields_equal", {"query": "different"}),),
    )])
    result = AgenticEngine().run_governed_intent(governed_intent, max_cycles=1, plan=active_plan)
    receipt = result.receipts[0]
    feedback = result.feedback_contexts[0]
    assert receipt.closure.decision is ClosureDecision.REPLAN
    assert receipt.plan_transition.transition is PlanTransitionType.REPLAN
    assert feedback.resulting_plan_revision == active_plan.revision
    assert feedback.resulting_plan_revision != receipt.plan_revision


def test_terminal_feedback_is_audit_evidence_not_continuation_authority():
    result = recurrent_result()
    assert result.receipts[-1].closure.decision is ClosureDecision.HALT_SUCCESS
    assert len(result.feedback_contexts) == len(result.receipts)
    engine = AgenticEngine()
    with pytest.raises(ValidationError, match="terminal closure"):
        engine._run_governed_transaction(
            intent(), two_step_plan(), GovernedBudget(result.final_remaining_budget),
            feedback_context=result.feedback_contexts[-1], source_receipt=result.receipts[-1],
            cycle_index=len(result.receipts),
        )


@pytest.mark.parametrize(
    "forge",
    [
        lambda context: replace(context, intent_id="other-intent"),
        lambda context: replace(context, intent_digest="other-digest"),
        lambda context: replace(context, source_cycle_index=9),
        lambda context: replace(context, source_receipt_digest="other-receipt"),
        lambda context: replace(context, resulting_state_fingerprint="other-state"),
        lambda context: replace(context, residual=replace(context.residual, uncertainty=("forged",))),
        lambda context: replace(context, memory_state_digest="other-memory"),
        lambda context: replace(context, remaining_budget=ResourceBudget((99.0,) * 8)),
        lambda context: replace(context, prior_closure=replace(context.prior_closure, reason_codes=("forged",))),
    ],
    ids=("intent", "intent-digest", "cycle", "receipt", "state", "residual", "memory", "budget-reset", "closure"),
)
def test_run_trajectory_rejects_forged_feedback(forge):
    result = recurrent_result()
    contexts = (forge(result.feedback_contexts[0]),) + result.feedback_contexts[1:]
    with pytest.raises(ValidationError):
        replace(result, feedback_contexts=contexts)


def test_run_trajectory_rejects_forged_plan_transition_and_revision():
    result = recurrent_result()
    context = result.feedback_contexts[0]
    transition = replace(
        context.plan_transition,
        transition=PlanTransitionType.REPLAN,
        reason_codes=("forged",),
    )
    forged = replace(
        context,
        plan_transition=transition,
        resulting_plan_revision=transition.resulting_plan_revision,
    )
    with pytest.raises(ValidationError):
        replace(result, feedback_contexts=(forged,) + result.feedback_contexts[1:])


def test_run_trajectory_rejects_skipped_or_duplicate_feedback():
    result = recurrent_result()
    with pytest.raises(ValidationError):
        replace(result, feedback_contexts=result.feedback_contexts[:1])
    forged_second = replace(
        result.receipts[1],
        incoming_feedback_digest=result.feedback_contexts[1].feedback_context_digest,
    )
    with pytest.raises(ValidationError):
        replace(result, receipts=(result.receipts[0], forged_second))


def finalized_first_cycle():
    engine = AgenticEngine()
    governed_intent = intent()
    active_plan = two_step_plan()
    budget = GovernedBudget(governed_intent.initial_budget)
    receipt = engine._run_governed_transaction(
        governed_intent, active_plan, budget, require_action_local=True
    )
    prior_revision = active_plan.revision
    engine.planner.commit_verified_step(active_plan, receipt)
    receipt = replace(receipt, plan_transition=PlanTransitionEvidence(
        prior_revision,
        receipt.proposal.proposal_digest,
        receipt.proposal.action_id,
        PlanTransitionType.VERIFIED_STEP,
        active_plan.revision,
        ("VERIFIED_STEP_COMMITTED",),
    ))
    feedback = engine._build_feedback_context(governed_intent, receipt, 0)
    return engine, governed_intent, active_plan, budget, receipt, feedback


@pytest.mark.parametrize("stale_surface", ["state", "memory", "budget", "plan"])
def test_stale_feedback_fails_before_proposal_admission_or_execution(monkeypatch, stale_surface):
    engine, governed_intent, active_plan, budget, receipt, feedback = finalized_first_cycle()
    calls = {"proposal": 0, "admission": 0, "execution": 0}

    def forbidden(*args, **kwargs):
        calls["proposal"] += 1
        raise AssertionError("proposal generation crossed stale feedback boundary")

    monkeypatch.setattr(engine.planner, "propose_step", forbidden)
    if stale_surface == "state":
        engine.engine.membrane.strings[0].activation += 0.01
    elif stale_surface == "memory":
        engine.engine.semantic_memory.governed_commits["forged"] = "mutation"
    elif stale_surface == "budget":
        budget.remaining = ResourceBudget(tuple(value + 1.0 for value in budget.remaining.limits))
    else:
        active_plan.steps.append(PlanStep(99, "forged", "SearchTool", {"query": "forged"}))
        active_plan.refresh_revision()
    before_budget = budget.remaining
    with pytest.raises(ValidationError, match="stale"):
        engine._run_governed_transaction(
            governed_intent, active_plan, budget, feedback_context=feedback,
            source_receipt=receipt, cycle_index=1, require_action_local=True,
        )
    assert calls == {"proposal": 0, "admission": 0, "execution": 0}
    assert budget.remaining == before_budget


def test_feedback_contract_rejects_hostile_structure():
    result = recurrent_result()
    context = result.feedback_contexts[0]
    with pytest.raises(ValidationError):
        replace(context, source_cycle_index=True)
    with pytest.raises(ValidationError):
        replace(context, source_receipt_digest="")
    with pytest.raises(ValidationError):
        replace(context, remaining_budget=ResourceBudget((float("nan"),) * 8))


def test_isolated_cycle_remains_feedback_free():
    receipt = AgenticEngine().run_governed_cycle(intent(), plan=two_step_plan())
    assert receipt.incoming_feedback_digest is None


def test_memory_stage_binds_candidate_to_authoritative_target():
    receipt = recurrent_result().receipts[0]
    candidate = ProvisionalMemoryCandidate(
        receipt.admission.intent_digest,
        receipt.plan_revision,
        receipt.proposal.proposal_digest,
        receipt.execution.execution_id,
        "observation-for-hostile-construction",
        stable_digest(receipt.verification),
        "declared-candidate-key",
        {"evidence": "qualified"},
        frozenset({"test"}),
        "policy-digest",
    )
    qualification = MemoryQualificationResult(
        MemoryQualificationStatus.QUALIFIED,
        candidate.candidate_digest,
        ("QUALIFIED",),
        "temporal-evidence",
    )
    valid = MemoryCommitResult(
        MemoryCommitStatus.COMMITTED,
        candidate.candidate_digest,
        candidate.candidate_key,
        receipt.memory.commit.memory_state_digest,
        ("COMMITTED",),
    )
    MemoryStageEvidence(candidate, qualification, True, "COMMIT_ACCEPTED", valid, ("ACCEPTED",))
    forged = replace(valid, target_key="different-valid-key")
    with pytest.raises(ValidationError, match="target mismatch"):
        MemoryStageEvidence(candidate, qualification, True, "COMMIT_ACCEPTED", forged, ("FORGED",))
