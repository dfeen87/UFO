"""Adversarial tests for bounded recurrent v5 governance."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    BaseTool,
    ClosureDecision,
    ExpectedPostcondition,
    IntentContract,
    Plan,
    PlanStep,
    ResourceBudget,
    SideEffectClass,
    ToolOutcomeUnknown,
    VerificationStatus,
)
from radial_membrane_ai.agentic.tools import ToolCapability
from radial_membrane_ai.exceptions import ValidationError


def intent(*, budget=(3.0,) * 8, authority=frozenset({"memory.read", "search"})):
    return IntentContract(
        "recurrent-intent", "research recurrence",
        (ExpectedPostcondition("fields_equal", {"query": "finish"}),),
        frozenset({"do-not-expand-authority"}), authority, ResourceBudget(budget),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def two_step_plan():
    return Plan("research recurrence", [
        PlanStep(1, "context", "MemoryRetrievalTool", {"type": "general", "tag": "context"}),
        PlanStep(2, "finish", "SearchTool", {"query": "finish"}),
    ])


def test_two_action_fresh_authority_and_unresolved_intermediate_intent():
    result = AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=two_step_plan())
    assert result.success and result.termination_reason == "INTENT_SATISFIED"
    assert len(result.receipts) == 2
    first, second = result.receipts
    assert first.verification.status is VerificationStatus.VERIFIED
    assert first.intent_satisfaction.status is not VerificationStatus.VERIFIED
    assert first.closure.decision is ClosureDecision.CONTINUE
    assert first.proposal.proposal_digest != second.proposal.proposal_digest
    assert first.admission.reservation != second.admission.reservation
    assert second.closure.decision is ClosureDecision.HALT_SUCCESS
    assert first.reconciliation.remaining_budget == second.reconciliation.budget_before


def test_shared_budget_blocks_second_action_without_execution(monkeypatch):
    engine = AgenticEngine()
    calls = 0
    original = engine.tool_registry.execute_admitted

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.tool_registry, "execute_admitted", counted)
    constrained = intent(budget=(0.45, 0.5, 0.5, 0.5, 0.2, 0.3, 0.0, 0.0))
    result = engine.run_governed_intent(constrained, max_cycles=3, plan=two_step_plan())
    assert calls == 1
    assert result.receipts[1].admission.decision.value == "BLOCK"
    assert result.receipts[1].execution.state.value == "NOT_EXECUTED"
    assert result.receipts[1].admission.reason_codes == ("BUDGET_INSUFFICIENT",)


def test_failed_verification_does_not_advance_step():
    plan = Plan("research recurrence", [PlanStep(
        1, "contradiction", "SearchTool", {"query": "actual"},
        expected_postconditions=(ExpectedPostcondition("fields_equal", {"query": "different"}),),
    )])
    result = AgenticEngine().run_governed_intent(intent(), max_cycles=1, plan=plan)
    assert result.receipts[0].verification.status is VerificationStatus.FAILED
    assert result.receipts[0].closure.decision is ClosureDecision.REPLAN
    assert not plan.steps[-1].completed


def test_replan_cannot_expand_intent_authority():
    governed_intent = intent(authority=frozenset({"search"}))
    plan = Plan("research recurrence", [PlanStep(
        1, "force replan", "SearchTool", {"query": "actual"},
        expected_postconditions=(ExpectedPostcondition("fields_equal", {"query": "different"}),),
    )])
    result = AgenticEngine().run_governed_intent(governed_intent, max_cycles=3, plan=plan)
    assert result.receipts[0].closure.decision is ClosureDecision.REPLAN
    assert result.receipts[1].proposal.tool_name == "MemoryRetrievalTool"
    assert "AUTHORITY_CAPABILITY_MISSING" in result.receipts[1].admission.reason_codes
    assert result.receipts[1].execution.state.value == "NOT_EXECUTED"
    assert result.intent_digest == governed_intent.intent_digest


class UnknownTool(BaseTool):
    name = "UnknownTool"
    description = "crosses a simulated irreversible boundary"

    def __init__(self):
        self.calls = 0

    def execute(self, params):
        self.calls += 1
        raise ToolOutcomeUnknown("remote acknowledgement lost")


def test_irreversible_outcome_unknown_escalates_once_and_consumes_reservation():
    engine = AgenticEngine()
    tool = UnknownTool()
    cost = (0.2,) * 8
    engine.tool_registry.register(tool, metadata=ToolCapability(
        "irreversible.write", SideEffectClass.IRREVERSIBLE, cost, False, False
    ))
    governed_intent = IntentContract(
        "unknown-intent", "unknown", (ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
        frozenset(), frozenset({"irreversible.write"}), ResourceBudget((1.0,) * 8),
        frozenset({SideEffectClass.IRREVERSIBLE}),
    )
    plan = Plan("unknown", [PlanStep(
        1, "write", tool.name, {},
        expected_postconditions=(ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
    )])
    result = engine.run_governed_intent(governed_intent, max_cycles=5, plan=plan)
    receipt = result.receipts[0]
    assert tool.calls == 1 and len(result.receipts) == 1
    assert receipt.execution.state.value == "OUTCOME_UNKNOWN"
    assert receipt.execution.success_reported is None and receipt.observation is None
    assert receipt.closure.decision is ClosureDecision.ESCALATE
    assert result.final_remaining_budget.limits == (0.8,) * 8


def test_handshake_exception_fails_closed_without_invocation():
    engine = AgenticEngine()
    tool = UnknownTool()
    engine.tool_registry.register(tool, metadata=ToolCapability(
        "guarded", SideEffectClass.OBSERVATIONAL, (0.1,) * 8, True, True, True
    ))
    governed_intent = IntentContract(
        "handshake", "guarded", (ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
        frozenset(), frozenset({"guarded"}), ResourceBudget((1.0,) * 8),
    )
    plan = Plan("guarded", [PlanStep(
        1, "guarded", tool.name, {},
        expected_postconditions=(ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
    )])

    def exploding():
        raise RuntimeError("handshake adapter failed")

    result = engine.run_governed_intent(governed_intent, max_cycles=2, plan=plan, handshake=exploding)
    assert tool.calls == 0
    assert "INVARIANT_HANDSHAKE_EXCEPTION" in result.receipts[0].admission.reason_codes
    assert result.receipts[0].closure.decision is ClosureDecision.HALT_FAILURE


def test_max_cycle_bound_and_deterministic_trajectory():
    first = AgenticEngine().run_governed_intent(intent(), max_cycles=1, plan=two_step_plan())
    second = AgenticEngine().run_governed_intent(intent(), max_cycles=1, plan=two_step_plan())
    assert first.termination_reason == "MAX_CYCLES_REACHED"
    assert len(first.receipts) == 1
    assert first.canonical() == second.canonical()
    with pytest.raises(ValueError):
        AgenticEngine().run_governed_intent(intent(), max_cycles=True)


def test_cross_receipt_forgery_is_rejected():
    result = AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=two_step_plan())
    forged_reconciliation = replace(
        result.receipts[1].reconciliation, budget_before=ResourceBudget((9.0,) * 8)
    )
    forged_second = replace(result.receipts[1], reconciliation=forged_reconciliation)
    with pytest.raises(ValidationError):
        replace(result, receipts=(result.receipts[0], forged_second))
