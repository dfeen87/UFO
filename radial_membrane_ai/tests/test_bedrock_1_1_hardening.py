"""Regression proofs for BEDROCK 1.1 resource, outcome, and evidence boundaries."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic.admission import GovernedBudget, ProspectiveAgenticAdmission
from radial_membrane_ai.agentic.contracts import (
    ActionProposal, ClosureDecision, ExecutionState, ExpectedPostcondition,
    IntentContract, ResourceBudget, SideEffectClass, stable_digest,
)
from radial_membrane_ai.agentic.engine import AgenticEngine
from radial_membrane_ai.agentic.planner import Plan, PlanStep
from radial_membrane_ai.agentic.tools import BaseTool, ToolCallResult, ToolCapability, ToolRegistry
from radial_membrane_ai.exceptions import ValidationError


def proposal():
    return ActionProposal(
        "budget", "plan", "action", "SearchTool", {"query": "ufo"},
        (ExpectedPostcondition("fields_equal", {"query": "ufo"}),),
        (0.25,) * 8, SideEffectClass.OBSERVATIONAL, frozenset({"search"}), "test",
    )


def test_duplicate_release_cannot_expand_budget():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    budget.release(reservation)
    budget.release(reservation)
    assert budget.remaining.limits == (1.0,) * 8


def test_foreign_release_rejected_without_budget_change():
    owner = GovernedBudget(ResourceBudget((1.0,) * 8))
    foreign = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = owner.reserve(proposal())
    with pytest.raises(ValidationError):
        foreign.release(reservation)
    assert foreign.remaining.limits == (1.0,) * 8


def test_reconcile_then_release_cannot_refund_consumed_work():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    budget.reconcile(reservation, (0.1,) * 8)
    with pytest.raises(ValidationError):
        budget.release(reservation)
    assert budget.remaining.limits == (0.9,) * 8


class SideEffectThenError(BaseTool):
    name = "SideEffectThenError"
    description = "bounded simulated write followed by lost acknowledgement"

    def __init__(self, mode="exception"):
        self.calls = 0
        self.mode = mode

    def execute(self, params):
        self.calls += 1
        if self.mode == "exception":
            raise RuntimeError("acknowledgement lost after write")
        result = ToolCallResult(self.name, True, "ok", data={"ok": True}, cost_vector=[0.1] * 8)
        result.success = "untrusted truthy value"
        result.cost_vector = [-0.5] * 8
        return result


def scenario(mode="exception"):
    tool = SideEffectThenError(mode)
    registry = ToolRegistry()
    registry.register(tool, metadata=ToolCapability(
        "write", SideEffectClass.IRREVERSIBLE, (0.2,) * 8, False, False,
    ))
    intent = IntentContract(
        "write", "write", (ExpectedPostcondition("fields_equal", {"ok": True}),),
        frozenset(), frozenset({"write"}), ResourceBudget((1.0,) * 8),
        frozenset({SideEffectClass.IRREVERSIBLE}),
    )
    plan = Plan("write", [PlanStep(
        1, "write", tool.name, {},
        expected_postconditions=(ExpectedPostcondition("fields_equal", {"ok": True}),),
    )])
    return AgenticEngine(tool_registry=registry), tool, intent, plan


def test_exception_after_side_effect_escalates_without_refund_or_recurrence():
    engine, tool, intent, plan = scenario()
    result = engine.run_governed_intent(intent, plan=plan, max_cycles=1)
    receipt = result.receipts[0]
    assert receipt.execution.state is ExecutionState.OUTCOME_UNKNOWN
    assert receipt.observation is None
    assert receipt.closure.decision is ClosureDecision.ESCALATE
    assert result.final_remaining_budget.limits == (0.8,) * 8
    assert tool.calls == 1


def test_mutated_result_is_unknown_evidence_not_an_uncaught_post_execution_error():
    engine, tool, intent, plan = scenario("malformed")
    result = engine.run_governed_intent(intent, plan=plan, max_cycles=1)
    assert result.receipts[0].execution.state is ExecutionState.OUTCOME_UNKNOWN
    assert result.final_remaining_budget.limits == (0.8,) * 8
    assert tool.calls == 1


@pytest.mark.parametrize("values", [{1: "numeric"}, {1: "hidden", "1": "authoritative"}])
def test_ambiguous_evidence_keys_are_rejected(values):
    with pytest.raises(ValidationError):
        ExpectedPostcondition("fields_equal", values)


def test_digest_rejects_numeric_keys():
    with pytest.raises(ValidationError):
        stable_digest({1: "x"})


def test_admission_copy_does_not_authorize_an_extra_invocation():
    engine, tool, intent, plan = scenario()
    action = engine.planner.propose_step(plan, intent)
    budget = GovernedBudget(intent.initial_budget)
    admission = ProspectiveAgenticAdmission().evaluate(intent, action, engine.state_fingerprint(), budget)
    copied = replace(admission)
    with pytest.raises(ValidationError):
        engine.tool_registry.execute_admitted(action, copied, intent, engine.state_fingerprint())
    assert tool.calls == 0


def test_negative_observed_cost_cannot_expand_budget():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    with pytest.raises(ValidationError):
        budget.reconcile(reservation, (-0.1,) * 8)
    assert budget.remaining.limits == (0.75,) * 8


def test_nonfinite_raw_digest_rejected():
    with pytest.raises(ValidationError):
        stable_digest({"data": float("nan")})


def test_untyped_mutable_postcondition_cannot_gain_identity():
    class MutableCondition:
        verifier = "fields_equal"
        expected = {"query": "ufo"}

    with pytest.raises(ValidationError):
        replace(proposal(), expected_postconditions=(MutableCondition(),))


@pytest.mark.parametrize("operation", ["release", "reconcile"])
@pytest.mark.parametrize("forgery", ["copy", "amount", "released"])
def test_modified_reservations_cannot_gain_owner_authority(operation, forgery):
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    changed = {"amount": {"reserved_cost": (10.0,) * 8}, "released": {"released": True}}.get(forgery, {})
    forged = replace(reservation, **changed)
    with pytest.raises(ValidationError):
        if operation == "release":
            budget.release(forged)
        else:
            budget.reconcile(forged, (0.0,) * 8)
    assert budget.remaining.limits == (0.75,) * 8


@pytest.mark.parametrize("first,second", [("release", "reconcile"), ("reconcile", "reconcile")])
def test_terminal_resource_evidence_cannot_be_reordered(first, second):
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    if first == "release":
        budget.release(reservation)
    else:
        budget.reconcile(reservation, (0.1,) * 8)
    before = budget.remaining
    with pytest.raises(ValidationError):
        budget.reconcile(reservation, (0.0,) * 8)
    assert budget.remaining == before


def test_release_is_idempotent_with_both_original_and_returned_receipt():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    released = budget.release(reservation)
    assert released.released
    assert budget.release(reservation) is released
    assert budget.release(released) is released
    assert budget.remaining.limits == (1.0,) * 8


def test_repeated_identical_proposals_have_distinct_owned_reservations():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    first = budget.reserve(proposal())
    second = budget.reserve(proposal())
    assert first.reservation_id != second.reservation_id
    budget.reconcile(first, (0.1,) * 8)
    budget.release(second)
    assert budget.remaining.limits == (0.9,) * 8


@pytest.mark.parametrize("cost", [(), (0.0,) * 7, (True,) * 8, (float("nan"),) * 8, (float("inf"),) * 8])
def test_malformed_reconciliation_is_atomic_and_can_be_corrected(cost):
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    reservation = budget.reserve(proposal())
    before = budget.remaining
    with pytest.raises(ValidationError):
        budget.reconcile(reservation, cost)
    assert budget.remaining == before
    assert budget.reconcile(reservation, (0.1,) * 8) is False
    assert budget.remaining.limits == (0.9,) * 8


def test_many_tiny_releases_do_not_inflate_remaining_budget():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    small = replace(proposal(), predicted_cost=(1e-16,) * 8)
    consumed = budget.reserve(proposal())
    budget.reconcile(consumed, (0.1,) * 8)
    before = budget.remaining
    for _ in range(100):
        reservation = budget.reserve(small)
        budget.release(reservation)
        assert budget.remaining == before


def test_finite_extreme_costs_and_overrun_never_create_nonfinite_availability():
    budget = GovernedBudget(ResourceBudget((1e308,) * 8))
    large = replace(proposal(), predicted_cost=(5e307,) * 8)
    first = budget.reserve(large)
    second = budget.reserve(large)
    assert budget.reconcile(first, (1e308,) * 8)
    budget.release(second)
    assert budget.remaining.limits == (0.0,) * 8


def test_admission_is_one_shot_even_after_unknown_result():
    engine, tool, intent, plan = scenario()
    action = engine.planner.propose_step(plan, intent)
    budget = GovernedBudget(intent.initial_budget)
    admission = ProspectiveAgenticAdmission().evaluate(intent, action, engine.state_fingerprint(), budget)
    record, result = engine.tool_registry.execute_admitted(action, admission, intent, engine.state_fingerprint())
    assert record.state is ExecutionState.OUTCOME_UNKNOWN and result is None
    with pytest.raises(ValidationError):
        engine.tool_registry.execute_admitted(action, admission, intent, engine.state_fingerprint())
    with pytest.raises(ValidationError):
        budget.release(admission.reservation)
    assert tool.calls == 1


def test_released_admission_cannot_invoke_tool():
    engine, tool, intent, plan = scenario()
    action = engine.planner.propose_step(plan, intent)
    budget = GovernedBudget(intent.initial_budget)
    admission = ProspectiveAgenticAdmission().evaluate(intent, action, engine.state_fingerprint(), budget)
    budget.release(admission.reservation)
    with pytest.raises(ValidationError):
        engine.tool_registry.execute_admitted(action, admission, intent, engine.state_fingerprint())
    assert tool.calls == 0


@pytest.mark.parametrize("behavior", ["exception", "malformed", "bad-type", "identity", "bad-data", "known-failure"])
def test_fault_after_invocation_has_truthful_state_and_no_memory_promotion(behavior):
    from radial_membrane_ai.agentic.memory import GovernedMemoryConfig, semantic_memory_digest
    from radial_membrane_ai.semantic_memory import AdmissibilityContext, MemoryPolicy

    engine, tool, intent, plan = scenario()
    engine = AgenticEngine(tool_registry=engine.tool_registry, memory_config=GovernedMemoryConfig(
        MemoryPolicy(), AdmissibilityContext(0.8, False, 1.0, 0.0),
    ))
    for string in engine.engine.membrane.strings:
        string.activation = 0.8
    for _ in range(engine.engine.membrane.temporal_state.long_horizon):
        engine.engine.membrane.temporal_state.update_tick_history(0.1, 0.1, 0.1)
    before = semantic_memory_digest(engine.engine.semantic_memory)
    original = tool.execute

    def execute(params):
        if behavior in {"exception", "malformed"}:
            tool.mode = behavior
            return original(params)
        tool.calls += 1
        if behavior == "bad-type":
            return {"success": True}
        result = ToolCallResult(tool.name, behavior != "known-failure", "ok", data={"ok": False},
                                cost_vector=[0.1] * 8)
        if behavior == "identity":
            result.tool_name = "OtherTool"
        if behavior == "bad-data":
            result.data = {"object": object()}
        return result

    tool.execute = execute
    result = engine.run_governed_intent(intent, plan=plan, max_cycles=5)
    first = result.receipts[0]
    if behavior == "known-failure":
        assert first.execution.state is ExecutionState.FAILED
        assert first.execution.observed_cost == (0.1,) * 8
        assert first.closure.decision is ClosureDecision.REPLAN
    else:
        assert len(result.receipts) == 1
        assert first.execution.state is ExecutionState.OUTCOME_UNKNOWN
        assert first.execution.observed_cost is None
        assert first.observation is None
        assert first.closure.decision is ClosureDecision.ESCALATE
        assert first.reconciliation.remaining_budget.limits == (0.8,) * 8
    assert not result.success
    assert tool.calls == 1
    assert semantic_memory_digest(engine.engine.semantic_memory) == before


def test_registry_snapshots_nested_result_and_revalidates_boolean_success():
    tool = SideEffectThenError()
    result = ToolCallResult(tool.name, True, "ok", data={"nested": {"values": [1]}}, cost_vector=[0.1] * 8)
    tool.execute = lambda params: result
    registry = ToolRegistry()
    registry.register(tool)
    captured = registry.execute_tool(tool.name, {})
    result.data["nested"]["values"].append(2)
    result.cost_vector[0] = 10.0
    assert captured.data == {"nested": {"values": [1]}}
    assert captured.cost_vector == [0.1] * 8
    with pytest.raises(ValidationError):
        ToolCallResult(tool.name, "true", "ok")


def test_observation_uses_immutable_execution_snapshot_not_returned_result(monkeypatch):
    engine, tool, intent, plan = scenario()
    tool.execute = lambda params: ToolCallResult(tool.name, True, "ok", data={"ok": True}, cost_vector=[0.1] * 8)
    original = engine.tool_registry.execute_admitted

    def mutate_after_record(*args, **kwargs):
        record, result = original(*args, **kwargs)
        result.data["ok"] = False
        result.output = "changed"
        return record, result

    monkeypatch.setattr(engine.tool_registry, "execute_admitted", mutate_after_record)
    receipt = engine.run_governed_cycle(intent, plan=plan)
    assert receipt.observation.evidence["data"]["ok"] is True
    assert receipt.observation.evidence["output"] == "ok"
    assert receipt.execution.data["ok"] is True


def test_digest_and_nested_contract_identity_are_stable_for_valid_data():
    nested = {"ordered": {"b": [1, {"c": 2}], "a": 3}}
    condition = ExpectedPostcondition("fields_equal", nested)
    original = stable_digest(condition)
    nested["ordered"]["b"][1]["c"] = 99
    assert stable_digest(condition) == original
    with pytest.raises(TypeError):
        condition.expected["ordered"]["a"] = 4
    assert stable_digest({"a": 1, "b": [2]}) == stable_digest({"b": (2,), "a": 1})
    assert stable_digest({"a": 1, "b": [2]}) == "0855bfc20eb6cfaa4be7d6b510c63dd22997718bddadcb1a4915cec3a16145b9"


@pytest.mark.parametrize("values", [{"nested": {1: "bad"}}, {True: "bad"}, {None: "bad"}])
def test_mapping_key_rejection_is_recursive(values):
    with pytest.raises(ValidationError):
        ExpectedPostcondition("fields_equal", values)
    with pytest.raises(ValidationError):
        stable_digest(values)


@pytest.mark.parametrize("cancellation", [KeyboardInterrupt, SystemExit])
def test_cancellation_does_not_reopen_invocation_authority_or_allow_refund(cancellation):
    engine, tool, intent, plan = scenario()
    action = engine.planner.propose_step(plan, intent)
    budget = GovernedBudget(intent.initial_budget)
    admission = ProspectiveAgenticAdmission().evaluate(intent, action, engine.state_fingerprint(), budget)

    def cancel(params):
        tool.calls += 1
        raise cancellation()

    tool.execute = cancel
    with pytest.raises(cancellation):
        engine.tool_registry.execute_admitted(action, admission, intent, engine.state_fingerprint())
    with pytest.raises(ValidationError):
        budget.release(admission.reservation)
    with pytest.raises(ValidationError):
        engine.tool_registry.execute_admitted(action, admission, intent, engine.state_fingerprint())
    assert budget.remaining.limits == (0.8,) * 8
    assert tool.calls == 1


def test_repaired_resource_and_evidence_trajectory_replays_deterministically():
    from radial_membrane_ai.tests.test_agentic_v5_recurrence import intent, two_step_plan

    first = AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=two_step_plan())
    replay = AgenticEngine().run_governed_intent(intent(), max_cycles=3, plan=two_step_plan())
    assert first.canonical() == replay.canonical()
    assert len(first.receipts) == 2
    assert first.receipts[0].reconciliation.remaining_budget == first.receipts[1].reconciliation.budget_before


def test_runtime_ledger_does_not_change_public_budget_value_equality():
    budget = GovernedBudget(ResourceBudget((1.0,) * 8))
    budget.release(budget.reserve(proposal()))
    assert budget == GovernedBudget(ResourceBudget((1.0,) * 8))
