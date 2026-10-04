from dataclasses import replace

import pytest

from radial_membrane_ai.agentic.admission import GovernedBudget, ProspectiveAgenticAdmission
from radial_membrane_ai.agentic.contracts import (
    ActionProposal,
    AdmissionDecision,
    ClosureDecision,
    ExpectedPostcondition,
    IntentContract,
    ResourceBudget,
    SideEffectClass,
)
from radial_membrane_ai.agentic.engine import AgenticEngine
from radial_membrane_ai.agentic.planner import Plan, PlanStep
from radial_membrane_ai.agentic.tools import BaseTool, ToolCallResult, ToolCapability, ToolRegistry
from radial_membrane_ai.exceptions import ValidationError


def make_intent(*, conditions=None, constraints=frozenset({"do-not-expand-authority"}), budget=(2.0,) * 8):
    return IntentContract(
        "intent-hardening",
        "search ufo",
        conditions or (ExpectedPostcondition("fields_equal", {"query": "search ufo"}),),
        constraints,
        frozenset({"search"}),
        ResourceBudget(budget),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def make_search_proposal(engine, intent, conditions):
    plan = engine.planner.create_plan(intent.goal)
    base = engine.planner.propose_step(plan, intent)
    return plan, ActionProposal(
        base.intent_id,
        base.plan_revision,
        base.action_id,
        base.tool_name,
        base.parameters,
        conditions,
        base.predicted_cost,
        base.side_effect_class,
        base.required_capabilities,
        base.provenance,
    )


def test_easier_action_postcondition_cannot_claim_intent_success():
    intent = make_intent(conditions=(ExpectedPostcondition("fields_equal", {"required": "A"}),))
    engine = AgenticEngine()
    action_conditions = (ExpectedPostcondition("fields_equal", {"query": "search ufo"}),)
    plan, proposal = make_search_proposal(engine, intent, action_conditions)
    receipt = engine.run_governed_cycle(intent, plan=plan, proposal=proposal)
    assert receipt.verification.status.value == "VERIFIED"
    assert receipt.intent_satisfaction.status.value == "UNKNOWN"
    assert receipt.residual.goal == ("INTENT_EVIDENCE_INSUFFICIENT",)
    assert receipt.residual.uncertainty == ("INTENT_EVIDENCE_INSUFFICIENT",)
    assert receipt.closure.decision is not ClosureDecision.HALT_SUCCESS


@pytest.mark.parametrize(
    "changed",
    [
        make_intent(budget=(3.0,) * 8),
        make_intent(constraints=frozenset()),
        IntentContract(
            "intent-hardening",
            "search ufo",
            (ExpectedPostcondition("fields_equal", {"different": True}),),
            frozenset({"do-not-expand-authority"}),
            frozenset({"search"}),
            ResourceBudget((2.0,) * 8),
            frozenset({SideEffectClass.OBSERVATIONAL}),
        ),
    ],
    ids=["budget-changed", "constraint-removed", "success-condition-changed"],
)
def test_complete_intent_digest_is_bound_at_execution(changed):
    engine = AgenticEngine()
    original = make_intent()
    plan = engine.planner.create_plan(original.goal)
    proposal = engine.planner.propose_step(plan, original)
    verdict = ProspectiveAgenticAdmission().evaluate(
        original, proposal, engine.state_fingerprint(), GovernedBudget(original.initial_budget)
    )
    with pytest.raises(ValidationError, match="stale, or mismatched"):
        ProspectiveAgenticAdmission.validate_execution_binding(
            changed, proposal, verdict, engine.state_fingerprint()
        )


def test_unknown_constraint_blocks_and_supported_authority_constraint_is_enforced():
    engine = AgenticEngine()
    unknown = make_intent(constraints=frozenset({"unknown-mandatory-rule"}))
    receipt = engine.run_governed_cycle(unknown)
    assert receipt.admission.decision is AdmissionDecision.BLOCK
    assert "UNKNOWN_MANDATORY_CONSTRAINT" in receipt.admission.reason_codes

    supported = make_intent()
    plan = engine.planner.create_plan(supported.goal)
    proposal = engine.planner.propose_step(plan, supported)
    unauthorized = replace(proposal, required_capabilities=frozenset({"forbidden"}))
    verdict = ProspectiveAgenticAdmission().evaluate(
        supported, unauthorized, engine.state_fingerprint(), GovernedBudget(supported.initial_budget)
    )
    assert "IMMUTABLE_AUTHORITY_CONSTRAINT_FAILED" in verdict.reason_codes


def test_forged_receipt_provenance_is_rejected():
    receipt = AgenticEngine().run_governed_cycle(make_intent())
    forged_execution = replace(receipt.execution, proposal_digest="forged")
    with pytest.raises(ValidationError, match="proposal provenance"):
        replace(receipt, execution=forged_execution)
    forged_observation = replace(receipt.observation, execution_id="forged")
    with pytest.raises(ValidationError, match="observation provenance"):
        replace(receipt, observation=forged_observation)


class CustomTool(BaseTool):
    name = "CustomTool"
    description = "Test-only deterministic tool."

    def __init__(self, observed_cost=(0.2,) * 8):
        self.observed_cost = observed_cost
        self.calls = 0

    def execute(self, params):
        self.calls += 1
        return ToolCallResult(
            self.name,
            True,
            "ok",
            data={"x": 1},
            cost_vector=list(self.observed_cost),
        )


def test_custom_tool_requires_explicit_v5_metadata_but_remains_legacy_usable():
    registry = ToolRegistry()
    tool = CustomTool()
    registry.register(tool)
    assert registry.execute_tool(tool.name, {}).success
    with pytest.raises(ValidationError, match="no trusted v5"):
        registry.get_metadata(tool.name)


def custom_proposal(plan, *, capability="custom.test", effect=SideEffectClass.OBSERVATIONAL,
                    cost=(0.1,) * 8, handshake=False):
    return ActionProposal(
        "cost-intent", plan.revision, "step-1", "CustomTool", {},
        (ExpectedPostcondition("fields_equal", {"x": 1}),), cost, effect,
        frozenset({capability}), "adversarial-test", handshake,
    )


def custom_intent():
    return IntentContract(
        "cost-intent", "custom", (ExpectedPostcondition("fields_equal", {"x": 1}),),
        frozenset({"do-not-expand-authority"}), frozenset({"custom.test", "forged"}),
        ResourceBudget((2.0,) * 8),
        frozenset({SideEffectClass.OBSERVATIONAL, SideEffectClass.IRREVERSIBLE}),
    )


def test_unconfigured_custom_tool_proposal_cannot_execute_in_v5():
    registry = ToolRegistry()
    tool = CustomTool()
    registry.register(tool)
    engine = AgenticEngine(tool_registry=registry)
    plan = Plan("custom", [PlanStep(1, "custom", tool.name, {})])
    with pytest.raises(ValidationError, match="no trusted v5"):
        engine.run_governed_cycle(custom_intent(), plan=plan, proposal=custom_proposal(plan))
    assert tool.calls == 0


@pytest.mark.parametrize(
    "forgery",
    [
        {"capability": "forged"},
        {"effect": SideEffectClass.IRREVERSIBLE},
        {"cost": (0.0,) * 8},
        {"handshake": True},
    ],
    ids=["capability", "side-effect", "predicted-cost", "handshake"],
)
def test_forged_tool_governance_metadata_cannot_execute(forgery):
    registry = ToolRegistry()
    tool = CustomTool()
    registry.register(tool, metadata=ToolCapability(
        "custom.test", SideEffectClass.OBSERVATIONAL, (0.1,) * 8
    ))
    engine = AgenticEngine(tool_registry=registry)
    plan = Plan("custom", [PlanStep(1, "custom", tool.name, {})])
    with pytest.raises(ValidationError, match="trusted tool capability"):
        engine.run_governed_cycle(
            custom_intent(), plan=plan, proposal=custom_proposal(plan, **forgery)
        )
    assert tool.calls == 0


def run_custom_cost_cycle(observed_cost, budget):
    registry = ToolRegistry()
    tool = CustomTool(observed_cost)
    metadata = ToolCapability(
        "custom.test", SideEffectClass.OBSERVATIONAL, (0.1,) * 8
    )
    registry.register(tool, metadata=metadata)
    engine = AgenticEngine(tool_registry=registry)
    intent = IntentContract(
        "cost-intent", "custom", (ExpectedPostcondition("fields_equal", {"x": 1}),),
        frozenset({"do-not-expand-authority"}), frozenset({"custom.test"}),
        ResourceBudget(budget), frozenset({SideEffectClass.OBSERVATIONAL}),
    )
    plan = Plan("custom", [PlanStep(1, "custom", tool.name, {})])
    return engine.run_governed_cycle(intent, plan=plan), tool


def test_estimate_error_within_budget_is_evidence_not_failure():
    receipt, _ = run_custom_cost_cycle((0.2,) * 8, (1.0,) * 8)
    assert any(receipt.residual.resource)
    assert receipt.reconciliation.hard_budget_breach is False
    assert receipt.closure.decision is ClosureDecision.HALT_SUCCESS


def test_hard_budget_breach_cannot_be_bypassed_by_verified_intent():
    receipt, _ = run_custom_cost_cycle((1.2,) * 8, (1.0,) * 8)
    assert receipt.intent_satisfaction.status.value == "VERIFIED"
    assert receipt.reconciliation.hard_budget_breach is True
    assert receipt.closure.decision is ClosureDecision.HALT_FAILURE


def test_all_non_admit_decisions_are_rejected_before_tool_invocation():
    engine = AgenticEngine()
    intent = make_intent()
    plan = engine.planner.create_plan(intent.goal)
    proposal = engine.planner.propose_step(plan, intent)
    admitted = ProspectiveAgenticAdmission().evaluate(
        intent, proposal, engine.state_fingerprint(), GovernedBudget(intent.initial_budget)
    )
    calls = 0
    original = engine.tool_registry.execute_tool

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    engine.tool_registry.execute_tool = counted
    for decision in (
        AdmissionDecision.CONSTRAIN,
        AdmissionDecision.REPROJECT,
        AdmissionDecision.BLOCK,
        AdmissionDecision.ESCALATE,
    ):
        with pytest.raises(ValidationError):
            engine.tool_registry.execute_admitted(
                proposal, replace(admitted, decision=decision), intent, engine.state_fingerprint()
            )
    assert calls == 0


def test_reflection_publication_failure_rolls_back_every_activation():
    engine = AgenticEngine()
    candidate = engine.reflection_engine.build_candidate(
        "proposal", [0.0] * 12, {0: 0.2, 1: 0.3}, "controlled failure"
    )

    class String:
        def __init__(self, value, fail=False):
            self._activation = value
            self.fail = fail

        @property
        def activation(self):
            return self._activation

        @activation.setter
        def activation(self, value):
            if self.fail:
                self.fail = False
                raise ValueError("injected publication failure")
            self._activation = value

    class Membrane:
        strings = [String(0.0), String(0.0, fail=True)] + [String(0.0) for _ in range(10)]

    membrane = Membrane()
    before = tuple(item.activation for item in membrane.strings)
    result = engine.reflection_engine.commit_candidate(membrane, candidate)
    assert result.committed is False
    assert tuple(item.activation for item in membrane.strings) == before
