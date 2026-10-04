"""Adversarial coverage for governed single-agent durable memory."""

from copy import deepcopy
from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    BaseTool,
    ClosureDecision,
    ExpectedPostcondition,
    GovernedMemoryConfig,
    MemoryCommitStatus,
    MemoryQualificationStatus,
    Plan,
    PlanStep,
    ResourceBudget,
    SideEffectClass,
    ToolOutcomeUnknown,
    IntentContract,
    VerificationStatus,
)
from radial_membrane_ai.agentic.tools import ToolCapability
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.semantic_memory import AdmissibilityContext, MemoryPolicy


def configured_engine(*, coherence=0.8, failure=None, qualified=True):
    config = GovernedMemoryConfig(
        MemoryPolicy(), AdmissibilityContext(coherence, False, 1.0, 0.0)
    )
    engine = AgenticEngine(memory_config=config, memory_failure_injector=failure)
    activation = 0.8 if qualified else 0.1
    for string in engine.engine.membrane.strings:
        string.activation = activation
    return engine


def governed_intent():
    return IntentContract(
        "memory-intent", "search memory evidence",
        (ExpectedPostcondition("fields_equal", {"query": "memory evidence"}),),
        frozenset(), frozenset({"search"}), ResourceBudget((3.0,) * 8),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def plan(expected="memory evidence"):
    return Plan("search memory evidence", [PlanStep(
        1, "observe", "SearchTool", {"query": "memory evidence"},
        expected_postconditions=(ExpectedPostcondition("fields_equal", {"query": expected}),),
    )])


def snapshot(engine):
    memory = engine.engine.semantic_memory
    return (deepcopy(memory.local_store), deepcopy(memory.history), deepcopy(memory.residuals),
            deepcopy(memory.curvature_state), deepcopy(engine.engine.sao_promotor.promotion_ledger))


def test_candidate_is_not_memory_and_temporal_provisional_has_no_side_effects():
    engine = configured_engine(qualified=False)
    before = snapshot(engine)
    receipt = engine.run_governed_cycle(governed_intent(), plan=plan())
    assert receipt.memory.candidate is not None
    assert receipt.memory.qualification.status is MemoryQualificationStatus.PROVISIONAL
    assert snapshot(engine) == before


def test_failed_verification_never_persists():
    engine = configured_engine()
    receipt = engine.run_governed_cycle(governed_intent(), plan=plan("contradiction"))
    assert receipt.verification.status is VerificationStatus.FAILED
    assert receipt.memory.candidate is not None
    assert receipt.memory.qualification.status is MemoryQualificationStatus.REJECTED
    assert not engine.engine.semantic_memory.local_store


@pytest.mark.parametrize("status", [VerificationStatus.UNKNOWN, VerificationStatus.PARTIAL])
def test_unknown_and_partial_verification_never_persist(monkeypatch, status):
    engine = configured_engine()
    original = engine.verifiers.verify

    def altered(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, status=status, reason_codes=(f"FORCED_{status.value}",))

    monkeypatch.setattr(engine.verifiers, "verify", altered)
    receipt = engine.run_governed_cycle(governed_intent(), plan=plan())
    assert receipt.memory.qualification.status is MemoryQualificationStatus.REJECTED
    assert not engine.engine.semantic_memory.local_store


def test_policy_rejection_occurs_before_any_write():
    engine = configured_engine(coherence=0.1)
    before = snapshot(engine)
    receipt = engine.run_governed_cycle(governed_intent(), plan=plan())
    assert receipt.memory.final_verdict == "POLICY_REJECTED"
    assert receipt.memory.commit.status is MemoryCommitStatus.NOT_ATTEMPTED
    assert snapshot(engine) == before


def test_blocked_action_has_no_candidate_or_memory():
    engine = configured_engine()
    blocked = replace(governed_intent(), initial_budget=ResourceBudget((0.0,) * 8))
    receipt = engine.run_governed_cycle(blocked, plan=plan())
    assert receipt.execution.state.value == "NOT_EXECUTED"
    assert receipt.memory.candidate is None
    assert not engine.engine.semantic_memory.local_store


class UnknownMemoryTool(BaseTool):
    name = "UnknownMemoryTool"
    description = "irreversible test boundary"

    def execute(self, params):
        raise ToolOutcomeUnknown("acknowledgement unavailable")


def test_outcome_unknown_escalates_without_memory():
    engine = configured_engine()
    engine.tool_registry.register(UnknownMemoryTool(), metadata=ToolCapability(
        "unknown.write", SideEffectClass.IRREVERSIBLE, (0.1,) * 8, False, False,
    ))
    intent = replace(
        governed_intent(), delegated_authority=frozenset({"unknown.write"}),
        permitted_side_effects=frozenset({SideEffectClass.IRREVERSIBLE}),
    )
    active_plan = Plan("unknown", [PlanStep(
        1, "unknown", "UnknownMemoryTool", {},
        expected_postconditions=(ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
    )])
    receipt = engine.run_governed_cycle(intent, plan=active_plan)
    assert receipt.execution.state.value == "OUTCOME_UNKNOWN"
    assert receipt.memory.candidate is None
    assert receipt.closure.decision is ClosureDecision.ESCALATE
    assert not engine.engine.semantic_memory.local_store


def test_success_replay_and_memory_state_binding():
    engine = configured_engine()
    active_plan = plan()
    proposal = engine.planner.propose_step(active_plan, governed_intent())
    stale_state = engine.state_fingerprint()
    first = engine.run_governed_cycle(governed_intent(), plan=active_plan, proposal=proposal)
    assert first.memory.commit.status is MemoryCommitStatus.COMMITTED
    assert len(engine.engine.semantic_memory.local_store) == 1
    assert engine.state_fingerprint() != stale_state
    replay = engine.memory_pipeline.govern(
        governed_intent(), proposal, first.execution.execution_id, first.observation,
        first.verification, engine.engine.membrane, engine.engine.boundary,
    )
    assert replay.commit.status is MemoryCommitStatus.ALREADY_COMMITTED
    assert len(engine.engine.semantic_memory.local_store) == 1
    assert len(engine.engine.sao_promotor.promotion_ledger) == 1
    with pytest.raises(ValidationError):
        engine.tool_registry.execute_admitted(proposal, first.admission, governed_intent(), engine.state_fingerprint())


def test_atomic_commit_failure_restores_every_memory_surface():
    def fail():
        raise RuntimeError("injected publication failure")

    engine = configured_engine(failure=fail)
    before = snapshot(engine)
    receipt = engine.run_governed_cycle(governed_intent(), plan=plan())
    assert receipt.memory.commit.status is MemoryCommitStatus.FAILED
    assert receipt.closure.decision is ClosureDecision.REFLECT
    assert snapshot(engine) == before


def test_forged_memory_provenance_and_plan_trajectory_are_rejected():
    engine = configured_engine(qualified=False)
    result = engine.run_governed_intent(governed_intent(), max_cycles=1, plan=plan())
    receipt = result.receipts[0]
    candidate = replace(receipt.memory.candidate, observation_id="other-observation")
    forged_memory = replace(receipt.memory, candidate=candidate, qualification=None)
    with pytest.raises(ValidationError):
        replace(receipt, memory=forged_memory)
    forged_transition = replace(receipt.plan_transition, resulting_plan_revision="forged-revision")
    with pytest.raises(ValidationError):
        replace(result, receipts=(replace(receipt, plan_transition=forged_transition),))


def test_canonical_memory_scenario_excludes_diagnostic_timestamps():
    first = configured_engine().run_governed_cycle(governed_intent(), plan=plan())
    second = configured_engine().run_governed_cycle(governed_intent(), plan=plan())
    assert first.canonical() == second.canonical()
    assert first.memory.commit.memory_state_digest == second.memory.commit.memory_state_digest
