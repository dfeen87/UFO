from dataclasses import FrozenInstanceError

import pytest

from radial_membrane_ai.agentic.admission import GovernedBudget, ProspectiveAgenticAdmission
from radial_membrane_ai.agentic.contracts import (
    AdmissionDecision, ClosureDecision, ExecutionState, ExpectedPostcondition, IntentContract,
    ResourceBudget, SideEffectClass, VerificationStatus,
)
from radial_membrane_ai.agentic.engine import AgenticEngine


def intent(*, authority=frozenset({"search"}), budget=(2.0,) * 8):
    return IntentContract(
        "intent-1", "search ufo", (ExpectedPostcondition("fields_equal", {"query": "search ufo"}),),
        frozenset({"do-not-expand-authority"}), authority, ResourceBudget(budget),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def test_complete_cycle_is_independently_verified_and_deterministic():
    left = AgenticEngine().run_governed_cycle(intent())
    right = AgenticEngine().run_governed_cycle(intent())
    assert left.execution.state is ExecutionState.EXECUTED
    assert left.verification.status is VerificationStatus.VERIFIED
    assert left.closure.decision is ClosureDecision.HALT_SUCCESS
    assert left.durable_memory_attempted is False
    assert left.canonical() == right.canonical()


def test_blocked_action_has_receipt_and_never_invokes_executor(monkeypatch):
    engine = AgenticEngine()
    invoked = 0

    def forbidden(*args, **kwargs):
        nonlocal invoked
        invoked += 1
        raise AssertionError("executor crossed a blocked boundary")

    monkeypatch.setattr(engine.tool_registry, "execute_tool", forbidden)
    receipt = engine.run_governed_cycle(intent(authority=frozenset()))
    assert invoked == 0
    assert receipt.admission.decision is AdmissionDecision.BLOCK
    assert receipt.execution.state is ExecutionState.NOT_EXECUTED
    assert receipt.observation is None
    assert receipt.closure.decision is ClosureDecision.HALT_FAILURE


def test_budget_blocks_prospectively_and_authority_is_immutable():
    contract = intent(budget=(0.0,) * 8)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        contract.goal = "changed"
    receipt = AgenticEngine().run_governed_cycle(contract)
    assert "BUDGET_INSUFFICIENT" in receipt.admission.reason_codes
    assert receipt.reconciliation.remaining_budget == contract.initial_budget


def test_state_and_parameter_binding_reject_stale_admission():
    engine = AgenticEngine()
    contract = intent()
    plan = engine.planner.create_plan(contract.goal)
    proposal = engine.planner.propose_step(plan, contract)
    fingerprint = engine.state_fingerprint()
    verdict = ProspectiveAgenticAdmission().evaluate(
        contract,
        proposal,
        fingerprint,
        GovernedBudget(contract.initial_budget),
    )
    with pytest.raises(Exception, match="stale, or mismatched"):
        ProspectiveAgenticAdmission.validate_execution_binding(contract, proposal, verdict, "different-state")


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), True, "1", 10 ** 10000],
    ids=["nan", "inf", "bool", "string", "huge-int"],
)
def test_budget_numeric_boundary_fails_closed(bad):
    with pytest.raises(Exception):
        ResourceBudget((bad,) + (1.0,) * 7)
