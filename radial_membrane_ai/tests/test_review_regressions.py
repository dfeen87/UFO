"""Regression coverage for governed recurrence and swarm review findings."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    ExpectedPostcondition,
    GovernedSwarmResult,
    IntentContract,
    Plan,
    PlanStep,
    ResourceBudget,
    SideEffectClass,
)
from radial_membrane_ai.exceptions import ValidationError


def _intent() -> IntentContract:
    return IntentContract(
        "review-intent",
        "retrieve memory",
        (ExpectedPostcondition("fields_present", {"fields": ("type",)}),),
        frozenset(),
        frozenset({"memory.read"}),
        ResourceBudget((2.0,) * 8),
        frozenset({SideEffectClass.OBSERVATIONAL}),
    )


def test_adapted_parameters_recompute_explicit_postconditions() -> None:
    planner = AgenticEngine().planner
    plan = Plan("retrieve memory", [PlanStep(
        1,
        "retrieve",
        "MemoryRetrievalTool",
        {"type": "original", "tag": "old"},
        expected_postconditions=(ExpectedPostcondition(
            "fields_equal", {"type": "original", "tag": "old"}
        ),),
    )])

    proposal = planner.propose_step(
        plan,
        _intent(),
        require_action_local=True,
        adapted_parameters={"type": "adapted", "tag": "new"},
    )

    assert proposal.parameters == {"type": "adapted", "tag": "new"}
    assert proposal.expected_postconditions == (ExpectedPostcondition(
        "fields_equal", {"type": "adapted", "tag": "new"}
    ),)


def test_swarm_result_binds_winner_and_initial_state() -> None:
    run = AgenticEngine().run_governed_intent(_intent(), max_cycles=1)
    fingerprint = run.receipts[0].initial_state_fingerprint

    result = GovernedSwarmResult("winner", fingerprint, "winner", run)
    assert result.governed_run is run

    with pytest.raises(ValidationError, match="selected agent"):
        GovernedSwarmResult("winner", fingerprint, "other", run)
    with pytest.raises(ValidationError, match="initial state"):
        GovernedSwarmResult("winner", "forged", "winner", run)
    with pytest.raises(ValidationError, match="initial receipt"):
        GovernedSwarmResult(
            "winner",
            fingerprint,
            "winner",
            replace(
                run,
                receipts=(),
                feedback_contexts=(),
                transition_signatures=(),
                cycles_executed=0,
                final_closure=None,
                success=False,
            ),
        )
