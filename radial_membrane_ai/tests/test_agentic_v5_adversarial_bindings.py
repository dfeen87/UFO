"""Regression tests for recurrent verification and swarm execution attribution."""

from dataclasses import replace
import json

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    AgenticSwarmEngine,
    AgentBid,
    ExpectedPostcondition,
    GovernedSwarmTask,
    IntentContract,
    Plan,
    PlanStep,
    ResourceBudget,
    VerificationStatus,
)
from radial_membrane_ai.agentic.contracts import stable_digest
from radial_membrane_ai.agentic.memory import semantic_memory_digest
from radial_membrane_ai.agentic.swarm import SwarmAuctioneer, SwarmContract
from radial_membrane_ai.agentic.tools import MemoryRetrievalTool, ToolRegistry
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.multi_agent.agent import UFOAgent


def recurrent_intent() -> IntentContract:
    return IntentContract(
        "rebind-intent",
        "rebind recurrence",
        (ExpectedPostcondition("fields_equal", {"never": "complete"}),),
        frozenset(),
        frozenset({"memory.read"}),
        ResourceBudget((3.0,) * 8),
    )


def test_production_recurrence_rebinds_selectors_and_preserves_independent_requirements():
    original_conditions = (
        ExpectedPostcondition("fields_equal", {"type": "original", "tag": "old"}),
        ExpectedPostcondition("fields_present", {"fields": ("entries",)}),
    )
    plan = Plan("rebind recurrence", [
        PlanStep(1, "first", "MemoryRetrievalTool", {"type": "original", "tag": "old"}),
        PlanStep(
            2,
            "adapted",
            "MemoryRetrievalTool",
            {"type": "original", "tag": "old"},
            expected_postconditions=original_conditions,
        ),
    ])

    result = AgenticEngine().run_governed_intent(recurrent_intent(), max_cycles=2, plan=plan)
    proposal = result.receipts[1].proposal

    assert proposal.parameters == {"type": "unresolved_uncertainty", "tag": "evidence"}
    assert proposal.expected_postconditions == (
        ExpectedPostcondition(
            "fields_equal", {"type": "unresolved_uncertainty", "tag": "evidence"}
        ),
        original_conditions[1],
    )
    assert result.receipts[1].verification.status is VerificationStatus.VERIFIED
    assert "CONTRADICTORY_EVIDENCE" not in result.receipts[1].verification.reason_codes
    assert proposal.source_feedback_digest == result.feedback_contexts[0].feedback_context_digest
    assert (
        proposal.source_transition_signature_digest
        == result.transition_signatures[0].transition_signature_digest
    )
    assert result.receipts[1].plan_revision == result.feedback_contexts[0].resulting_plan_revision


def test_no_adaptation_leaves_explicit_postconditions_and_digest_stable():
    engine = AgenticEngine()
    conditions = (ExpectedPostcondition("fields_equal", {"type": "original", "tag": "old"}),)
    plan = Plan("rebind recurrence", [PlanStep(
        1, "same", "MemoryRetrievalTool", {"type": "original", "tag": "old"},
        expected_postconditions=conditions,
    )])
    first = engine.planner.propose_step(plan, recurrent_intent())
    replay = engine.planner.propose_step(plan, recurrent_intent())
    assert first.expected_postconditions is conditions
    assert first.expected_postconditions == conditions
    assert first.proposal_digest == replay.proposal_digest


def test_ambiguous_parameter_postcondition_rebinding_fails_closed():
    plan = Plan("rebind recurrence", [
        PlanStep(1, "first", "MemoryRetrievalTool", {"type": "original", "tag": "old"}),
        PlanStep(
            2,
            "ambiguous",
            "MemoryRetrievalTool",
            {"type": "original", "tag": "old"},
            expected_postconditions=(ExpectedPostcondition("selector_equals", {"type": "original"}),),
        ),
    ])
    with pytest.raises(ValidationError, match="cannot be safely rebound"):
        AgenticEngine().run_governed_intent(recurrent_intent(), max_cycles=2, plan=plan)


def test_output_equals_hidden_selector_dependency_fails_before_adapted_execution(monkeypatch):
    original_output = json.dumps({
        "type": "original",
        "tag": "old",
        "entries": [
            "Memory record [original] with tag 'old' retrieved.",
            "Prior governed reflection confirmed stable Lyapunov energy balance.",
        ],
    })
    plan = Plan("rebind recurrence", [
        PlanStep(1, "first", "MemoryRetrievalTool", {"type": "original", "tag": "old"}),
        PlanStep(
            2,
            "opaque",
            "MemoryRetrievalTool",
            {"type": "original", "tag": "old"},
            expected_postconditions=(
                ExpectedPostcondition("output_equals", {"value": original_output}),
            ),
        ),
    ])
    executions = 0
    original_execute = MemoryRetrievalTool.execute

    def counted_execute(self, params):
        nonlocal executions
        executions += 1
        return original_execute(self, params)

    monkeypatch.setattr(MemoryRetrievalTool, "execute", counted_execute)
    with pytest.raises(ValidationError, match="cannot be safely rebound"):
        AgenticEngine().run_governed_intent(recurrent_intent(), max_cycles=2, plan=plan)
    assert executions == 1


def swarm_intent(intent_id: str = "swarm-binding") -> IntentContract:
    return IntentContract(
        intent_id,
        "bound swarm",
        (ExpectedPostcondition("fields_equal", {"never": "complete"}),),
        frozenset(),
        frozenset({"memory.read"}),
        ResourceBudget((2.0,) * 8),
    )


def swarm_result(agents=None):
    agents = agents or [
        UFOAgent("b", capabilities=["general"]),
        UFOAgent("a", capabilities=["general"]),
    ]
    task = GovernedSwarmTask("bound-contract", "bound swarm", ("general",), 10.0, swarm_intent())
    return AgenticSwarmEngine(agents=agents).run_governed_swarm_tasks((task,), max_cycles=1)[0]


def execution_digest(result, run=None, pre_state=None):
    run = run or result.governed_run
    pre_state = pre_state or result.pre_execution_state_fingerprint
    return stable_digest({
        "selection_digest": result.selection_digest,
        "selected_agent_id": result.selected_agent_id,
        "execution_principal_id": run.execution_principal_id,
        "pre_execution_state_fingerprint": pre_state,
        "initial_receipt_digest": stable_digest(run.receipts[0]),
        "intent_digest": result.task.intent.intent_digest,
    })


def test_swarm_rejects_wrong_winner_bid_contract_digest_and_intent():
    result = swarm_result()
    loser_bid = result.admissible_bids[1]
    with pytest.raises(ValidationError, match="auction winner"):
        replace(result, selected_agent_id=loser_bid.agent_id)
    with pytest.raises(ValidationError, match="winning bid"):
        replace(result, winning_bid=loser_bid)
    forged_bid = replace(result.winning_bid, contract_id="other-contract")
    with pytest.raises(ValidationError, match="winning bid|inadmissible"):
        replace(result, winning_bid=forged_bid)
    with pytest.raises(ValidationError, match="selection evidence"):
        replace(result, selection_digest="0" * 64)
    other_task = replace(result.task, intent=swarm_intent("different-intent"))
    with pytest.raises(ValidationError, match="explicit intent|selection evidence"):
        replace(result, task=other_task)


def test_swarm_rejects_prestate_cross_agent_substitution_and_late_mutation():
    selected = swarm_result()
    other_agent = UFOAgent("other", capabilities=["general"])
    other_agent.membrane.strings[0].activation = 0.61
    substitute = swarm_result([other_agent])

    with pytest.raises(ValidationError, match="execution principal"):
        replace(selected, governed_run=substitute.governed_run)
    with pytest.raises(ValidationError, match="pre-execution state"):
        replace(selected, pre_execution_state_fingerprint="f" * 64)

    # Even a recomputed caller digest cannot hide execution initialized from a
    # state different from the one captured at selection.
    forged_digest = execution_digest(selected, run=substitute.governed_run)
    with pytest.raises(ValidationError, match="execution principal"):
        replace(
            selected,
            governed_run=substitute.governed_run,
            execution_binding_digest=forged_digest,
        )


def test_swarm_rejects_identical_state_cross_agent_execution_principal_substitution():
    agent_a = UFOAgent("a", capabilities=["general"])
    agent_b = UFOAgent("b", capabilities=["general"])
    selected = swarm_result([agent_b, agent_a])
    substitute = swarm_result([agent_b])

    assert selected.selected_agent_id == "a"
    assert selected.pre_execution_state_fingerprint == substitute.pre_execution_state_fingerprint
    assert selected.governed_run.execution_principal_id == "a"
    assert substitute.governed_run.execution_principal_id == "b"
    with pytest.raises(ValidationError, match="execution principal"):
        replace(selected, governed_run=substitute.governed_run)

    forged_digest = execution_digest(selected, run=substitute.governed_run)
    with pytest.raises(ValidationError, match="execution principal"):
        replace(
            selected,
            governed_run=substitute.governed_run,
            execution_binding_digest=forged_digest,
        )


class MalformedAuctioneer(SwarmAuctioneer):
    def __init__(self, defect):
        super().__init__()
        self.defect = defect

    def run_auction(self, contract_id, bids):
        awarded = super().run_auction(contract_id, bids)
        if self.defect == "loser":
            awarded.winning_bid = sorted(bids, key=lambda bid: (-bid.bid_score, bid.agent_id))[1]
            awarded.assigned_agent_id = awarded.winning_bid.agent_id
        elif self.defect == "inadmissible":
            awarded.winning_bid = AgentBid(
                "a", contract_id, 100.0, awarded.max_budget + 1.0, 0.0,
                tuple(awarded.required_capabilities), 1.0,
            )
            awarded.assigned_agent_id = "a"
        elif self.defect == "wrong-agent":
            awarded.assigned_agent_id = "b"
        elif self.defect == "wrong-contract":
            return SwarmContract(
                "other-contract", awarded.goal, list(awarded.required_capabilities),
                awarded.max_budget, awarded.assigned_agent_id, "AWARDED", awarded.winning_bid,
            )
        return awarded


@pytest.mark.parametrize("defect", ("loser", "inadmissible", "wrong-agent", "wrong-contract"))
def test_malformed_auction_award_fails_before_execution_or_state_change(monkeypatch, defect):
    agents = [
        UFOAgent("b", capabilities=["general"]),
        UFOAgent("a", capabilities=["general"]),
    ]
    swarm = AgenticSwarmEngine(agents=agents, auctioneer=MalformedAuctioneer(defect))
    before = tuple((
        tuple(string.activation for string in agent.membrane.strings),
        agent.membrane.temporal_state.accumulated_tension,
        tuple(agent.boundary.get_radius(string.theta) for string in agent.membrane.strings),
        semantic_memory_digest(agent.semantic_memory),
    ) for agent in agents)
    governed_runs = 0
    tool_executions = 0

    def forbidden_run(*args, **kwargs):
        nonlocal governed_runs
        governed_runs += 1
        raise AssertionError("governed execution occurred before award validation")

    def forbidden_tool(*args, **kwargs):
        nonlocal tool_executions
        tool_executions += 1
        raise AssertionError("tool execution occurred before award validation")

    monkeypatch.setattr(AgenticEngine, "run_governed_intent", forbidden_run)
    monkeypatch.setattr(ToolRegistry, "execute_tool", forbidden_tool)
    task = GovernedSwarmTask(
        "bound-contract", "bound swarm", ("general",), 10.0, swarm_intent()
    )
    with pytest.raises(ValidationError):
        swarm.run_governed_swarm_tasks((task,), max_cycles=1)

    after = tuple((
        tuple(string.activation for string in agent.membrane.strings),
        agent.membrane.temporal_state.accumulated_tension,
        tuple(agent.boundary.get_radius(string.theta) for string in agent.membrane.strings),
        semantic_memory_digest(agent.semantic_memory),
    ) for agent in agents)
    assert governed_runs == tool_executions == 0
    assert after == before


def test_swarm_selection_and_execution_bindings_replay_deterministically():
    first = swarm_result()
    replay = swarm_result()
    assert first.selected_agent_id == replay.selected_agent_id == "a"
    assert first.winning_bid == replay.winning_bid
    assert first.selection_digest == replay.selection_digest
    assert first.pre_execution_state_fingerprint == replay.pre_execution_state_fingerprint
    assert first.execution_binding_digest == replay.execution_binding_digest
