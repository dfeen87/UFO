"""Release-gate tests for adaptive recurrence and governed swarm execution."""

from dataclasses import replace

import pytest

from radial_membrane_ai.agentic import (
    AgenticEngine,
    AgenticSwarmEngine,
    BaseTool,
    ExpectedPostcondition,
    GovernedSwarmTask,
    IntentContract,
    Plan,
    PlanStep,
    ResourceBudget,
    SideEffectClass,
)
from radial_membrane_ai.agentic.contracts import AdmissionDecision, ExecutionState
from radial_membrane_ai.agentic.contracts import stable_digest
from radial_membrane_ai.agentic.memory import semantic_memory_digest
from radial_membrane_ai.agentic.tools import ToolCallResult, ToolCapability, ToolRegistry
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.multi_agent.agent import UFOAgent


class EvidenceTool(BaseTool):
    name = "EvidenceTool"
    description = "Produces controlled observational evidence."

    def execute(self, params):
        data = {"ok": True}
        if params.get("signal"):
            data["signal"] = "yes"
        return ToolCallResult(self.name, True, "evidence", data=data, cost_vector=(0.1,) * 8)


def adaptive_intent():
    return IntentContract(
        "adaptive-intent",
        "adaptive evidence",
        (
            ExpectedPostcondition("fields_equal", {"signal": "yes"}),
            ExpectedPostcondition("fields_equal", {"future": "complete"}),
        ),
        frozenset({"do-not-expand-authority"}),
        frozenset({"evidence.read", "memory.read"}),
        ResourceBudget((3.0,) * 8),
    )


def adaptive_run(signal):
    registry = ToolRegistry()
    registry.register(EvidenceTool(), metadata=ToolCapability(
        "evidence.read", SideEffectClass.OBSERVATIONAL, (0.1,) * 8, False, False
    ))
    plan = Plan("adaptive evidence", [
        PlanStep(
            1, "collect", "EvidenceTool", {"signal": signal},
            expected_postconditions=(ExpectedPostcondition("fields_present", {"fields": ("ok",)}),),
        ),
        PlanStep(2, "retrieve", "MemoryRetrievalTool", {"type": "general", "tag": "context"}),
    ])
    return AgenticEngine(tool_registry=registry).run_governed_intent(
        adaptive_intent(), max_cycles=2, plan=plan
    )


def test_valid_feedback_changes_only_declared_bounded_retrieval_semantics():
    partial = adaptive_run(True)
    unknown = adaptive_run(False)
    partial_proposal = partial.receipts[1].proposal
    unknown_proposal = unknown.receipts[1].proposal
    assert partial.receipts[0].closure.decision.value == "CONTINUE"
    assert unknown.receipts[0].closure.decision.value == "CONTINUE"
    assert partial_proposal.parameters == {"type": "partial_intent_evidence", "tag": "verify"}
    assert unknown_proposal.parameters == {"type": "unresolved_uncertainty", "tag": "evidence"}
    assert partial_proposal.required_capabilities == unknown_proposal.required_capabilities == {"memory.read"}
    assert partial_proposal.side_effect_class is unknown_proposal.side_effect_class
    assert partial_proposal.source_feedback_digest == partial.feedback_contexts[0].feedback_context_digest
    assert (partial_proposal.source_transition_signature_digest
            == partial.transition_signatures[0].transition_signature_digest)
    assert partial_proposal.proposal_digest != unknown_proposal.proposal_digest
    assert partial.receipts[1].admission.proposal_digest == partial_proposal.proposal_digest


def test_adaptive_trajectory_replays_and_rejects_forged_proposal_binding():
    first = adaptive_run(True)
    replay = adaptive_run(True)
    assert first.canonical() == replay.canonical()
    forged = replace(first.receipts[1].proposal, source_feedback_digest="0" * 64)
    with pytest.raises(ValidationError, match="missing, stale, or forged"):
        AgenticEngine().run_governed_cycle(adaptive_intent(), proposal=forged)


def swarm_intent(authority=frozenset({"memory.read"})):
    return IntentContract(
        "swarm-intent", "general swarm task",
        (ExpectedPostcondition("fields_equal", {"never": "automatic"}),),
        frozenset({"do-not-expand-authority"}), authority, ResourceBudget((1.0,) * 8),
    )


def test_governed_swarm_selection_is_not_authorization_and_preserves_winner_state(monkeypatch):
    winner = UFOAgent("a-winner", capabilities=["general"])
    loser = UFOAgent("z-loser", capabilities=["general"])
    winner.membrane.strings[0].activation = 0.37
    swarm = AgenticSwarmEngine(agents=[loser, winner])
    expected_initial_state = stable_digest({
        "activations": tuple(float(s.activation) for s in winner.membrane.strings),
        "tension": float(winner.membrane.temporal_state.accumulated_tension),
        "boundary": tuple(float(winner.boundary.get_radius(s.theta)) for s in winner.membrane.strings),
        "semantic_memory": semantic_memory_digest(winner.semantic_memory),
    })
    monkeypatch.setattr(
        AgenticEngine,
        "run_goal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy execution used")),
    )
    task = GovernedSwarmTask("governed-1", "general swarm task", ("general",), 10.0, swarm_intent())
    result = swarm.run_governed_swarm_tasks((task,), max_cycles=1)[0]
    assert result.selected_agent_id == "a-winner"
    assert result.governed_run.receipts[0].admission.decision is AdmissionDecision.ADMIT
    assert result.governed_run.receipts[0].execution.state is ExecutionState.EXECUTED
    assert result.governed_run.receipts[0].initial_state_fingerprint == expected_initial_state
    assert winner.membrane.strings[0].activation == 0.37
    assert loser.membrane.strings[0].activation != 0.37
    assert result.governed_run.final_remaining_budget != task.intent.initial_budget


def test_swarm_capability_and_scalar_budget_cannot_supply_execution_authority():
    swarm = AgenticSwarmEngine(agents=[UFOAgent("winner", capabilities=["general", "retrieval"])])
    task = GovernedSwarmTask(
        "governed-2", "general swarm task", ("retrieval",), 1000.0,
        swarm_intent(authority=frozenset({"search"})),
    )
    first = swarm.run_governed_swarm_tasks((task,), max_cycles=2)[0]
    receipt = first.governed_run.receipts[0]
    assert receipt.admission.decision is AdmissionDecision.BLOCK
    assert receipt.execution.state is ExecutionState.NOT_EXECUTED
    assert first.governed_run.final_remaining_budget == task.intent.initial_budget
    assert "AUTHORITY_CAPABILITY_MISSING" in receipt.admission.reason_codes


def test_governed_swarm_selection_and_trajectory_are_deterministic_and_bound():
    def execute():
        agents = [UFOAgent("b", capabilities=["general"]), UFOAgent("a", capabilities=["general"])]
        task = GovernedSwarmTask("deterministic", "general swarm task", ("general",), 10.0, swarm_intent())
        return AgenticSwarmEngine(agents=agents).run_governed_swarm_tasks((task,), max_cycles=1)[0]

    first, replay = execute(), execute()
    assert first.selected_agent_id == replay.selected_agent_id == "a"
    assert first.selection_digest == replay.selection_digest
    assert first.governed_run.canonical() == replay.governed_run.canonical()
    with pytest.raises(ValidationError, match="selection evidence"):
        replace(first, selection_digest="0" * 64)
