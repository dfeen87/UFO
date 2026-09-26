"""Adversarial regression tests for deterministic Version 3 operation."""

import math

import pytest

from radial_membrane_ai.agentic.planner import GoalPlanner, Plan, PlanStep
from radial_membrane_ai.agentic.reflection import ReflectionEngine
from radial_membrane_ai.agentic.swarm import AgentBid, SwarmAuctioneer
from radial_membrane_ai.agentic.tools import BaseTool, PythonCodeExecutorTool, ToolCallResult, ToolRegistry
from radial_membrane_ai.cost import RuntimeCostVector
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.temporal import TemporalMembraneState


class ExplodingTool(BaseTool):
    name = "ExplodingTool"
    description = "Raises to exercise registry instrumentation."

    def execute(self, params):
        raise RuntimeError("boom")


def test_plan_graph_recovery_is_ordered_and_bounded() -> None:
    registry = ToolRegistry()
    registry.register(ExplodingTool())
    planner = GoalPlanner(registry)
    plan = Plan(
        "recover",
        [PlanStep(1, "failure", "ExplodingTool"), PlanStep(2, "next", "SearchTool", {"query": "ufo"})],
    )
    result = planner.execute_next_step(plan)
    assert result is not None and not result.success and result.data["instrumented"]
    assert plan.steps[1].tool_params["_recovery"] is True
    assert len({step.step_id for step in plan.steps}) == len(plan.steps)


def test_executor_blocks_escape_and_results_reject_nan() -> None:
    executor = PythonCodeExecutorTool()
    assert not executor.execute({"code": "__import__('os').getcwd()"}).success
    assert not executor.execute({"code": "(1).__class__"}).success
    with pytest.raises(ValidationError):
        ToolCallResult("bad", True, "bad", tension_delta=math.nan)


def test_reflections_are_repeatable_at_threshold() -> None:
    engine = ReflectionEngine(tension_threshold=1.0, curvature_threshold=2.0)
    first = engine.evaluate_and_reflect("x", None, 1.0, 0.0)
    second = engine.evaluate_and_reflect("x", None, 1.0, 0.0)
    assert first is not None and first.timestamp == 1.0
    assert second is not None and second.timestamp == 2.0


def test_auction_rejects_cross_contract_and_partial_capability_bids() -> None:
    auction = SwarmAuctioneer()
    auction.create_contract("c", "goal", ["code", "verify"])
    wrong = AgentBid("a", "other", 10.0, 1.0, 1.0, ["code", "verify"])
    partial = AgentBid("b", "c", 10.0, 1.0, 1.0, ["code"])
    assert auction.run_auction("c", [wrong, partial]).status == "FAILED"


def test_cost_and_tension_envelopes_reject_non_finite_loads() -> None:
    with pytest.raises(ValidationError):
        RuntimeCostVector(math.inf, 0, 0, 0, 0, 0, 0, 0)
    state = TemporalMembraneState()
    with pytest.raises(ValidationError):
        state.update_tension_accumulation(math.inf, 0, 0, 0)
    with pytest.raises(ValidationError):
        state.update_tick_history(0, -1, 0)
