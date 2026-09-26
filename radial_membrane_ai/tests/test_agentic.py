# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Unit and integration tests for Version 3 Agentic AI subsystem.
Ensures 100% line coverage for tools, planner, reflection, swarm, engines, workloads, and CLI.
"""

from __future__ import annotations

import argparse
import pytest
from radial_membrane_ai.agentic.tools import (
    BaseTool,
    ToolCallResult,
    ToolRegistry,
    SearchTool,
    PythonCodeExecutorTool,
    APIRequestTool,
    DatabaseQueryTool,
    MemoryRetrievalTool,
)
from radial_membrane_ai.agentic.planner import GoalPlanner, Plan, PlanStep
from radial_membrane_ai.agentic.reflection import ReflectionEngine
from radial_membrane_ai.agentic.swarm import SwarmAuctioneer
from radial_membrane_ai.agentic.engine import AgenticEngine, AgenticSwarmEngine
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.workloads.agentic_workload import create_agentic_workload
from ufo_cli import run_agentic_cli, run_agentic_swarm_cli


class DummyTool(BaseTool):
    name = "DummyTool"
    description = "Dummy tool for abstract class testing."

    def execute(self, params: dict) -> ToolCallResult:
        return super().execute(params)  # Calls ABC method pass


def test_tools_coverage() -> None:
    dummy = DummyTool()
    res_dummy = dummy.execute({})
    assert res_dummy is None

    search = SearchTool()
    res1 = search.execute({"query": "ufo architecture"})
    assert res1.success is True
    assert "The UFO is a governed" in res1.output

    res_empty = search.execute({"query": ""})
    assert res_empty.success is False
    assert search.execute({}).success is False

    res_nomatch = search.execute({"query": "nonexistent_xyz_term"})
    assert res_nomatch.success is True

    py_tool = PythonCodeExecutorTool()
    assert py_tool.execute({"code": "   "}).success is False
    assert py_tool.execute({"code": "def bad("}).success is False
    assert py_tool.execute({"code": "result = 100"}).success is True
    assert py_tool.execute({"code": "x = 5\nx * 2"}).success is True
    assert py_tool.execute({"code": "1 / 0"}).success is False

    api_tool = APIRequestTool()
    assert api_tool.execute({}).success is False
    assert api_tool.execute({"endpoint": "https://api.ufo.ai"}).success is True

    db_tool = DatabaseQueryTool()
    assert db_tool.execute({}).success is True
    assert db_tool.execute({"filter_key": "region", "filter_value": "Analytical"}).success is True

    mem_tool = MemoryRetrievalTool()
    assert mem_tool.execute({"type": "semantic"}).success is True

    reg = ToolRegistry()
    assert len(reg.list_tools()) >= 5
    assert reg.execute_tool("SearchTool", {"query": "agentic"}).success is True
    with pytest.raises(KeyError):
        reg.get_tool("UnregisteredTool")


def test_planner_coverage() -> None:
    planner = GoalPlanner(max_cost_limit=5.0)

    # Goal branch 1: search
    plan_search = planner.create_plan("Search knowledge base for ufo docs")
    assert plan_search.steps[0].tool_name == "SearchTool"

    # Goal branch 2: api / fetch
    plan_api = planner.create_plan("Fetch data from network API")
    assert plan_api.steps[0].tool_name == "APIRequestTool"

    # Goal branch 3: general
    plan_gen = planner.create_plan("General task without explicit keywords")
    assert plan_gen.steps[0].tool_name == "MemoryRetrievalTool"

    # Step without tool name
    empty_step_plan = Plan(
        goal="Empty tool step", steps=[PlanStep(step_id=1, description="Manual step", completed=False)]
    )
    assert planner.execute_next_step(empty_step_plan) is None
    assert empty_step_plan.status == "COMPLETED"

    # Finished plan check
    finished_plan = Plan(
        goal="Finished", steps=[PlanStep(step_id=1, description="Done", completed=True)], status="COMPLETED"
    )
    assert finished_plan.current_step() is None
    assert finished_plan.is_finished() is True
    assert planner.execute_next_step(finished_plan) is None

    # Replanning with PythonCodeExecutorTool in remaining steps
    code_plan = planner.create_plan("Execute code for calculation")
    # Force step 1 failure
    code_plan.steps[0].tool_params = {"type": None, "tag": None}
    code_plan.total_cost = 100.0  # trigger cost replan
    planner.execute_next_step(code_plan)
    assert code_plan.status == "IN_PROGRESS"


def test_reflection_engine_coverage() -> None:
    refl_eng = ReflectionEngine(tension_threshold=1.0, curvature_threshold=2.0)

    # No trigger
    assert refl_eng.evaluate_and_reflect("Normal", ToolCallResult("SearchTool", True, "ok"), 0.5, 0.5) is None

    # Tool failure trigger
    rec_fail = refl_eng.evaluate_and_reflect("Failed", ToolCallResult("SearchTool", False, "err"), 0.5, 0.5)
    assert rec_fail is not None and rec_fail.trigger_type == "TOOL_FAILURE"

    # Suggested deltas for tool failure
    deltas_fail = refl_eng.get_suggested_activation_deltas(rec_fail)
    assert deltas_fail[1] == 0.2

    # Instability trigger
    rec_instab = refl_eng.evaluate_and_reflect("Instable", ToolCallResult("SearchTool", True, "ok"), 3.0, 5.0)
    assert rec_instab is not None and rec_instab.trigger_type == "MEMBRANE_INSTABILITY"

    # Dual trigger
    rec_dual = refl_eng.evaluate_and_reflect("Dual", ToolCallResult("SearchTool", False, "err"), 3.0, 5.0)
    assert rec_dual is not None and rec_dual.trigger_type == "DUAL_TRIGGER"


def test_swarm_auctioneer_coverage() -> None:
    auctioneer = SwarmAuctioneer()
    contract = auctioneer.create_contract("C1", "Task", required_capabilities=["code"], max_budget=10.0)

    # Ineligible bid
    assert auctioneer.calculate_agent_bid("A1", contract, ["search"], 0.5, 1.0) is None

    # Overbudget or unstable bid
    assert auctioneer.calculate_agent_bid("A1", contract, ["code"], 0.5, -0.1) is None

    # Valid bid
    bid = auctioneer.calculate_agent_bid("A1", contract, ["code"], 0.5, 1.0)
    assert bid is not None

    # Auction run
    assert auctioneer.run_auction("C1", []).status == "FAILED"
    awarded = auctioneer.run_auction("C1", [bid])
    assert awarded is not None and awarded.status == "AWARDED"

    with pytest.raises(KeyError):
        auctioneer.run_auction("NONEXISTENT", [])


def test_agentic_engine_coverage() -> None:
    # High tension threshold to force reflection in single-agent run
    eng = AgenticEngine()
    eng.reflection_engine.tension_threshold = 0.01  # force tension reflection trigger

    res = eng.run_goal("Execute code for computation", max_steps=5)
    assert res.success is True
    assert len(res.reflections) > 0

    # Finished plan run_goal return
    res_finished = eng.run_goal("Execute code for computation", max_steps=0)
    assert res_finished is not None

    # Run goal with empty plan steps
    eng_empty = AgenticEngine()
    eng_empty.planner.create_plan = lambda goal: Plan(goal=goal, steps=[])
    res_empty = eng_empty.run_goal("No steps", max_steps=5)
    assert res_empty is not None


def test_agentic_swarm_engine_coverage() -> None:
    custom_agents = [
        UFOAgent(agent_id="Custom_1", capabilities=["code"]),
        UFOAgent(agent_id="Custom_2", capabilities=["search"]),
    ]
    swarm_eng = AgenticSwarmEngine(agents=custom_agents)
    goals = [
        {"goal": "Calculate numbers with code", "required_capabilities": ["code"], "max_budget": 10.0},
    ]
    res = swarm_eng.run_swarm_goals(goals, ticks_per_contract=2)
    assert len(res.contracts) == 1


def test_agentic_workload_and_cli() -> None:
    workload = create_agentic_workload()
    assert workload.name == "Version 3 Agentic AI Stress Workload"

    args_quiet = argparse.Namespace(quiet=True, steps=2, agentic=True, agentic_swarm=False, mode="agentic")
    run_agentic_cli(args_quiet)
    run_agentic_swarm_cli(args_quiet)

    args_verbose = argparse.Namespace(quiet=False, steps=2, agentic=False, agentic_swarm=True, mode="agentic-swarm")
    run_agentic_cli(args_verbose)
    run_agentic_swarm_cli(args_verbose)
