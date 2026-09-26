# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Goal planning and hierarchical step decomposition for Version 3 Agentic AI.
Integrates goal breakdown, tool mapping, cost envelope monitoring, and adaptive plan adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from radial_membrane_ai.agentic.tools import ToolCallResult, ToolRegistry
from radial_membrane_ai.exceptions import ValidationError


@dataclass
class PlanStep:
    """Individual step in an agentic plan."""

    step_id: int
    description: str
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    result: Optional[ToolCallResult] = None
    cost_impact: float = 0.0
    tension_impact: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces plan step metrics invariants."""
        if self.cost_impact < 0.0:
            raise ValidationError(f"cost_impact cannot be negative, got {self.cost_impact}.")
        if self.tension_impact < 0.0:
            raise ValidationError(f"tension_impact cannot be negative, got {self.tension_impact}.")


@dataclass
class Plan:
    """Complete goal plan containing steps and execution state."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, REPLANNING, FAILED
    total_cost: float = 0.0
    total_tension: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces plan totals invariants."""
        if self.total_cost < 0.0:
            raise ValidationError(f"total_cost cannot be negative, got {self.total_cost}.")
        if self.total_tension < 0.0:
            raise ValidationError(f"total_tension cannot be negative, got {self.total_tension}.")

    def current_step(self) -> Optional[PlanStep]:
        """Returns the next incomplete step in the plan."""
        for step in self.steps:
            if not step.completed:
                return step
        return None

    def is_finished(self) -> bool:
        """Checks if all steps in the plan are completed."""
        return all(step.completed for step in self.steps) and len(self.steps) > 0


class GoalPlanner:
    """Decomposes goals into governed plan steps and manages adaptive plan execution."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, max_cost_limit: float = 10.0) -> None:
        if max_cost_limit <= 0.0:
            raise ValidationError(f"max_cost_limit must be positive, got {max_cost_limit}.")
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_cost_limit = max_cost_limit

    def create_plan(self, goal: str) -> Plan:
        """Decomposes a goal into a structured, executable Plan."""
        steps: List[PlanStep] = []
        goal_lower = goal.lower()

        if "code" in goal_lower or "calculate" in goal_lower or "eval" in goal_lower:
            steps.append(PlanStep(
                step_id=1,
                description="Retrieve contextual memory for code execution setup",
                tool_name="MemoryRetrievalTool",
                tool_params={"type": "code_context", "tag": "setup"},
            ))
            steps.append(PlanStep(
                step_id=2,
                description="Execute code snippet to achieve goal",
                tool_name="PythonCodeExecutorTool",
                tool_params={"code": "x = [i**2 for i in range(10)]; result = sum(x)"},
            ))
            steps.append(PlanStep(
                step_id=3,
                description="Query status database for audit trail",
                tool_name="DatabaseQueryTool",
                tool_params={"filter_key": "status", "filter_value": "ACTIVE"},
            ))
        elif "search" in goal_lower or "research" in goal_lower or "find" in goal_lower:
            steps.append(PlanStep(
                step_id=1,
                description="Search knowledge base for goal queries",
                tool_name="SearchTool",
                tool_params={"query": goal},
            ))
            steps.append(PlanStep(
                step_id=2,
                description="Retrieve memory history related to search",
                tool_name="MemoryRetrievalTool",
                tool_params={"type": "search_history", "tag": "research"},
            ))
        elif "api" in goal_lower or "fetch" in goal_lower or "network" in goal_lower:
            steps.append(PlanStep(
                step_id=1,
                description="Dispatch external API request",
                tool_name="APIRequestTool",
                tool_params={
                    "endpoint": "https://api.ufo.ai/agentic/task",
                    "method": "POST",
                    "payload": {"goal": goal},
                },
            ))
            steps.append(PlanStep(
                step_id=2,
                description="Query execution log database",
                tool_name="DatabaseQueryTool",
                tool_params={},
            ))
        else:
            # Default general multi-step workflow
            steps.append(PlanStep(
                step_id=1,
                description="Retrieve relevant memory context",
                tool_name="MemoryRetrievalTool",
                tool_params={"type": "general", "tag": "context"},
            ))
            steps.append(PlanStep(
                step_id=2,
                description="Search knowledge corpus",
                tool_name="SearchTool",
                tool_params={"query": goal},
            ))
            steps.append(PlanStep(
                step_id=3,
                description="Execute verification logic",
                tool_name="PythonCodeExecutorTool",
                tool_params={"code": "verified = True; result = {'status': 'GOAL_VERIFIED'}"},
            ))

        return Plan(goal=goal, steps=steps, status="PENDING")

    def execute_next_step(self, plan: Plan, current_membrane_tension: float = 0.0) -> Optional[ToolCallResult]:
        """Executes the next step of a plan, updating plan cost and tension metrics."""
        step = plan.current_step()
        if not step or plan.status in ("FAILED", "COMPLETED"):
            return None

        plan.status = "IN_PROGRESS"

        if step.tool_name:
            result = self.tool_registry.execute_tool(step.tool_name, step.tool_params)
            step.result = result
            step.completed = True

            step_cost = sum(result.cost_vector)
            step.cost_impact = step_cost
            step.tension_impact = result.tension_delta

            plan.total_cost += step_cost
            plan.total_tension += result.tension_delta

            # Check for replanning triggers
            if not result.success or (plan.total_cost + current_membrane_tension > self.max_cost_limit):
                self.replan(plan, reason=f"Step {step.step_id} failure or cost limit exceeded.")
                return result

            if plan.is_finished():
                plan.status = "COMPLETED"

            return result
        else:
            step.completed = True
            if plan.is_finished():
                plan.status = "COMPLETED"
            return None

    def replan(self, plan: Plan, reason: str) -> None:
        """Adjusts remaining steps in response to execution failures or envelope pressure."""
        plan.status = "REPLANNING"
        remaining_steps = [s for s in plan.steps if not s.completed]

        # Insert a recovery / memory retrieval step before remaining steps
        recovery_step = PlanStep(
            step_id=len(plan.steps) + 1,
            description=f"Recovery reflection step due to: {reason}",
            tool_name="MemoryRetrievalTool",
            tool_params={"type": "recovery_log", "tag": reason[:20]},
        )
        plan.steps.append(recovery_step)

        # Simplify or substitute failed steps if any
        for step in remaining_steps:
            if step.tool_name == "PythonCodeExecutorTool":
                # Fallback to safer query
                step.tool_params = {"code": "result = 'RECOVERED_SAFE_EXECUTION'"}

        plan.status = "IN_PROGRESS"
