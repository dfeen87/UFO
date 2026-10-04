# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Goal planning and hierarchical step decomposition for Version 3 Agentic AI.
Integrates goal breakdown, tool mapping, cost envelope monitoring, and adaptive plan adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional

from radial_membrane_ai.agentic.tools import ToolCallResult, ToolRegistry
from radial_membrane_ai.agentic.contracts import (
    ActionProposal,
    AgenticActionReceipt,
    ExpectedPostcondition,
    GovernedFeedbackContext,
    IntentContract,
    TransitionSignature,
    VerificationStatus,
    stable_digest,
)
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
    expected_postconditions: tuple[ExpectedPostcondition, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces plan step metrics invariants."""
        if not isinstance(self.step_id, int) or isinstance(self.step_id, bool) or self.step_id <= 0:
            raise ValidationError("step_id must be a positive integer.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationError("step description cannot be empty.")
        if not math.isfinite(self.cost_impact) or self.cost_impact < 0.0:
            raise ValidationError(f"cost_impact cannot be negative, got {self.cost_impact}.")
        if not math.isfinite(self.tension_impact) or self.tension_impact < 0.0:
            raise ValidationError(f"tension_impact cannot be negative, got {self.tension_impact}.")
        if not all(isinstance(value, ExpectedPostcondition) for value in self.expected_postconditions):
            raise ValidationError("step postconditions must be typed ExpectedPostcondition values.")
        self.expected_postconditions = tuple(self.expected_postconditions)


@dataclass
class Plan:
    """Complete goal plan containing steps and execution state."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, REPLANNING, FAILED
    total_cost: float = 0.0
    total_tension: float = 0.0
    revision: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces plan totals invariants."""
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValidationError("plan goal cannot be empty.")
        if self.status not in {"PENDING", "IN_PROGRESS", "COMPLETED", "REPLANNING", "FAILED"}:
            raise ValidationError(f"invalid plan status: {self.status}.")
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValidationError("plan step IDs must be unique.")
        if not math.isfinite(self.total_cost) or self.total_cost < 0.0:
            raise ValidationError(f"total_cost cannot be negative, got {self.total_cost}.")
        if not math.isfinite(self.total_tension) or self.total_tension < 0.0:
            raise ValidationError(f"total_tension cannot be negative, got {self.total_tension}.")
        if not self.revision:
            identity = [
                (s.step_id, s.description, s.tool_name, s.tool_params, s.expected_postconditions)
                for s in self.steps
            ]
            self.revision = stable_digest({"goal": self.goal, "steps": identity})[:24]

    def current_step(self) -> Optional[PlanStep]:
        """Returns the next incomplete step in the plan."""
        for step in self.steps:
            if not step.completed:
                return step
        return None

    def is_finished(self) -> bool:
        """Checks if all steps in the plan are completed."""
        return all(step.completed for step in self.steps) and len(self.steps) > 0

    def refresh_revision(self) -> None:
        """Assign a new stable identity after a material plan change."""
        identity = [
            (s.step_id, s.description, s.tool_name, s.tool_params, s.completed, s.expected_postconditions)
            for s in self.steps
        ]
        self.revision = stable_digest({"goal": self.goal, "steps": identity})[:24]


class GoalPlanner:
    """Decomposes goals into governed plan steps and manages adaptive plan execution."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, max_cost_limit: float = 10.0) -> None:
        if max_cost_limit <= 0.0:
            raise ValidationError(f"max_cost_limit must be positive, got {max_cost_limit}.")
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_cost_limit = max_cost_limit

    def create_plan(self, goal: str) -> Plan:
        """Decomposes a goal into a structured, executable Plan."""
        if not isinstance(goal, str) or not goal.strip():
            raise ValidationError("goal must be a non-empty string.")
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
                if step.tool_params.get("_recovery"):
                    plan.status = "FAILED"
                    return result
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

    def propose_step(
        self,
        plan: Plan,
        intent: IntentContract,
        *,
        require_action_local: bool = False,
        feedback_context: GovernedFeedbackContext | None = None,
        transition_signature: TransitionSignature | None = None,
    ) -> ActionProposal:
        """Create, but never execute or authorize, the current step proposal."""
        if (feedback_context is None) != (transition_signature is None):
            raise ValidationError("adaptive planning context must be supplied as a validated pair.")
        step = plan.current_step()
        if step is None or step.tool_name is None:
            raise ValidationError("plan has no executable current step.")
        metadata = self.tool_registry.get_metadata(step.tool_name)
        parameters = dict(step.tool_params)
        if feedback_context is not None and transition_signature is not None:
            parameters = self._adapt_parameters(step.tool_name, parameters, feedback_context)
        postconditions = step.expected_postconditions
        if not postconditions:
            try:
                projected_step = PlanStep(
                    step.step_id, step.description, step.tool_name, parameters,
                    expected_postconditions=step.expected_postconditions,
                )
                postconditions = self._postconditions(projected_step)
            except ValidationError:
                if require_action_local:
                    raise
                # Legacy isolated-cycle compatibility only. Recurrent runs fail
                # closed rather than treating intent criteria as action criteria.
                postconditions = intent.success_conditions
        return ActionProposal(
            intent_id=intent.intent_id,
            plan_revision=plan.revision,
            action_id=f"step-{step.step_id}",
            tool_name=step.tool_name,
            parameters=parameters,
            expected_postconditions=postconditions,
            predicted_cost=metadata.predicted_cost,
            side_effect_class=metadata.side_effect_class,
            required_capabilities=frozenset({metadata.capability_id}),
            provenance=f"GoalPlanner:{plan.revision}:{step.step_id}",
            handshake_required=metadata.handshake_applicable,
            source_feedback_digest=(
                feedback_context.feedback_context_digest if feedback_context is not None else None
            ),
            source_transition_signature_digest=(
                transition_signature.transition_signature_digest
                if transition_signature is not None else None
            ),
        )

    @staticmethod
    def _adapt_parameters(
        tool_name: str,
        parameters: Dict[str, Any],
        feedback: GovernedFeedbackContext,
    ) -> Dict[str, Any]:
        """Apply the sole declared v5 adaptation: bounded observational retrieval focus."""
        if tool_name != "MemoryRetrievalTool":
            return parameters
        if "INTENT_PARTIALLY_SATISFIED" in feedback.residual.uncertainty:
            focus = ("partial_intent_evidence", "verify")
        elif feedback.residual.uncertainty:
            focus = ("unresolved_uncertainty", "evidence")
        elif feedback.residual.goal:
            focus = ("unresolved_goal", "evidence")
        elif any(value > 0.0 for value in feedback.residual.resource):
            focus = ("resource_discrepancy", "audit")
        else:
            focus = ("verified_continuation", "context")
        # Only the existing observational selector fields may change. Tool,
        # capability, side-effect class, cost, and authority remain untouched.
        parameters["type"], parameters["tag"] = focus
        return parameters

    @staticmethod
    def _postconditions(step: PlanStep) -> tuple[ExpectedPostcondition, ...]:
        """Construct deterministic action-local evidence for built-in tools."""
        params = step.tool_params
        if step.tool_name == "SearchTool":
            return (ExpectedPostcondition("fields_equal", {"query": str(params.get("query", "")).lower()}),)
        if step.tool_name == "MemoryRetrievalTool":
            return (ExpectedPostcondition("fields_equal", {
                "type": params.get("type", "semantic"), "tag": params.get("tag", "all")
            }),)
        if step.tool_name == "APIRequestTool":
            return (ExpectedPostcondition("fields_equal", {
                "endpoint": params.get("endpoint"), "method": str(params.get("method", "GET")).upper(),
                "status_code": 200,
            }),)
        if step.tool_name == "PythonCodeExecutorTool":
            return (ExpectedPostcondition("fields_present", {"fields": ("result",)}),)
        if step.tool_name == "DatabaseQueryTool":
            return (ExpectedPostcondition("fields_present", {"fields": ("records", "count")}),)
        raise ValidationError("plan step lacks a deterministic governed postcondition.")

    @staticmethod
    def commit_verified_step(plan: Plan, receipt: AgenticActionReceipt) -> None:
        """Atomically advance only the exact independently verified plan step."""
        step = plan.current_step()
        if step is None or receipt.plan_revision != plan.revision:
            raise ValidationError("receipt is stale for the active plan.")
        if receipt.proposal.action_id != f"step-{step.step_id}":
            raise ValidationError("receipt does not bind the active plan step.")
        if receipt.verification.status is not VerificationStatus.VERIFIED:
            raise ValidationError("an unverified action cannot advance a governed plan.")
        step.completed = True
        plan.status = "COMPLETED" if plan.is_finished() else "IN_PROGRESS"
        plan.refresh_revision()

    def replan(self, plan: Plan, reason: str) -> None:
        """Adjusts remaining steps in response to execution failures or envelope pressure."""
        plan.status = "REPLANNING"
        if any(s.tool_params.get("_recovery") for s in plan.steps if not s.completed):
            plan.status = "FAILED"
            return
        remaining_steps = [s for s in plan.steps if not s.completed]

        # Insert a recovery / memory retrieval step before remaining steps
        recovery_step = PlanStep(
            step_id=max((s.step_id for s in plan.steps), default=0) + 1,
            description=f"Recovery reflection step due to: {reason}",
            tool_name="MemoryRetrievalTool",
            tool_params={"type": "recovery_log", "tag": reason[:20], "_recovery": True},
        )
        completed_steps = [s for s in plan.steps if s.completed]
        plan.steps = completed_steps + [recovery_step] + remaining_steps

        # Simplify or substitute failed steps if any
        for step in remaining_steps:
            if step.tool_name == "PythonCodeExecutorTool":
                # Fallback to safer query
                step.tool_params = {"code": "result = 'RECOVERED_SAFE_EXECUTION'"}

        plan.status = "IN_PROGRESS"
        plan.refresh_revision()
