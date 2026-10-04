# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Unified Agentic Engines for Version 3 U.F.O. architecture.
Integrates goal planning, tool execution, dual-trigger reflection, and swarm bidding into runtime engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np

from radial_membrane_ai.agentic.planner import GoalPlanner, Plan
from radial_membrane_ai.agentic.reflection import ReflectionEngine, ReflectionRecord
from radial_membrane_ai.agentic.swarm import AgentBid, SwarmAuctioneer, SwarmContract
from radial_membrane_ai.agentic.tools import ToolRegistry
from radial_membrane_ai.agentic.admission import GovernedBudget, ProspectiveAgenticAdmission
from radial_membrane_ai.agentic.closure import AgenticClosure
from radial_membrane_ai.agentic.contracts import (
    ActionProposal,
    AdmissionDecision,
    AgenticActionReceipt,
    AgenticResidual,
    ClosureDecision,
    ExecutionRecord,
    ExecutionState,
    GovernedFeedbackContext,
    IntentContract,
    GovernedRunResult,
    Observation,
    MemoryCommitStatus,
    PlanTransitionEvidence,
    PlanTransitionType,
    ReflectionResult,
    ResourceReconciliation,
    VerificationStatus,
    admission_binding_digest,
    stable_digest,
)
from radial_membrane_ai.agentic.verification import VerificationRegistry
from radial_membrane_ai.agentic.memory import (
    GovernedMemoryConfig,
    GovernedMemoryPipeline,
    semantic_memory_digest,
)
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine, MultiAgentRunResult
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine, SingleAgentRunResult


@dataclass
class AgenticRunResult:
    """Diagnostic result from an AgenticEngine execution run."""

    plan: Plan
    reflections: List[ReflectionRecord] = field(default_factory=list)
    single_agent_result: Optional[SingleAgentRunResult] = None
    final_tension: float = 0.0
    final_curvature: float = 0.0
    success: bool = True


class AgenticEngine:
    """Governed single-agent engine orchestrating plans, tools, membrane feedback, and reflections."""

    def __init__(
        self,
        engine: Optional[SingleAgentEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[GoalPlanner] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        memory_config: GovernedMemoryConfig | None = None,
        memory_failure_injector: Any = None,
    ) -> None:
        self.engine = engine or SingleAgentEngine()
        self.tool_registry = tool_registry or ToolRegistry()
        self.planner = planner or GoalPlanner(tool_registry=self.tool_registry)
        self.reflection_engine = reflection_engine or ReflectionEngine(
            residual_ledger=getattr(self.engine.governor, "residual_ledger", None) or self.engine.ledger
        )
        self.admission = ProspectiveAgenticAdmission()
        self.verifiers = VerificationRegistry()
        self.closure = AgenticClosure()
        self.memory_pipeline = GovernedMemoryPipeline(
            self.engine.semantic_memory, self.engine.sao_promotor, memory_config,
            failure_injector=memory_failure_injector,
        )

    def state_fingerprint(self) -> str:
        """Fingerprint only the authoritative state relevant to a v5 action."""
        state = {
            "activations": tuple(float(s.activation) for s in self.engine.membrane.strings),
            "tension": float(self.engine.membrane.temporal_state.accumulated_tension),
            "boundary": tuple(float(self.engine.boundary.get_radius(s.theta)) for s in self.engine.membrane.strings),
            "semantic_memory": semantic_memory_digest(self.engine.semantic_memory),
        }
        return stable_digest(state)

    def run_governed_cycle(
        self,
        intent: IntentContract,
        *,
        plan: Plan | None = None,
        proposal: ActionProposal | None = None,
        stability_allowed: bool = True,
        policy_allowed: bool = True,
        handshake: Any = None,
    ) -> AgenticActionReceipt:
        """Execute exactly one v5 governed transaction; never recur."""
        if not isinstance(intent, IntentContract):
            raise ValueError("run_governed_cycle requires an IntentContract.")
        active_plan = plan or self.planner.create_plan(intent.goal)
        return self._run_governed_transaction(
            intent, active_plan, GovernedBudget(intent.initial_budget), proposal=proposal,
            stability_allowed=stability_allowed, policy_allowed=policy_allowed, handshake=handshake,
        )

    def _run_governed_transaction(
        self,
        intent: IntentContract,
        active_plan: Plan,
        budget: GovernedBudget,
        *,
        proposal: ActionProposal | None = None,
        stability_allowed: bool = True,
        policy_allowed: bool = True,
        handshake: Any = None,
        require_action_local: bool = False,
        feedback_context: GovernedFeedbackContext | None = None,
        source_receipt: AgenticActionReceipt | None = None,
        cycle_index: int = 0,
    ) -> AgenticActionReceipt:
        """Shared single-action primitive; the supplied budget remains runtime-owned."""
        if feedback_context is not None:
            self._validate_feedback_context(
                intent, active_plan, budget, feedback_context, source_receipt, cycle_index
            )
        elif source_receipt is not None or cycle_index != 0:
            raise ValidationError("a recurrent cycle requires its immediately preceding feedback context.")
        initial_fingerprint = self.state_fingerprint()
        exact_proposal = proposal or self.planner.propose_step(
            active_plan, intent, require_action_local=require_action_local
        )
        if exact_proposal.plan_revision != active_plan.revision:
            raise ValueError("proposal is not bound to the supplied plan revision.")

        # Caller-created proposals are untrusted.  Validate their prospective
        # governance claims before admission can reserve budget or grant authority.
        self.tool_registry.validate_governed_proposal(exact_proposal)

        budget_before = budget.remaining
        admission = self.admission.evaluate(
            intent, exact_proposal, initial_fingerprint, budget,
            stability_allowed=stability_allowed, policy_allowed=policy_allowed, handshake=handshake,
        )
        admission_digest = admission_binding_digest(admission)
        observation = None
        hard_budget_breach = False
        if admission.decision is AdmissionDecision.ADMIT:
            execution, tool_result = self.tool_registry.execute_admitted(
                exact_proposal, admission, intent, self.state_fingerprint()
            )
            assert admission.reservation is not None
            if execution.state is ExecutionState.OUTCOME_UNKNOWN:
                # The reservation stays consumed. Unknown cost is never invented.
                observation = None
            else:
                assert tool_result is not None
                observation = Observation(
                    observation_id=f"observation-{exact_proposal.proposal_digest[:16]}",
                    execution_id=execution.execution_id,
                    evidence={"output": tool_result.output, "data": tool_result.data},
                    complete=True,
                )
                assert execution.observed_cost is not None
                hard_budget_breach = budget.reconcile(admission.reservation, execution.observed_cost)
        else:
            execution = ExecutionRecord(
                execution_id=f"execution-{exact_proposal.proposal_digest[:16]}",
                proposal_digest=exact_proposal.proposal_digest,
                admission_digest=admission_digest,
                state=ExecutionState.NOT_EXECUTED,
                tool_name=exact_proposal.tool_name,
                success_reported=None,
            )

        verification = self.verifiers.verify(exact_proposal, observation)
        intent_satisfaction = self.verifiers.verify_intent(intent, observation)
        observed = execution.observed_cost
        resource_residual = tuple(
            max(0.0, observed_value - predicted)
            for observed_value, predicted in zip(observed or (0.0,) * 8, exact_proposal.predicted_cost)
        )
        policy_residual = tuple(
            code for code in admission.reason_codes
            if "AUTHORITY" in code or "POLICY" in code or "SIDE_EFFECT" in code
        )
        residual = AgenticResidual(
            proposal_digest=exact_proposal.proposal_digest,
            goal=tuple(dict.fromkeys(
                (
                    verification.reason_codes
                    if verification.status is not VerificationStatus.VERIFIED
                    else ()
                )
                + (
                    intent_satisfaction.reason_codes
                    if intent_satisfaction.status is not VerificationStatus.VERIFIED
                    else ()
                )
            )),
            resource=resource_residual,
            policy_authority=policy_residual,
            stability=tuple(code for code in admission.reason_codes if "STABILITY" in code),
            uncertainty=(
                tuple(intent_satisfaction.reason_codes)
                if intent_satisfaction.status in {
                    VerificationStatus.PARTIAL,
                    VerificationStatus.UNKNOWN,
                }
                else ()
            ),
        )

        reflection = ReflectionResult(False, False, None, ("NO_REFLECTION_REQUIRED",))
        should_reflect = execution.state is ExecutionState.FAILED or bool(residual.stability)
        if should_reflect:
            before = [float(s.activation) for s in self.engine.membrane.strings]
            deltas = {1: 0.2, 9: 0.15} if execution.state is ExecutionState.FAILED else {4: -0.2, 5: -0.2}
            try:
                candidate = self.reflection_engine.build_candidate(
                    exact_proposal.proposal_digest, before, deltas, "governed-cycle residual"
                )
                reflection = self.reflection_engine.commit_candidate(self.engine.membrane, candidate)
            except ValidationError:
                reflection = ReflectionResult(True, False, None, ("CANDIDATE_VALIDATION_REJECTED",))

        memory = self.memory_pipeline.govern(
            intent, exact_proposal, execution.execution_id, observation, verification,
            self.engine.membrane, self.engine.boundary,
        )
        resulting_fingerprint = self.state_fingerprint()
        closure = self.closure.evaluate(
            admission=admission, execution_state=execution.state, verification=verification,
            intent_satisfaction=intent_satisfaction,
            residual=residual, remaining_budget=budget.remaining,
            authority_valid=not policy_residual, reflection=reflection,
            hard_budget_breach=hard_budget_breach,
            memory_commit_failed=memory.commit.status is MemoryCommitStatus.FAILED,
        )
        reconciliation = ResourceReconciliation(
            admission.reservation,
            exact_proposal.predicted_cost,
            observed,
            budget.remaining,
            hard_budget_breach,
            budget_before,
        )
        return AgenticActionReceipt(
            intent.intent_id, active_plan.revision, exact_proposal, initial_fingerprint, admission,
            reconciliation, execution, observation, verification, intent_satisfaction, residual, reflection,
            resulting_fingerprint, closure,
            durable_memory_attempted=memory.commit.status is not MemoryCommitStatus.NOT_ATTEMPTED,
            memory=memory,
            plan_transition=PlanTransitionEvidence(
                active_plan.revision, exact_proposal.proposal_digest, exact_proposal.action_id,
                PlanTransitionType.NONE, active_plan.revision, ("PLAN_UNCHANGED",),
            ),
            incoming_feedback_digest=(
                feedback_context.feedback_context_digest if feedback_context is not None else None
            ),
        )

    def _build_feedback_context(
        self,
        intent: IntentContract,
        receipt: AgenticActionReceipt,
        cycle_index: int,
    ) -> GovernedFeedbackContext:
        """Bind reconciled state only after the receipt's plan transition is final."""
        if receipt.memory is None or receipt.plan_transition is None:
            raise ValidationError("finalized receipt evidence is required for feedback.")
        return GovernedFeedbackContext(
            intent.intent_id,
            intent.intent_digest,
            cycle_index,
            stable_digest(receipt),
            receipt.resulting_state_fingerprint,
            receipt.residual,
            receipt.memory.commit.memory_state_digest,
            receipt.reconciliation.remaining_budget,
            receipt.plan_transition,
            receipt.plan_transition.resulting_plan_revision,
            receipt.closure,
        )

    def _validate_feedback_context(
        self,
        intent: IntentContract,
        active_plan: Plan,
        budget: GovernedBudget,
        feedback: GovernedFeedbackContext,
        source_receipt: AgenticActionReceipt | None,
        cycle_index: int,
    ) -> None:
        """Fail closed before proposal generation, admission, or reservation."""
        if not isinstance(feedback, GovernedFeedbackContext) or source_receipt is None:
            raise ValidationError("recurrent feedback requires typed source evidence.")
        expected = self._build_feedback_context(intent, source_receipt, cycle_index - 1)
        if feedback != expected:
            raise ValidationError("incoming feedback is not bound to the immediately preceding receipt.")
        if feedback.prior_closure.decision not in {ClosureDecision.CONTINUE, ClosureDecision.REPLAN}:
            raise ValidationError("terminal closure feedback cannot initiate an autonomous cycle.")
        if (
            feedback.resulting_state_fingerprint != self.state_fingerprint()
            or feedback.memory_state_digest != semantic_memory_digest(self.engine.semantic_memory)
            or feedback.remaining_budget != budget.remaining
            or feedback.resulting_plan_revision != active_plan.revision
        ):
            raise ValidationError("incoming feedback is stale relative to authoritative runtime state.")

    def run_governed_intent(
        self,
        intent: IntentContract,
        *,
        max_cycles: int,
        plan: Plan | None = None,
        stability_allowed: bool = True,
        policy_allowed: bool = True,
        handshake: Any = None,
    ) -> GovernedRunResult:
        """Run a finite trajectory whose authority is re-earned on every cycle."""
        if not isinstance(intent, IntentContract):
            raise ValueError("run_governed_intent requires an IntentContract.")
        if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles <= 0:
            raise ValueError("max_cycles must be a positive integer.")
        active_plan = plan or self.planner.create_plan(intent.goal)
        budget = GovernedBudget(intent.initial_budget)
        receipts: list[AgenticActionReceipt] = []
        feedback_contexts: list[GovernedFeedbackContext] = []
        termination = "MAX_CYCLES_REACHED"

        for cycle_index in range(max_cycles):
            receipt = self._run_governed_transaction(
                intent, active_plan, budget, stability_allowed=stability_allowed,
                policy_allowed=policy_allowed, handshake=handshake, require_action_local=True,
                feedback_context=(feedback_contexts[-1] if feedback_contexts else None),
                source_receipt=(receipts[-1] if receipts else None),
                cycle_index=cycle_index,
            )
            decision = receipt.closure.decision
            prior_revision = active_plan.revision
            transition_type = PlanTransitionType.NONE
            transition_reasons: tuple[str, ...] = ("PLAN_UNCHANGED",)

            if receipt.verification.status is VerificationStatus.VERIFIED:
                self.planner.commit_verified_step(active_plan, receipt)
                transition_type = PlanTransitionType.VERIFIED_STEP
                transition_reasons = ("VERIFIED_STEP_COMMITTED",)

            if decision is ClosureDecision.CONTINUE:
                receipt = replace(receipt, plan_transition=PlanTransitionEvidence(
                    prior_revision, receipt.proposal.proposal_digest, receipt.proposal.action_id,
                    transition_type, active_plan.revision, transition_reasons,
                ))
                receipts.append(receipt)
                feedback_contexts.append(self._build_feedback_context(intent, receipt, cycle_index))
                if active_plan.current_step() is None:
                    termination = "PLAN_EXHAUSTED_WITH_INTENT_UNRESOLVED"
                    break
                continue
            if decision is ClosureDecision.REPLAN:
                self.planner.replan(active_plan, receipt.closure.reason_codes[0])
                transition_type = PlanTransitionType.REPLAN
                transition_reasons = tuple(receipt.closure.reason_codes)
                receipt = replace(receipt, plan_transition=PlanTransitionEvidence(
                    prior_revision, receipt.proposal.proposal_digest, receipt.proposal.action_id,
                    transition_type, active_plan.revision, transition_reasons,
                ))
                receipts.append(receipt)
                feedback_contexts.append(self._build_feedback_context(intent, receipt, cycle_index))
                if active_plan.status == "FAILED":
                    termination = "REPLAN_FAILED_CLOSED"
                    break
                continue

            receipt = replace(receipt, plan_transition=PlanTransitionEvidence(
                prior_revision, receipt.proposal.proposal_digest, receipt.proposal.action_id,
                transition_type, active_plan.revision, transition_reasons,
            ))
            receipts.append(receipt)
            feedback_contexts.append(self._build_feedback_context(intent, receipt, cycle_index))
            termination = {
                ClosureDecision.HALT_SUCCESS: "INTENT_SATISFIED",
                ClosureDecision.HALT_FAILURE: "GOVERNED_FAILURE",
                ClosureDecision.ESCALATE: "AUTOMATION_ESCALATED",
                ClosureDecision.CONSTRAIN: "CONSTRAINT_TRANSITION_UNAVAILABLE",
                ClosureDecision.REFLECT: "REFLECTION_UNRESOLVED",
            }[decision]
            break

        final_closure = receipts[-1].closure if receipts else None
        return GovernedRunResult(
            intent.intent_id, intent.intent_digest, tuple(receipts), len(receipts), budget.remaining,
            final_closure, termination,
            bool(final_closure and final_closure.decision is ClosureDecision.HALT_SUCCESS),
            active_plan.revision,
            tuple(feedback_contexts),
        )

    def run_goal(self, goal: str, max_steps: int = 10) -> AgenticRunResult:
        """Executes an agentic goal plan step-by-step with membrane updates and reflection."""
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer.")
        plan = self.planner.create_plan(goal)
        reflections: List[ReflectionRecord] = []
        last_single_result: Optional[SingleAgentRunResult] = None

        step_counter = 0
        while not plan.is_finished() and step_counter < max_steps and plan.status != "FAILED":
            step_counter += 1
            current_step = plan.current_step()
            if not current_step:
                break

            # Tick membrane state
            self.engine.tick(task_value=0.5, excitation=np.ones(12) * 0.2)

            cur_tension = float(self.engine.membrane.temporal_state.accumulated_tension)
            cur_curvature = float(
                sum(abs(self.engine.boundary.curvature(s.theta)) for s in self.engine.membrane.strings)
            )

            # Validate admissibility gate closure invariants prior to executing tool step
            self.engine.admissibility_gate.evaluate_closure(self.engine.membrane, self.engine.boundary)

            # Execute plan step tool call
            tool_res = self.planner.execute_next_step(plan, current_membrane_tension=cur_tension)

            # Evaluate dual-trigger reflection
            reflection = self.reflection_engine.evaluate_and_reflect(
                step_description=current_step.description,
                tool_result=tool_res,
                current_tension=cur_tension,
                current_curvature=cur_curvature,
            )

            if reflection:
                reflections.append(reflection)
                # Apply posture adjustment deltas to membrane
                deltas = self.reflection_engine.get_suggested_activation_deltas(reflection)
                for idx, delta in deltas.items():
                    if 0 <= idx < 12:
                        self.engine.membrane.strings[idx].activation = max(
                            0.0, min(1.0, self.engine.membrane.strings[idx].activation + delta)
                        )

        final_tension_val = float(self.engine.membrane.temporal_state.accumulated_tension)
        final_curvature_val = float(
            sum(abs(self.engine.boundary.curvature(s.theta)) for s in self.engine.membrane.strings)
        )

        return AgenticRunResult(
            plan=plan,
            reflections=reflections,
            single_agent_result=last_single_result,
            final_tension=final_tension_val,
            final_curvature=final_curvature_val,
            success=(plan.status == "COMPLETED"),
        )


@dataclass
class AgenticSwarmRunResult:
    """Diagnostic result from an AgenticSwarmEngine execution run."""

    contracts: List[SwarmContract]
    agent_results: Dict[str, AgenticRunResult] = field(default_factory=dict)
    multi_agent_result: Optional[MultiAgentRunResult] = None
    auction_history: List[Dict[str, Any]] = field(default_factory=list)


class AgenticSwarmEngine:
    """Governed multi-agent swarm engine orchestrating bidding, task delegation, and swarm execution."""

    def __init__(
        self,
        agents: Optional[Sequence[UFOAgent]] = None,
        multi_engine: Optional[MultiAgentEngine] = None,
        auctioneer: Optional[SwarmAuctioneer] = None,
    ) -> None:
        if agents is None:
            a1 = UFOAgent(agent_id="Agent_Analytical", capabilities=["code", "math", "verify"])
            a2 = UFOAgent(agent_id="Agent_Research", capabilities=["search", "retrieval", "context"])
            a3 = UFOAgent(agent_id="Agent_General", capabilities=["general", "api", "query"])
            self.agents = [a1, a2, a3]
        else:
            self.agents = list(agents)

        self.multi_engine = multi_engine or MultiAgentEngine(custom_agents=self.agents)
        self.auctioneer = auctioneer or SwarmAuctioneer()

    def run_swarm_goals(
        self,
        goals: Sequence[Dict[str, Any]],
        ticks_per_contract: int = 5,
    ) -> AgenticSwarmRunResult:
        """Executes a set of contract goals via swarm bidding and governed agent execution."""
        if not isinstance(ticks_per_contract, int) or isinstance(ticks_per_contract, bool) or ticks_per_contract < 0:
            raise ValueError("ticks_per_contract must be a non-negative integer.")
        contracts: List[SwarmContract] = []
        agent_results: Dict[str, AgenticRunResult] = {}

        # 1. Multi-agent tick cycle to establish baseline membrane states
        for _ in range(2):
            self.multi_engine.tick(task_value=0.5, excitation=np.ones(12) * 0.15)

        for g_idx, g_info in enumerate(goals):
            c_id = f"CONTRACT_{g_idx + 1}"
            goal_text = str(g_info.get("goal", "Execute swarm task"))
            caps = g_info.get("required_capabilities", ["general"])
            budget = float(g_info.get("max_budget", 10.0))
            if not math.isfinite(budget):
                raise ValueError("contract budget must be finite.")

            contract = self.auctioneer.create_contract(
                contract_id=c_id,
                goal=goal_text,
                required_capabilities=caps,
                max_budget=budget,
            )

            # Solicit bids from agents
            bids: List[AgentBid] = []
            for agent in self.agents:
                cur_tension = float(agent.membrane.temporal_state.accumulated_tension)
                stability_margin = 2.0 - min(1.9, cur_tension)

                bid = self.auctioneer.calculate_agent_bid(
                    agent_id=agent.agent_id,
                    contract=contract,
                    agent_capabilities=agent.capabilities,
                    current_tension=cur_tension,
                    stability_band_margin=stability_margin,
                )
                if bid:
                    bids.append(bid)

            # Run auction
            awarded_contract = self.auctioneer.run_auction(c_id, bids)
            if awarded_contract is not None:
                contracts.append(awarded_contract)
            if awarded_contract and awarded_contract.assigned_agent_id:
                # Find winning agent and run goal
                winning_agent = next(a for a in self.agents if a.agent_id == awarded_contract.assigned_agent_id)
                single_eng = SingleAgentEngine(membrane=winning_agent.membrane)
                agentic_eng = AgenticEngine(engine=single_eng)
                res = agentic_eng.run_goal(goal_text, max_steps=ticks_per_contract)
                agent_results[f"{c_id}_{winning_agent.agent_id}"] = res

        return AgenticSwarmRunResult(
            contracts=contracts,
            agent_results=agent_results,
            multi_agent_result=None,
            auction_history=self.auctioneer.auction_history,
        )
