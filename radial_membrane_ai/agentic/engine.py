# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Unified Agentic Engines for Version 3 U.F.O. architecture.
Integrates goal planning, tool execution, dual-trigger reflection, and swarm bidding into runtime engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np

from radial_membrane_ai.agentic.planner import GoalPlanner, Plan
from radial_membrane_ai.agentic.reflection import ReflectionEngine, ReflectionRecord
from radial_membrane_ai.agentic.swarm import AgentBid, SwarmAuctioneer, SwarmContract
from radial_membrane_ai.agentic.tools import ToolRegistry
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
    ) -> None:
        self.engine = engine or SingleAgentEngine()
        self.tool_registry = tool_registry or ToolRegistry()
        self.planner = planner or GoalPlanner(tool_registry=self.tool_registry)
        self.reflection_engine = reflection_engine or ReflectionEngine(
            residual_ledger=getattr(self.engine.governor, "residual_ledger", None) or self.engine.ledger
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
