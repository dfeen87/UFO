#!/usr/bin/env python3
# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Version 3 Agentic AI Simulation Demonstration Script.
Demonstrates autonomous goal planning, tool execution, dual-trigger reflection, and swarm contract bidding.
"""

from __future__ import annotations

from radial_membrane_ai.agentic.engine import AgenticEngine, AgenticSwarmEngine
from radial_membrane_ai.utils import set_deterministic_env


def main() -> None:
    set_deterministic_env(seed=0)

    print("=" * 80)
    print("🛸 UFO Version 3.0.0 — Governed Agentic AI Simulation Demonstration 🛸".center(80))
    print("=" * 80)
    print()

    # Part 1: Governed Single-Agent Goal Execution & Reflection
    print("--- Part 1: Single-Agent Governed Goal Planning & Tool Execution ---")
    agentic_engine = AgenticEngine()
    goal = "Execute python code calculations and search knowledge base for UFO architecture"

    print(f"Submitting Goal: '{goal}'")
    run_res = agentic_engine.run_goal(goal, max_steps=5)

    print(f"\nGoal Status: {run_res.plan.status}")
    print(f"Total Plan Cost: {run_res.plan.total_cost:.4f}")
    print(f"Total Plan Tension: {run_res.plan.total_tension:.4f}")
    print(f"Final Membrane Tension: {run_res.final_tension:.4f}")
    print(f"Reflections Logged: {len(run_res.reflections)}")

    for idx, step in enumerate(run_res.plan.steps, 1):
        print(f"  Step [{idx}]: {step.description}")
        print(f"    - Tool: {step.tool_name}")
        print(f"    - Completed: {step.completed}")
        if step.result:
            print(f"    - Output: {step.result.output[:100]}...")
            print(f"    - Cost Impact: {step.cost_impact:.4f} | Tension Delta: {step.tension_impact:.4f}")

    if run_res.reflections:
        print("\nLogged Reflection Events:")
        for refl in run_res.reflections:
            print(f"  • Trigger: {refl.trigger_type} | Step: '{refl.step_description}'")
            print(f"    Action Taken: {refl.action_taken}")

    print("\n" + "=" * 80)

    # Part 2: Multi-Agent Swarm Contract Bidding & Auction
    print("--- Part 2: Multi-Agent Swarm Contract Bidding & Role Auctioning ---")
    swarm_engine = AgenticSwarmEngine()

    goals = [
        {
            "goal": "Execute complex factorial calculation with code tool",
            "required_capabilities": ["code"],
            "max_budget": 10.0,
        },
        {
            "goal": "Search UFO architecture documentation corpus",
            "required_capabilities": ["search"],
            "max_budget": 5.0,
        },
        {
            "goal": "Query agent status database records",
            "required_capabilities": ["query"],
            "max_budget": 6.0,
        },
    ]

    swarm_res = swarm_engine.run_swarm_goals(goals, ticks_per_contract=3)

    print(f"Total Swarm Contracts Processed: {len(swarm_res.contracts)}")
    for contract in swarm_res.contracts:
        print(f"\nContract ID: {contract.contract_id}")
        print(f"  Goal: {contract.goal}")
        print(f"  Required Capabilities: {contract.required_capabilities}")
        print(f"  Assigned Agent: {contract.assigned_agent_id}")
        if contract.winning_bid:
            print(f"  Winning Bid Score: {contract.winning_bid.bid_score:.4f}")
            print(f"  Winning Estimated Cost: {contract.winning_bid.estimated_cost:.4f}")

    print("\n" + "=" * 80)
    print("🛸 Version 3.0.0 Agentic AI Simulation Complete Successfully! 🛸")
    print("=" * 80)


if __name__ == "__main__":
    main()
