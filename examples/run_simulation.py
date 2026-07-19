#!/usr/bin/env python3
"""
Demonstration script executing the complete governed simulation engine for the U.F.O. architecture.
Shows Single-Agent and Multi-Agent simulations driving through Green -> Yellow -> Red stability transitions.
"""

from __future__ import annotations
import os
import numpy as np

from radial_membrane_ai.ufo_engine import (
    SingleAgentEngine,
    MultiAgentEngine,
    CostWeights,
    StabilityBandConfig
)
from radial_membrane_ai.ufo_engine.serialization import export_simulation_results_to_json, ledger_to_json


def main() -> None:
    print("=" * 70)
    print("🛸 U.F.O. GOVERNED SIMULATION ENGINE DEMO 🛸")
    print("=" * 70)

    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Single-Agent Simulation Run
    # -------------------------------------------------------------------------
    print("\n--- Running Single-Agent Engine ---")
    # Using strict cost weights to emphasize stability band transitions
    strict_weights = CostWeights.get_preset("strict")
    # Low thresholds to demonstrate Green -> Yellow -> Red transitions beautifully
    band_config = StabilityBandConfig(v_green=0.03, v_red=0.15)

    single_engine = SingleAgentEngine(
        cost_weights=strict_weights,
        band_config=band_config
    )

    # Define task value and excitation sequences
    # We purposefully scale excitation from low to extreme to trigger transitions: Green -> Yellow -> Red
    steps = 9
    task_values = [0.8, 0.9, 0.95, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1]
    excitations = [
        np.ones(12) * 0.1,  # Low excitation (Green)
        np.ones(12) * 0.2,  # Low-mid excitation (Green)
        np.ones(12) * 0.3,  # Mid excitation (Yellow transition)
        np.ones(12) * 1.5,  # High excitation (Yellow/Red transition)
        np.ones(12) * 3.5,  # Extreme excitation (Red)
        np.ones(12) * 5.0,  # Extreme excitation (Red)
        np.ones(12) * 0.1,  # Dissipation / Recovery phase
        np.ones(12) * 0.05,  # Dissipation
        np.ones(12) * 0.01  # Fully dissipated / Recovered
    ]

    single_results = single_engine.run(
        n_steps=steps,
        task_value_sequence=task_values,
        excitation_sequence=excitations
    )

    print(f"Single-Agent Simulation finished {steps} steps.")
    print("Lyapunov Energy history:")
    for step, v_val in enumerate(single_results.v_history):
        band = single_results.band_history[step]
        print(f"  Step {step+1}: Energy V(t) = {v_val:.4f} [{band.upper()}]")

    print("\nInterventions Log:")
    for idx, intervention in enumerate(single_results.interventions):
        print(f"  Tick {idx+1}: {intervention}")

    # Export single-agent results to logs/
    single_out_path = "logs/simulation_run_single_agent.json"
    export_simulation_results_to_json(single_results.__dict__, single_out_path)
    print(f"\nSaved single agent simulation results to {single_out_path}")

    single_ledger_path = "logs/simulation_ledger_single_agent.json"
    ledger_to_json(single_engine.ledger, single_ledger_path)
    print(f"Saved single agent ledger to {single_ledger_path}")

    # -------------------------------------------------------------------------
    # 2. Multi-Agent Simulation Run
    # -------------------------------------------------------------------------
    print("\n--- Running Multi-Agent Engine ---")
    multi_engine = MultiAgentEngine(
        n_agents=3,
        cost_weights=CostWeights.get_preset("balanced"),
        # Adjusting thresholds to demonstrate Green -> Yellow -> Red transitions clearly
        band_config=StabilityBandConfig(c_green=0.6, c_red=0.5)
    )

    # Let's run steps with escalating compliance degradation and cost load
    multi_task_values = [0.9, 0.9, 0.9, 0.8, 0.8, 0.5]
    multi_excitations = [
        np.ones(12) * 0.5,  # Nominal (Green)
        np.ones(12) * 0.5,  # Nominal (Green)
        np.ones(12) * 1.5,  # Mid excitation (Yellow)
        np.ones(12) * 3.0,  # High tension (Red)
        np.ones(12) * 4.5,  # High tension (Red)
        np.ones(12) * 0.1,  # Safe mode recovery
    ]

    # Dynamically inject bad policy/compliance & high latencies on step 3 to trigger red band and quarantine
    # Agent 2 starts experiencing heavy policy compliance drops and latency spikes
    multi_engine.agents[1].shard.policy_compliance = 0.95
    multi_engine.agents[1].shard.latency = 15.0

    # Execute simulation step-by-step to log transitions
    for step_idx in range(6):
        t_val = multi_task_values[step_idx]
        excite = multi_excitations[step_idx]

        # Trigger compliance and latency fault at step 3 to force a hard quarantine
        if step_idx == 3:
            print("\n>>> Fault Injection: Agent 2 policy compliance degraded to 0.1, latency spiked to 2000 ms. <<<")
            multi_engine.agents[1].shard.policy_compliance = 0.1
            multi_engine.agents[1].shard.latency = 2000.0

        multi_engine.tick(t_val, excite)

    multi_results = MultiAgentEngine.run_result = {
        "h_hol_history": multi_engine.h_hol_history,
        "c_mesh_history": multi_engine.c_mesh_history,
        "band_history": multi_engine.band_history,
        "agent_coherences": multi_engine.agent_coherences,
        "interventions": multi_engine.interventions,
        "sao_events": multi_engine.sao_events,
        "residual_history": multi_engine.residual_history
    }

    print("\nMulti-Agent Coherence history:")
    for step, c_val in enumerate(multi_results["c_mesh_history"]):
        band = multi_results["band_history"][step]
        h_hol = multi_results["h_hol_history"][step]
        print(f"  Step {step+1}: C_mesh(t) = {c_val:.4f} | H_hol(t) = {h_hol:.4f} [{band.upper()}]")

    print("\nInterventions Log:")
    for idx, intervention in enumerate(multi_results["interventions"]):
        print(f"  Tick {idx+1}: {intervention}")

    # Export multi-agent results
    multi_out_path = "logs/simulation_run_multi_agent.json"
    export_simulation_results_to_json(multi_results, multi_out_path)
    print(f"\nSaved multi agent simulation results to {multi_out_path}")

    multi_ledger_path = "logs/simulation_ledger_multi_agent.json"
    ledger_to_json(multi_engine.mesh_governance.ledger, multi_ledger_path)
    print(f"Saved multi agent ledger to {multi_ledger_path}")

    print("\n" + "=" * 70)
    print("🛸 DEMO RUN COMPLETED SUCCESSFULLY! 🛸")
    print("=" * 70)


if __name__ == "__main__":
    main()
