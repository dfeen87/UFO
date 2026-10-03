#!/usr/bin/env python3
# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Dynamic Terminal Execution Interface for the U.F.O. Governed Runtime.
"""

from __future__ import annotations
import os
import sys
import argparse
import time
import numpy as np
from typing import List, Any

# Import package components
from radial_membrane_ai.utils import set_deterministic_env
from radial_membrane_ai import __version__
from radial_membrane_ai.exceptions import GovernanceError
from radial_membrane_ai.ufo_engine import SingleAgentEngine, MultiAgentEngine
from radial_membrane_ai.agentic.engine import AgenticEngine, AgenticSwarmEngine
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.visualization.visualizer import MeshVisualizer
from radial_membrane_ai.workloads.workload import StabilityBand, SAOLevel, EnvelopeState
from radial_membrane_ai.workloads.engine import (
    WorkloadFrame,
    WorkloadFrameMetrics,
    WorkloadTrace,
    WorkloadResult,
    RegimeTransition,
    SAOEvent
)

# Colors and spacing formatting constants
ANSI_PURPLE = "\033[95m"
ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def print_banner(text: str, color: str = ANSI_CYAN) -> None:
    border = "=" * 75
    print(f"{color}{border}")
    print(f"{ANSI_BOLD}{text.center(75)}{ANSI_RESET}{color}")
    print(f"{border}{ANSI_RESET}\n")


def format_metric(name: str, value: Any, color: str) -> str:
    return f"{name}: {color}{value}{ANSI_RESET}"


def append_to_halt_log(message: str) -> None:
    os.makedirs("logs", exist_ok=True)
    with open("logs/governance_halts.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def run_agentic_cli(args: argparse.Namespace) -> None:
    if not args.quiet:
        print_banner(f"🛸 UFO VERSION {__version__} AGENTIC AI RUNTIME 🛸", ANSI_CYAN)

    agentic_engine = AgenticEngine()
    goal = "Execute code calculation and search knowledge base for UFO architecture"
    res = agentic_engine.run_goal(goal=goal, max_steps=args.steps or 10)

    if not args.quiet:
        print(f"{ANSI_BOLD}Goal:{ANSI_RESET} {res.plan.goal}")
        print(f"{ANSI_BOLD}Plan Status:{ANSI_RESET} {res.plan.status}")
        print(f"{ANSI_BOLD}Total Plan Cost:{ANSI_RESET} {res.plan.total_cost:.4f}")
        print(f"{ANSI_BOLD}Total Plan Tension:{ANSI_RESET} {res.plan.total_tension:.4f}")
        print(f"{ANSI_BOLD}Reflections Triggered:{ANSI_RESET} {len(res.reflections)}")
        for i, step in enumerate(res.plan.steps, 1):
            print(f"  [{i}] {step.description} -> Completed: {step.completed} | Tool: {step.tool_name}")
            if step.result:
                print(f"      Out: {step.result.output[:80]}...")


def run_agentic_swarm_cli(args: argparse.Namespace) -> None:
    if not args.quiet:
        print_banner(f"🛸 UFO VERSION {__version__} AGENTIC SWARM RUNTIME 🛸", ANSI_PURPLE)

    swarm_engine = AgenticSwarmEngine()
    goals = [
        {"goal": "Calculate factorial series with python code", "required_capabilities": ["code"], "max_budget": 8.0},
        {
            "goal": "Search UFO architecture docs and retrieve memory",
            "required_capabilities": ["search"],
            "max_budget": 5.0,
        },
    ]

    swarm_res = swarm_engine.run_swarm_goals(goals, ticks_per_contract=5)

    if not args.quiet:
        print(f"{ANSI_BOLD}Swarm Contracts Awarded:{ANSI_RESET} {len(swarm_res.contracts)}")
        for contract in swarm_res.contracts:
            winner = contract.assigned_agent_id
            score = contract.winning_bid.bid_score if contract.winning_bid else 0.0
            print(
                f"  • Contract [{contract.contract_id}]: {contract.goal[:40]}... -> "
                f"Awarded To: {winner} (Score: {score:.2f})"
            )


def run_simulation(args: argparse.Namespace) -> None:
    # 1. Deterministic Seeding Confirmation
    seed = args.seed if args.seed is not None else 0
    set_deterministic_env(seed=seed)
    if not args.quiet:
        print(f"{ANSI_GREEN}{ANSI_BOLD}[Deterministic Seed Initialized: {seed}]{ANSI_RESET}\n")

    # 2. Select Engine Mode
    # Resolution rules:
    # --single takes priority, --multi is other mode. argparse groups can handle mutual exclusion,
    # but let's enforce custom messaging.
    if args.single and args.multi:
        print(f"{ANSI_RED}{ANSI_BOLD}Governance Error: Ambiguous execution. "
              f"Both --single and --multi were passed.{ANSI_RESET}")
        sys.exit(1)
        return

    # Resolve mode based on arguments
    mode = "single"
    if args.single:
        mode = "single"
    elif args.multi:
        mode = "multi"
    elif args.mode:
        mode = args.mode

    steps = args.steps if args.steps is not None else (5 if args.demo or args.demo_governance else 10)

    # 3. Handle Demo vs Custom inputs
    if args.demo_governance:
        # Intentionally escalate to trigger GovernanceError
        # A single-agent high excitation / high tension tick triggers hard limits immediately
        task_values = [0.9] * steps
        excitations = [np.ones(12) * 8.5] * steps
    elif args.demo:
        # Run a safe 5-step simulation
        task_values = [0.8, 0.85, 0.9, 0.75, 0.7]
        excitations = [np.ones(12) * 0.1, np.ones(12) * 0.15, np.ones(12) * 0.1, np.ones(12) * 0.05, np.ones(12) * 0.01]
        steps = 5
    else:
        # Standard run with custom arguments
        task_value_val = args.task_value if args.task_value is not None else 0.8
        excitation_val = args.excitation if args.excitation is not None else 0.5
        task_values = [task_value_val] * steps
        excitations = [np.ones(12) * excitation_val] * steps

    # Initialize Trace collection
    trace_frames: List[WorkloadFrame] = []
    regime_transitions: List[RegimeTransition] = []
    sao_events: List[SAOEvent] = []

    # Initialize Engine
    engine_sa: SingleAgentEngine | None = None
    engine_ma: MultiAgentEngine | None = None

    legacy_hs = getattr(args, "legacy_handshake", False)
    legacy_m = getattr(args, "legacy_mode", "strict") or "strict"

    if mode == "single":
        if not args.quiet:
            print_banner("🛸 UFO GOVERNED SINGLE-AGENT SIMULATION RUNTIME 🛸", ANSI_PURPLE)
        engine_sa = SingleAgentEngine(
            seed=seed,
            enable_legacy_handshake=legacy_hs,
            legacy_handshake_mode=legacy_m,
        )
    else:
        if not args.quiet:
            print_banner("🛸 UFO GOVERNED MULTI-AGENT SIMULATION RUNTIME 🛸", ANSI_BLUE)
        engine_ma = MultiAgentEngine(
            n_agents=3,
            seed=seed,
            enable_legacy_handshake=legacy_hs,
            legacy_handshake_mode=legacy_m,
        )

    # Summary metric trackers
    max_curvature_observed = 0.0
    max_tension_observed = 0.0
    coherence_list: List[float] = []
    prev_regime: str | None = None
    rollbacks_count = 0
    quarantines_count = 0

    halt_message = (
        "GovernanceError correctly raised during local simulation due to temporal tension exceeding hard limit. "
        "Confirms stability enforcement and deterministic halt behavior."
    )

    try:
        for step_idx in range(steps):
            tick_num = step_idx + 1

            if not args.quiet:
                # Dynamic tick animation display
                anim_chars = ["◐", "◓", "◑", "◒"]
                anim = anim_chars[step_idx % len(anim_chars)]
                print(f"{ANSI_BOLD}{ANSI_CYAN}{anim} Executing Tick {tick_num}/{steps}...{ANSI_RESET}", end="\r")
                sys.stdout.flush()
                time.sleep(0.05)  # slight delay for animation feedback

            # Capture state pre-tick to monitor transitions
            if mode == "single":
                assert engine_sa is not None
                current_regime = (
                    engine_sa.regime_manager.get_regime_for_agent("single_agent").regime_type.value
                )
            else:
                assert engine_ma is not None
                current_regime = (
                    engine_ma.regime_manager.get_regime_for_agent("agent_1").regime_type.value
                )

            if prev_regime is not None and current_regime != prev_regime:
                regime_transitions.append(RegimeTransition(
                    step_index=step_idx, from_regime=prev_regime, to_regime=current_regime
                ))
            prev_regime = current_regime

            # Execute tick
            t_val = task_values[step_idx % len(task_values)]
            excite = excitations[step_idx % len(excitations)]

            if mode == "single":
                assert engine_sa is not None
                band = engine_sa.tick(t_val, excite)
                interventions = engine_sa.interventions
            else:
                assert engine_ma is not None
                band = engine_ma.tick(t_val, excite)
                interventions = engine_ma.interventions

            # Monitor interventions for rollbacks/quarantines
            for inter in interventions[-3:]:
                if "Rollback" in inter or "rollback" in inter:
                    rollbacks_count += 1
                if "quarantine" in inter or "Quarantine" in inter:
                    quarantines_count += 1

            # Metric updates
            if mode == "single":
                assert engine_sa is not None
                curv = float(max([engine_sa.boundary.curvature(s.theta) for s in engine_sa.membrane.strings]))
                t_state = getattr(engine_sa.membrane, "temporal_state", None)
                tens = t_state.accumulated_tension if t_state is not None else 0.0
                coh = engine_sa.compute_local_coherence()
                active_sao = len(engine_sa.sao_events)
            else:
                assert engine_ma is not None
                curvatures = [
                    max([a.boundary.curvature(s.theta) for s in a.membrane.strings])
                    for a in engine_ma.agents if a.shard.state != ShardState.QUARANTINED
                ]
                curv = float(np.mean(curvatures)) if curvatures else 0.0
                tens = engine_ma.temporal_state.accumulated_tension
                active_agents = [a for a in engine_ma.agents if a.shard.state != ShardState.QUARANTINED]
                from radial_membrane_ai.collective_reasoning.coherence import coherence_score
                coh = coherence_score(active_agents) if active_agents else 1.0
                active_sao = len(engine_ma.sao_events)

            max_curvature_observed = max(max_curvature_observed, curv)
            max_tension_observed = max(max_tension_observed, tens)
            coherence_list.append(coh)

            # Track frames for optional visualization
            band_upper = band.upper()
            metrics_obj = WorkloadFrameMetrics(
                step_index=step_idx,
                curvature=curv,
                tension=tens,
                coherence=coh,
                regime=current_regime,
                stability_band=(
                    StabilityBand[band_upper]
                    if band_upper in StabilityBand.__members__
                    else StabilityBand.GREEN
                ),
                sao_level=SAOLevel.MID if active_sao > len(sao_events) else SAOLevel.NONE,
                envelope_state=EnvelopeState.ADMIT
            )
            if active_sao > len(sao_events):
                sao_events.append(SAOEvent(step_index=step_idx, level=SAOLevel.MID))

            from radial_membrane_ai.visualization.visualizer import extract_geom_and_vchannel
            if mode == "single":
                assert engine_sa is not None
                geom, vch = extract_geom_and_vchannel(engine_sa.membrane, engine_sa.boundary)
            else:
                assert engine_ma is not None
                from radial_membrane_ai.multi_agent.cluster import UFOCluster
                temp_c = UFOCluster(cluster_id="global_mesh")
                temp_c.agents = engine_ma.agents
                temp_c.update_cluster_membrane()
                geom, vch = extract_geom_and_vchannel(temp_c.membrane, temp_c.boundary)

            trace_frames.append(WorkloadFrame(metrics=metrics_obj, membrane_geometry=geom, vchannels=vch))

            # Tick-by-tick Output Style
            if not args.quiet:
                # Color code metrics
                curv_str = format_metric("curvature", f"{curv:.4f}", ANSI_PURPLE)
                tens_str = format_metric("tension", f"{tens:.4f}", ANSI_BLUE)
                coh_str = format_metric("coherence", f"{coh:.4f}", ANSI_GREEN)
                band_str = format_metric("stability band", f"{band.upper()}", ANSI_YELLOW)

                tick_p = f"[{ANSI_BOLD}Tick {tick_num:2d}/{steps:2d}{ANSI_RESET}]"
                print(f"{tick_p} - {curv_str} | {tens_str} | {coh_str} | {band_str}")

                # Verbose Telemetry
                if args.verbose:
                    print(f"  {ANSI_BOLD}Regime:{ANSI_RESET} {current_regime.upper()}")
                    if mode == "single":
                        assert engine_sa is not None
                        st_max = max(s.stiffness for s in engine_sa.membrane.strings)
                        print(f"    - Lyapunov Energy: {engine_sa.v_history[-1]:.4f}")
                        print(f"    - Stiffness Max: {st_max:.4f}")
                    else:
                        assert engine_ma is not None
                        for agent in engine_ma.agents:
                            q_state = agent.shard.state.name
                            trust = agent.shard.trust_score
                            compl = agent.shard.policy_compliance
                            print(f"    - {agent.agent_id}: State={q_state}, "
                                  f"Trust={trust:.2f}, Compliance={compl:.2f}")

    except GovernanceError as e:
        # Handle GovernanceError with descriptive messaging and exact halt logging
        print("\n" + "=" * 75)
        print(f"{ANSI_RED}{ANSI_BOLD}🚨 GOVERNED HALT ENFORCED 🚨{ANSI_RESET}")
        print("=" * 75)
        print(f"{ANSI_RED}{ANSI_BOLD}Violation Details:{ANSI_RESET} {e}")
        print(f"{ANSI_YELLOW}{ANSI_BOLD}Halt Ledger Code:{ANSI_RESET} {halt_message}")
        print("=" * 75 + "\n")

        # Logging to governance_halts.log and stdout
        append_to_halt_log(halt_message)

        # Print final summary block even upon early halt
        avg_coherence = float(np.mean(coherence_list)) if coherence_list else 1.0
        print_summary_block(
            max_curvature=max_curvature_observed,
            max_tension=max_tension_observed,
            coherence_stability=avg_coherence,
            transitions=len(regime_transitions),
            promotions=len(sao_events),
            rollbacks=rollbacks_count,
            quarantines=quarantines_count,
            halted=True,
            halt_reason=str(e),
            args=args
        )
        sys.exit(1)

    # 4. Final Clean Execution (No GovernanceError)
    avg_coherence = float(np.mean(coherence_list)) if coherence_list else 1.0
    print_summary_block(
        max_curvature=max_curvature_observed,
        max_tension=max_tension_observed,
        coherence_stability=avg_coherence,
        transitions=len(regime_transitions),
        promotions=len(sao_events),
        rollbacks=rollbacks_count,
        quarantines=quarantines_count,
        halted=False,
        halt_reason=None,
        args=args
    )

    # 5. Optional Visualization export
    if args.visualize:
        if not args.quiet:
            print(f"{ANSI_CYAN}Generating full simulation visual report...{ANSI_RESET}")
        visualizer = MeshVisualizer(samples_resolution=100)
        trace_obj = WorkloadTrace(frames=trace_frames, result=WorkloadResult(frames=[f.metrics for f in trace_frames]))
        report = visualizer.render_workload_trace(trace_obj)
        os.makedirs("logs/visualization", exist_ok=True)
        out_png_path = "logs/visualization/ufo_simulation_timeline.png"
        if report.rendered is not None:
            report.rendered.save(out_png_path, dpi=(120, 120))
            if not args.quiet:
                print(f"{ANSI_GREEN}Saved visualization dashboard to: {ANSI_BOLD}{out_png_path}{ANSI_RESET}")


def print_summary_block(
    max_curvature: float,
    max_tension: float,
    coherence_stability: float,
    transitions: int,
    promotions: int,
    rollbacks: int,
    quarantines: int,
    halted: bool,
    halt_reason: str | None,
    args: argparse.Namespace
) -> None:
    # Deterministic final governed summary block format
    border = "*" * 75
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{border}")
    print("🛸 GOVERNED SIMULATION FINAL SUMMARY".center(75))
    print(f"{border}{ANSI_RESET}")

    status_str = f"{ANSI_RED}HALTED (Governance Breach)" if halted else f"{ANSI_GREEN}COMPLETED (Nominal)"
    print(f"  • {ANSI_BOLD}Simulation Status:{ANSI_RESET} {status_str}")
    if halted and halt_reason:
        print(f"    {ANSI_RED}- Reason:{ANSI_RESET} {halt_reason}")

    print(f"  • {ANSI_BOLD}Maximum Curvature:{ANSI_RESET} {ANSI_PURPLE}{max_curvature:.4f}{ANSI_RESET}")
    print(f"  • {ANSI_BOLD}Maximum Tension:{ANSI_RESET} {ANSI_BLUE}{max_tension:.4f}{ANSI_RESET}")
    c_stable_msg = f"{coherence_stability:.4f}"
    print(f"  • {ANSI_BOLD}Coherence Stability (Average):{ANSI_RESET} {ANSI_GREEN}{c_stable_msg}{ANSI_RESET}")
    print(f"  • {ANSI_BOLD}Regime Transitions:{ANSI_RESET} {transitions}")
    print(f"  • {ANSI_BOLD}SAO Promotions Event Count:{ANSI_RESET} {promotions}")
    print(f"  • {ANSI_BOLD}Rollbacks / Restorations Enforced:{ANSI_RESET} {rollbacks}")
    print(f"  • {ANSI_BOLD}Quarantine Enforcements:{ANSI_RESET} {quarantines}")
    print(f"{ANSI_CYAN}{border}{ANSI_RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic Terminal Interface for the U.F.O. Governed Runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"UFO {__version__}")

    # Simulation modes
    parser.add_argument("--single", action="store_true", help="Run in Single-Agent Mode (Default)")
    parser.add_argument("--multi", action="store_true", help="Run in Multi-Agent Mode")
    parser.add_argument("--agentic", action="store_true", help="Run in Version 4 Agentic AI Mode")
    parser.add_argument("--agentic-swarm", action="store_true", help="Run in Version 4 Agentic Swarm Mode")
    parser.add_argument(
        "--mode", choices=["single", "multi", "agentic", "agentic-swarm"], help="Select Simulation Mode"
    )

    # Parameters
    parser.add_argument("--steps", type=int, help="Number of ticks / execution steps")
    parser.add_argument("--excitation", type=float, help="Constant excitation sequence factor [0.0 - 10.0]")
    parser.add_argument("--task-value", type=float, help="Constant task value factor [0.0 - 1.0]")
    parser.add_argument("--seed", type=int, help="Deterministic initialization seed (default: 0)")

    # Execution Options
    parser.add_argument(
        "--legacy-handshake", action="store_true",
        help="Enable Invariant Handshake for Legacy hardware interoperability"
    )
    parser.add_argument(
        "--legacy-mode", choices=["strict", "soft", "simulation"], default="soft",
        help="Legacy handshake mode (default: soft)"
    )
    parser.add_argument("--visualize", action="store_true", help="Save timeline PNG under logs/visualization/")
    parser.add_argument("--quiet", action="store_true", help="CI/CD quiet mode: only output final summary")
    parser.add_argument("--verbose", action="store_true", help="Full governed telemetry trace output")
    parser.add_argument("--demo", action="store_true", help="Run 5-step safe simulation demo")
    parser.add_argument(
        "--demo-governance", action="store_true",
        help="Trigger early GovernanceError demonstration"
    )

    args = parser.parse_args()

    try:
        if getattr(args, "agentic", False) or getattr(args, "mode", None) == "agentic":
            run_agentic_cli(args)
        elif getattr(args, "agentic_swarm", False) or getattr(args, "mode", None) == "agentic-swarm":
            run_agentic_swarm_cli(args)
        else:
            run_simulation(args)
    except KeyboardInterrupt:
        print(f"\n{ANSI_YELLOW}Execution interrupted by user. Exiting cleanly...{ANSI_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
