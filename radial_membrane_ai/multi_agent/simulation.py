"""
Multi-Agent Simulation environment exploring coupled membranes and mesh governance.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.coupling import InterAgentVChannel, GlobalHolisticGovernor
from radial_membrane_ai.multi_agent.governance import MultiAgentMeshGovernance
from radial_membrane_ai.shard import ShardState


class MultiAgentSimulation:
    """
    Simulates a multi-agent ensemble of UFO agents sharing coupled membranes,
    inter-agent channels, global Holistic Governor, and federated governance.
    """

    def __init__(self, n_agents: int = 3) -> None:
        """
        Initializes the multi-agent simulation.

        Args:
            n_agents: Number of agents in the simulation (typically 3).
        """
        # Create agents with diverse focuses and cost sensitivities
        self.agents: List[UFOAgent] = []
        regimes = ["analytical", "creative", "balanced", "balanced"]
        sensitivities = [1.0, 2.0, 1.2, 1.0]

        for i in range(n_agents):
            self.agents.append(
                UFOAgent(
                    agent_id=f"agent_{i+1}",
                    cost_sensitivity=sensitivities[i % len(sensitivities)],
                    kernel_regime=regimes[i % len(regimes)],
                    state=ShardState.IDLE,
                    trust_score=0.9,
                    policy_compliance=0.95,
                    latency=12.0 + i * 5.0
                )
            )

        # Global layers
        self.global_governor = GlobalHolisticGovernor()
        self.mesh_governance = MultiAgentMeshGovernance(self.agents)

        # Inter-agent coupling V-channels
        self.channels: List[InterAgentVChannel] = []
        self._init_coupling_channels()

        # History tracking
        self.h_hol_history: List[float] = []
        self.c_mesh_history: List[float] = []
        self.agent_coherences: Dict[str, List[float]] = {a.agent_id: [] for a in self.agents}
        self.intervention_log: List[str] = []

    def _init_coupling_channels(self) -> None:
        """
        Sets up default inter-agent coupling V-Channels in a ring/coupled fashion.
        """
        n = len(self.agents)
        if n < 2:
            return

        for i in range(n):
            src = self.agents[i]
            tgt = self.agents[(i + 1) % n]
            # Set up Type-W (workload) and Type-T (tension) channels
            self.channels.append(InterAgentVChannel(src, tgt, "Type-W", coupling_strength=0.15))
            self.channels.append(InterAgentVChannel(tgt, src, "Type-T", coupling_strength=0.1))

    def run_step(self, scenario_type: str, task_value: float = 0.8) -> None:
        """
        Runs a single simulation cycle.

        Args:
            scenario_type: "cooperative", "competitive", "policy_tension", or "sao_block"
            task_value: Importance of the current task.
        """
        # Define base excitation based on scenario
        excitation = np.ones(12, dtype=np.float64) * 0.5

        if scenario_type == "cooperative":
            # Cooperative: Mutually reinforcing excitation in analytical quadrant (first 4 strings)
            excitation[0:4] = 0.95
        elif scenario_type == "competitive":
            # Competitive: Conflicting excitation pulling in different directions
            # Agent 1 (Analytical) vs. Agent 2 (Creative)
            excitation[0:3] = 0.9
            excitation[6:9] = 0.9
            # Boost cost factors to induce higher tension and cost-related collapse
            for idx, agent in enumerate(self.agents):
                agent.shard.cost_factor = 2.5 + idx
        elif scenario_type == "policy_tension":
            # Policy tension: Agent 1 does an locally valid routing but violates global compliance
            excitation[8] = 1.2  # high creativity
            self.agents[0].shard.policy_compliance = 0.15  # Heavy policy violation
            self.agents[0].shard.trust_score = 0.3
        elif scenario_type == "sao_block":
            # Force massive excitation on all strings to trigger SAO block
            excitation = np.ones(12, dtype=np.float64) * 5.0

        # 1. Update individual agents locally
        for agent in self.agents:
            # Skip if agent is isolated/quarantined in hard intervention
            if agent.shard.state == ShardState.QUARANTINED:
                continue
            agent.step(task_value, excitation)

        # 2. Propagate inter-agent coupling signals
        for channel in self.channels:
            channel.propagate()

        # 3. Compute global fields
        h_hol = self.global_governor.compute_global_holistic_field(self.agents)

        # Use custom weights to allow C_mesh to reach the green band (>= 0.7) under optimal conditions
        c_mesh = self.mesh_governance.compute_mesh_coherence(
            w_q=0.5, w_t=0.5, w_e=0.3, w_r=0.1, w_p=0.1, w_l=0.1, w_f=0.1
        )

        self.h_hol_history.append(h_hol)
        self.c_mesh_history.append(c_mesh)

        for agent in self.agents:
            self.agent_coherences[agent.agent_id].append(agent.shard.quality_score)

        # 4. Evaluate stability bands and apply interventions
        if c_mesh >= 0.7:
            # Green band: normal operation
            self.intervention_log.append("Mesh operating in Green Band (Nominal).")
        elif c_mesh >= 0.4:
            # Yellow band: Soft interventions (damping activations to reduce tension)
            self.intervention_log.append(f"Soft Intervention (Yellow Band): C_mesh={c_mesh:.4f}. Damping activations.")
            for agent in self.agents:
                for s in agent.membrane.strings:
                    s.activation *= 0.85
        else:
            # Red band: Hard interventions (heavy throttling, quarantine, fallback)
            self.intervention_log.append(
                f"Hard Intervention (Red Band): C_mesh={c_mesh:.4f}. "
                "Throttling, Quarantining, and Fallback."
            )
            # Throttle activations heavily
            for agent in self.agents:
                for s in agent.membrane.strings:
                    s.activation *= 0.5

            # Run mesh audit to quarantine low performing/violating agents
            self.mesh_governance.run_mesh_audit()

            # Fallback check: If all but one agent are quarantined or unhealthy, fallback to single agent
            active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
            if len(active_agents) == 1:
                self.intervention_log.append(f"Fallback triggered: Sole active agent is {active_agents[0].agent_id}.")

        # 5. Run SAO Paired promotions (promote representation from agent pairs)
        if len(self.agents) >= 2:
            # Pair consecutive agents
            for i in range(len(self.agents) - 1):
                agent_l = self.agents[i]
                agent_r = self.agents[i + 1]
                if agent_l.shard.state != ShardState.QUARANTINED and agent_r.shard.state != ShardState.QUARANTINED:
                    # Shared capacity limit is dynamically modulated by mesh coherence
                    limit = max(0.02, 0.5 * c_mesh)
                    verdict, p_sao, _ = self.mesh_governance.execute_sao_promotion(
                        agent_l, agent_r, shared_capacity_limit=limit
                    )
                    if verdict == "block":
                        self.intervention_log.append(
                            f"SAO Promotion Blocked between {agent_l.agent_id} and {agent_r.agent_id}. "
                            f"Residual p_SAO={p_sao:.4f} exceeds threshold."
                        )

    def run(self, n_steps: int, scenario_type: str) -> None:
        """
        Runs the simulation for a number of steps.

        Args:
            n_steps: Number of simulation steps.
            scenario_type: Scenario to execute.
        """
        for _ in range(n_steps):
            self.run_step(scenario_type)
