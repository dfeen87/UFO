# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Inter-agent coupling layer and global Holistic Governor.
"""

from __future__ import annotations
import numpy as np

from radial_membrane_ai.multi_agent.agent import UFOAgent


class InterAgentVChannel:
    """
    Represents a typed V-Channel between two agents carrying signals between membranes.
    """

    def __init__(
        self,
        source_agent: UFOAgent,
        target_agent: UFOAgent,
        channel_type: str = "Type-W",
        coupling_strength: float = 0.1
    ) -> None:
        """
        Initializes the InterAgentVChannel.

        Args:
            source_agent: The source UFOAgent.
            target_agent: The target UFOAgent.
            channel_type: The type of signal carried ("Type-W" for workload/activation,
                          "Type-T" for tension/stiffness, "Type-R" for residuals).
            coupling_strength: Scaling factor for the signal propagation.
        """
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.channel_type = channel_type
        self.coupling_strength = coupling_strength

    def propagate(self) -> None:
        """
        Propagates signals from source agent to target agent.
        """
        if self.source_agent.shard.state.name in ("QUARANTINED", "REVOKED"):
            return  # Isolated/quarantined agents cannot propagate signals

        for i in range(12):
            src_str = self.source_agent.membrane.strings[i]
            tgt_str = self.target_agent.membrane.strings[i]

            if self.channel_type == "Type-W":
                # Workload propagation: source activation reinforces target activation
                delta_act = self.coupling_strength * src_str.activation
                tgt_str.activation = max(0.0, min(1.0, tgt_str.activation + delta_act))

            elif self.channel_type == "Type-T":
                # Tension propagation: high tension in source increases target cost/tension
                delta_tension = self.coupling_strength * src_str.tension
                tgt_str.tension = max(0.0, min(1.0, tgt_str.tension + delta_tension))

            elif self.channel_type == "Type-R":
                # Residual/Error propagation: source cost influences target cost
                delta_cost = self.coupling_strength * src_str.cost
                tgt_str.cost += delta_cost


class GlobalHolisticGovernor:
    """
    A global supervisor integrating all agents, string states, V-channel metrics,
    bounded envelopes, and global policy goals to evaluate global system coherence H_hol(t).
    """

    def __init__(
        self,
        w_a: float = 0.4,
        w_c: float = 0.4,
        w_r: float = 0.2
    ) -> None:
        """
        Initializes the GlobalHolisticGovernor.

        Args:
            w_a: Weight for activation average.
            w_c: Weight for coherence average.
            w_r: Weight for residual/energy average.
        """
        self.w_a = w_a
        self.w_c = w_c
        self.w_r = w_r
        self.h_history: list[float] = []

    def compute_global_holistic_field(self, agents: list[UFOAgent]) -> float:
        """
        Computes the global Holistic Governor field H_hol(t):
        H_hol(t) = (1/N) * sum_k [ (w_a * A_avg^k + w_c * C_avg^k - w_r * E_energy^k) * Policy_compliance^k ]

        Args:
            agents: List of UFOAgent instances in the multi-agent ensemble.

        Returns:
            Scalar representation of global field coherence.
        """
        if not agents:
            return 0.0

        field_sum = 0.0
        for agent in agents:
            # S^k - average activation
            a_avg = float(np.mean([s.activation for s in agent.membrane.strings]))

            # V^k - local coherence score C^k
            c_val = agent.compute_local_coherence()

            # E^k - bounded compute envelope brim energy
            brim_energy = agent.envelope.compute_brim_energy(agent.membrane, agent.boundary, samples=12)

            # Policy compliance P^k
            p_comp = agent.shard.policy_compliance

            agent_term = (self.w_a * a_avg + self.w_c * c_val - self.w_r * brim_energy) * p_comp
            field_sum += agent_term

        h_hol = field_sum / len(agents)
        self.h_history.append(h_hol)
        return h_hol
