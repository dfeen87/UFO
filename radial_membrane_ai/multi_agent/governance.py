"""
Federated shard mesh governance, mesh coherence score, and SAO promotion logic.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List, Tuple

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.shard import ShardState
from radial_membrane_ai.residuals import ResidualLedger


class MultiAgentMeshGovernance:
    """
    Manages the federated mesh of UFO agents, performing audit operations,
    computing the global mesh coherence score, and routing SAO promotions.
    """

    def __init__(self, agents: List[UFOAgent]) -> None:
        """
        Initializes the mesh governance with a set of agents.

        Args:
            agents: List of UFOAgent instances.
        """
        self.agents = agents
        self.ledger = ResidualLedger()
        self.promotion_history: List[Dict[str, Any]] = []

    def compute_mesh_coherence(
        self,
        w_q: float = 0.2,
        w_t: float = 0.2,
        w_e: float = 0.15,
        w_r: float = 0.15,
        w_p: float = 0.1,
        w_l: float = 0.1,
        w_f: float = 0.1
    ) -> float:
        """
        Computes the global mesh coherence score C_mesh(t):
        C_mesh(t) = normalize(w_q * Q_route + w_t * T_trust + w_e * E_economic -
                              w_r * R_residual - w_p * P_policy - w_l * L_latency - w_f * F_failure)
        """
        if not self.agents:
            return 1.0

        # Extract metrics
        # 1. Q_route: Average quality across active agent shards
        active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
        if active_agents:
            q_route = sum(a.shard.quality_score for a in active_agents) / len(active_agents)
        else:
            q_route = 0.5

        # 2. T_trust: Average trust score of the agents
        t_trust = sum(a.shard.trust_score for a in self.agents) / len(self.agents)

        # 3. E_economic: Average cost factor (economic efficiency)
        # Higher cost factor means lower efficiency: 1.0 / (cost_factor + 1.0)
        e_economic = sum(1.0 / (a.shard.cost_factor + 1.0) for a in self.agents) / len(self.agents)

        # 4. R_residual: Average residual magnitude from agents' Brim envelopes
        r_residual = sum(a.residual_history[-1] if a.residual_history else 0.0 for a in self.agents) / len(self.agents)

        # 5. P_policy: Average policy compliance penalty (1.0 - policy_compliance)
        p_policy = sum(1.0 - a.shard.policy_compliance for a in self.agents) / len(self.agents)

        # 6. L_latency: Normalized latency (e.g., average latency in ms / 100.0)
        l_latency = sum(a.shard.latency for a in self.agents) / len(self.agents) / 100.0

        # 7. F_failure: Failure rate (fraction of quarantined or saturated agents)
        failures = len([a for a in self.agents if a.shard.state in (ShardState.QUARANTINED, ShardState.SATURATED)])
        f_failure = failures / len(self.agents)

        # Compute raw term
        raw_val = (
            w_q * q_route +
            w_t * t_trust +
            w_e * e_economic -
            w_r * r_residual -
            w_p * p_policy -
            w_l * l_latency -
            w_f * f_failure
        )

        # Apply sigmoid normalization to map to [0, 1]
        return 1.0 / (1.0 + math.exp(-raw_val))

    def run_mesh_audit(self) -> None:
        """
        Audits all agent shards in the mesh. Quarantines agents that exhibit
        high residual/Brim energy or low trust and policy compliance.
        """
        for agent in self.agents:
            # Audit based on latest residual energy and trust
            last_residual = agent.residual_history[-1] if agent.residual_history else 0.0
            if (last_residual > 1.5) or (agent.shard.trust_score < 0.4) or (agent.shard.policy_compliance < 0.3):
                agent.shard.state = ShardState.QUARANTINED
                # Log a failure in the residual ledger
                self.ledger.log_failure(
                    record_id=f"audit_quarantine_{agent.agent_id}_{len(self.ledger.records)}",
                    error_type="governance_quarantine",
                    shard_id=agent.agent_id,
                    severity="critical",
                    details={"residual": last_residual, "trust": agent.shard.trust_score}
                )

    def execute_sao_promotion(
        self,
        agent_l: UFOAgent,
        agent_r: UFOAgent,
        shared_capacity_limit: float = 0.8
    ) -> Tuple[str, float, np.ndarray]:
        """
        Executes a Symmetric Ascension Operator (SAO) promotion gate for a pair of agents.

        Steps:
        1. Extract paired states x = (x_L, x_R) from both agents (activation vectors).
        2. Perform symmetry alignment S(x) -> x_sym = (x_L + x_R) / 2.
        3. Admissibility test against a shared receiving layer capacity.
        4. Project x_sym to admissible space if it exceeds the limit.
        5. Log the promotion residual p_SAO for audit.
        6. Preserve residuals within each agent.

        Args:
            agent_l: The left UFOAgent.
            agent_r: The right UFOAgent.
            shared_capacity_limit: The shared admissible capacity of the receiving layer.

        Returns:
            A tuple (verdict, p_sao, projected_state).
        """
        # 1. Extract paired states (activation vectors)
        x_l = agent_l.membrane.get_activation_vector()
        x_r = agent_r.membrane.get_activation_vector()

        # 2. Symmetry alignment S(x) -> x_sym
        x_sym = (x_l + x_r) / 2.0

        # 3. Admissibility test against shared capacity
        # Find maximum activation in the aligned state
        max_act = float(np.max(x_sym))

        # 4. Projection P(x_sym)
        # Projects to admissible space: clamp to shared_capacity_limit
        x_proj = np.minimum(x_sym, shared_capacity_limit)

        # 5. Residual separation p_SAO
        p_sao_vector = x_sym - x_proj
        p_sao = float(np.linalg.norm(p_sao_vector))

        # Determine promotion verdict
        # - Block if residual is extremely high
        # - Constrain if residual is moderate
        # - Ascend if admissible and aligned
        if p_sao > 0.4:
            verdict = "block"
            # Add penalty to agent compliance and trust for audit tracking
            agent_l.shard.policy_compliance = max(0.1, agent_l.shard.policy_compliance - 0.1)
            agent_r.shard.policy_compliance = max(0.1, agent_r.shard.policy_compliance - 0.1)
        elif p_sao > 0.15:
            verdict = "constrain"
        else:
            verdict = "ascend"

        # Log promotion residual for audit
        promo_record = {
            "agent_l": agent_l.agent_id,
            "agent_r": agent_r.agent_id,
            "p_sao": p_sao,
            "verdict": verdict,
            "aligned_state": x_sym,
            "projected_state": x_proj
        }
        self.promotion_history.append(promo_record)

        # Log failed/constrained promotions in the ledger
        if verdict in ("block", "constrain"):
            self.ledger.log_failure(
                record_id=f"sao_fail_{len(self.ledger.records)}",
                error_type="sao_ascension_rejection",
                shard_id=f"{agent_l.agent_id}_and_{agent_r.agent_id}",
                severity="high" if verdict == "block" else "medium",
                details={"p_sao": p_sao, "verdict": verdict}
            )

        # Preserve residuals across agents: append promotion residual to agent residual history
        agent_l.residual_history.append(p_sao)
        agent_r.residual_history.append(p_sao)

        return verdict, p_sao, x_proj
