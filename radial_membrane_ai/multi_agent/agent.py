"""
UFOAgent class representing a governed multi-agent facet.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.semantic_memory.core import AgentSemanticMemory

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.governor import Governor
from radial_membrane_ai.envelope import BrimEnvelope
from radial_membrane_ai.shard import FederatedShard, ShardState
from radial_membrane_ai.channels import channel_coherence, update_radius_along_channel


class UFOAgent:
    """
    Represents an individual governed agent with its own membrane, strings,
    V-channel graph, bounded compute envelope, and local coherence score.
    """

    semantic_memory: AgentSemanticMemory

    def __init__(
        self,
        agent_id: str,
        cost_sensitivity: float = 1.0,
        kernel_regime: str = "balanced",
        state: ShardState = ShardState.IDLE,
        trust_score: float = 1.0,
        policy_compliance: float = 1.0,
        latency: float = 10.0
    ) -> None:
        """
        Initializes the UFOAgent.

        Args:
            agent_id: Unique identifier for the agent.
            cost_sensitivity: Factor scaling the local cost sensitivity.
            kernel_regime: The operational focus ("analytical", "creative", "balanced").
            state: Initial ShardState of the agent.
            trust_score: Initial trust score of the agent.
            policy_compliance: Policy compliance score of the agent.
            latency: Local latency in ms.
        """
        self.agent_id = agent_id
        self.cost_sensitivity = cost_sensitivity
        self.kernel_regime = kernel_regime

        # Core local components
        self.membrane = RadialMembrane()
        self.boundary = BoundaryGeometry()
        self.governor = Governor()
        self.envelope = BrimEnvelope()

        # Shard representation for mesh integration
        self.shard = FederatedShard(
            shard_id=agent_id,
            state=state,
            capacity=1.0,
            trust_score=trust_score,
            consent_granted=True,
            privacy_level=1.0,
            policy_compliance=policy_compliance,
            cost_factor=cost_sensitivity,
            latency=latency,
            quality_score=1.0
        )

        # Metrics history
        self.coherence_history: list[float] = []
        self.activation_history: list[np.ndarray] = []
        self.residual_history: list[float] = []

    def compute_local_coherence(self) -> float:
        """
        Computes the local coherence score C^k(t).
        Defined as the average V-Channel coherence across all string pairs.
        """
        coherences = []
        for i in range(12):
            for j in range(12):
                if i != j:
                    coherences.append(channel_coherence(self.membrane, i + 1, j + 1, samples=16))
        return float(np.mean(coherences)) if coherences else 1.0

    def step(
        self,
        task_value: float,
        excitation: np.ndarray,
        tool_loads: list[float] | np.ndarray | None = None,
        context_loads: list[float] | np.ndarray | None = None
    ) -> None:
        """
        Executes a single local update step for this agent.

        Args:
            task_value: Importance of the current task.
            excitation: Excitation input vector of length 12.
            tool_loads: Tool load vector.
            context_loads: Context load vector.
        """
        # Apply cost sensitivity to local governor configuration
        self.governor.config.suppression_weight = self.cost_sensitivity * 0.5

        # Apply kernel regime to excitation
        adjusted_excitation = np.copy(excitation)
        if self.kernel_regime == "analytical":
            # Boost analytical quadrant strings (index 1-4, which is 0-3 in 0-based index)
            adjusted_excitation[0:4] *= 1.5
        elif self.kernel_regime == "creative":
            # Boost creative / exploration strings (index 7-9, which is 6-8 in 0-based index)
            adjusted_excitation[6:9] *= 1.5

        # 1. Update activations via Governor
        self.governor.update_membrane(
            membrane=self.membrane,
            task_value=task_value,
            task_excitation=adjusted_excitation,
            tool_loads=tool_loads,
            context_loads=context_loads
        )

        # 2. Update reasoning radii along local V-channels
        r_max = 2.0
        old_radii = [s.radius for s in self.membrane.strings]
        for t_idx in range(12):
            target = self.membrane.strings[t_idx]
            # Internal growth
            r_internal = target.activation * r_max * 0.3

            # Propagated growth along V-channels
            r_propagated = 0.0
            for s_idx in range(12):
                if s_idx != t_idx:
                    source = self.membrane.strings[s_idx]
                    orig_radius = source.radius
                    source.radius = old_radii[s_idx]

                    r_prop = update_radius_along_channel(
                        source=source,
                        target=target,
                        r_max=r_max
                    )
                    source.radius = orig_radius
                    if r_prop > r_propagated:
                        r_propagated = r_prop

            target_radius_target = max(r_internal, r_propagated)
            target.radius = 0.8 * target.radius + 0.2 * target_radius_target

        # 3. Update boundary geometry
        self.boundary.update_boundary(
            membrane=self.membrane,
            task_value=task_value
        )

        # 4. Check compute envelope (Brim) and update shard state
        verdict, meta = self.envelope.evaluate_envelope(self.membrane, self.boundary)
        if verdict == "block":
            for s in self.membrane.strings:
                s.activation *= 0.5
            self.shard.state = ShardState.SATURATED
        elif verdict == "constrain":
            for s in self.membrane.strings:
                s.activation *= 0.8

        # Update shard representation metrics
        avg_act = float(np.mean([s.activation for s in self.membrane.strings]))
        self.shard.capacity = max(0.1, 2.0 - avg_act)
        self.shard.quality_score = self.compute_local_coherence()

        # Track history
        self.coherence_history.append(self.shard.quality_score)
        self.activation_history.append(self.membrane.get_activation_vector())
        self.residual_history.append(meta.get("brim_energy", 0.0))
