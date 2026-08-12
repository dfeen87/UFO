# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
UFOCluster abstraction representing a governed cluster-level entity on top of agents.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.semantic_memory.core import MeshSemanticMemory
from radial_membrane_ai.semantic_memory.curvature import MemoryCurvatureState
from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.governance import MultiAgentMeshGovernance
from radial_membrane_ai.shard import ShardState


class UFOCluster:
    """
    Represents a governed cluster of UFO agents.
    Each cluster has its own membrane, governance, semantic memory,
    and local stability physics, acting as a first-class entity.
    """

    def __init__(
        self,
        cluster_id: str,
        role: str = "general",
        config: Optional[Dict[str, Any]] = None,
        agents: Optional[List[UFOAgent]] = None
    ) -> None:
        """
        Initializes the UFOCluster.

        Args:
            cluster_id: Unique identifier for the cluster.
            role: Functional role/profile of the cluster (e.g. "analytical", "creative").
            config: Key-value dictionary for custom parameters.
            agents: Optional initial list of UFOAgents in this cluster.
        """
        self.cluster_id = cluster_id
        self.role = role
        self.config = config if config is not None else {}

        # 1. Cluster-level membrane and boundary geometry
        self.membrane = RadialMembrane()
        self.boundary = BoundaryGeometry()

        # 2. Cluster-local semantic memory field
        self.semantic_memory = MeshSemanticMemory()

        # 3. Cluster-local stability state
        self.stability_curvature = MemoryCurvatureState()
        self.stability_band: str = "green"  # "green", "yellow", or "red"
        self.tension_metric: float = 0.0
        self.quarantine_timer: int = 0
        self.coherence_history: List[float] = []
        self.tension_history: List[float] = []

        # 4. Agents assigned to this cluster
        self.agents: List[UFOAgent] = []
        if agents is not None:
            for agent in agents:
                self.add_agent(agent)

        # 5. Local Cluster-level governance layer
        self.mesh_governance = MultiAgentMeshGovernance(self.agents)

    def add_agent(self, agent: UFOAgent) -> None:
        """
        Assigns an agent to this cluster and updates governance.
        """
        if agent not in self.agents:
            self.agents.append(agent)
            self.mesh_governance = MultiAgentMeshGovernance(self.agents)

    def remove_agent(self, agent_id: str) -> Optional[UFOAgent]:
        """
        Removes an agent from this cluster by agent_id and updates governance.
        """
        for agent in self.agents:
            if agent.agent_id == agent_id:
                self.agents.remove(agent)
                self.mesh_governance = MultiAgentMeshGovernance(self.agents)
                return agent
        return None

    def compute_cluster_coherence(self) -> float:
        """
        Computes the aggregate cluster coherence.
        """
        if not self.agents:
            return 1.0
        return self.mesh_governance.compute_mesh_coherence()

    def update_cluster_membrane(self) -> None:
        """
        Updates cluster-level membrane states (activations, radii) and boundary
        as an aggregate over active/non-quarantined cluster agents.
        Also updates cluster stability physics (tension and stability bands).
        """
        active_agents = [a for a in self.agents if a.shard.state != ShardState.QUARANTINED]
        if not active_agents:
            # Revert to default/flat membrane if no active agents
            for s in self.membrane.strings:
                s.activation = 0.0
                s.radius = 0.0
                s.tension = 0.0
            self.tension_metric = 0.0
            self.stability_band = "green"
            return

        # Aggregate activation, radius and tension
        avg_activations = np.zeros(12, dtype=np.float64)
        avg_radii = np.zeros(12, dtype=np.float64)
        avg_tensions = np.zeros(12, dtype=np.float64)

        for agent in active_agents:
            avg_activations += agent.membrane.get_activation_vector()
            avg_radii += np.array([s.radius for s in agent.membrane.strings], dtype=np.float64)
            avg_tensions += np.array([s.tension for s in agent.membrane.strings], dtype=np.float64)

        avg_activations /= len(active_agents)
        avg_radii /= len(active_agents)
        avg_tensions /= len(active_agents)

        for i, s in enumerate(self.membrane.strings):
            s.activation = float(avg_activations[i])
            s.radius = float(avg_radii[i])
            s.tension = float(avg_tensions[i])

        # Aggregate boundary geometry
        avg_deviations = {idx: 0.0 for idx in range(1, 13)}
        for agent in active_agents:
            for idx in range(1, 13):
                avg_deviations[idx] += agent.boundary.radius_deviation[idx]
        for idx in range(1, 13):
            avg_deviations[idx] /= len(active_agents)
        self.boundary.radius_deviation = avg_deviations

        # Update stability physics metrics
        # Tension is derived from string tensions + memory curvature/tension
        self.tension_metric = float(np.mean(avg_tensions)) + self.stability_curvature.tension
        self.tension_history.append(self.tension_metric)

        coherence = self.compute_cluster_coherence()
        self.coherence_history.append(coherence)

        # Classify local stability band based on coherence and tension
        if coherence >= 0.7 and self.tension_metric < 1.0:
            self.stability_band = "green"
        elif coherence >= 0.4 and self.tension_metric < 2.5:
            self.stability_band = "yellow"
        else:
            self.stability_band = "red"
