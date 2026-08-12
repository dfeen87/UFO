# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Multi-Agent Membrane Coupling and Holistic Mesh Governance package.
"""

from __future__ import annotations

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.multi_agent.cross_channels import CrossClusterVChannel, ClusterPolicyEnvelope
from radial_membrane_ai.multi_agent.hierarchical_governance import (
    HierarchicalHolisticGovernor,
    HierarchicalSAOPromotion
)

__all__ = [
    "UFOAgent",
    "UFOCluster",
    "CrossClusterVChannel",
    "ClusterPolicyEnvelope",
    "HierarchicalHolisticGovernor",
    "HierarchicalSAOPromotion"
]
