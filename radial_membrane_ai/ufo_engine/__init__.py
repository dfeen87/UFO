"""
U.F.O. Governed Simulation Engine Subpackage.

Implements the runtime orchestration layer coordinating:
- Single-Agent Engine (Lyapunov stability bands, governor suppression, V-channels)
- Multi-Agent Engine (coupled V-channels, holistic governor field, mesh coherence, shard quarantine)
- Configurable Cost Taxonomy and Stability Bands
- Full tick cycle integration, persistent residual logging, and audit ledgers
"""

from __future__ import annotations

from radial_membrane_ai.ufo_engine.config import CostWeights, StabilityBandConfig
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine, SingleAgentRunResult
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine, MultiAgentRunResult

__all__ = [
    "CostWeights",
    "StabilityBandConfig",
    "SingleAgentEngine",
    "SingleAgentRunResult",
    "MultiAgentEngine",
    "MultiAgentRunResult",
]
