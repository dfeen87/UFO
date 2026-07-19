"""
Configuration, presets, and weights for the U.F.O. Simulation Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Literal


@dataclass(frozen=True)
class CostWeights:
    """
    Weighting configuration for the 8-dimensional runtime cost vector. Exposes presets.
    """
    tokens: float = 0.05
    depth: float = 0.4
    context: float = 0.01
    retrievals: float = 0.2
    tool_calls: float = 0.3
    latency: float = 0.5
    corrections: float = 0.6
    recovery: float = 0.8

    @classmethod
    def get_preset(cls, preset_name: Literal["strict", "balanced", "exploratory"]) -> CostWeights:
        """
        Returns a CostWeights instance with pre-configured weights based on the preset.
        """
        if preset_name == "strict":
            return cls(
                tokens=0.2,       # higher penalty on verbosity
                depth=0.5,        # high penalty on reasoning depth
                context=0.05,     # slightly higher penalty on context usage
                retrievals=0.4,   # high penalty on retrievals
                tool_calls=0.5,   # high penalty on agentic tools
                latency=0.8,      # very high penalty on latency
                corrections=0.9,  # extremely high penalty on correction overhead
                recovery=1.0      # highest penalty on instability recovery
            )
        elif preset_name == "exploratory":
            return cls(
                tokens=0.01,      # very low penalty on verbosity/expansion
                depth=0.1,        # low penalty on deep reasoning
                context=0.005,    # low context penalty
                retrievals=0.05,  # low retrieval penalty
                tool_calls=0.1,   # low tool use penalty
                latency=0.2,      # low latency penalty
                corrections=0.3,  # softer penalty on corrections
                recovery=0.5      # softer penalty on recovery
            )
        else:  # "balanced"
            return cls()

    def to_dict(self) -> Dict[str, float]:
        return {
            "tokens": self.tokens,
            "depth": self.depth,
            "context": self.context,
            "retrievals": self.retrievals,
            "tool_calls": self.tool_calls,
            "latency": self.latency,
            "corrections": self.corrections,
            "recovery": self.recovery
        }


@dataclass
class StabilityBandConfig:
    """
    Configuration for stability bands and thresholds for Single and Multi Agent Engines.
    """
    # Single-Agent thresholds based on Lyapunov-style energy V(t)
    v_green: float = 2.0
    v_red: float = 5.0

    # Multi-Agent thresholds based on mesh coherence C_mesh(t)
    c_green: float = 0.7
    c_red: float = 0.4
