# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Multidimensional Runtime Cost Vector and Projection module.

This module implements the multidimensional cost taxonomy and quality-preserving
reduction described in Section 7 of Feeney (2025).
"""

from __future__ import annotations
from dataclasses import dataclass
import math

from radial_membrane_ai.exceptions import ValidationError


@dataclass(frozen=True)
class RuntimeCostVector:
    """
    Multidimensional Runtime Cost Vector.

    Ref: Section 7.1 of Feeney (2025).
    Tracks resource utilization and overhead metrics across 8 dimensions.

    Attributes:
        tokens: Number of generated tokens / verbosity.
        depth: Reasoning depth/radius costs.
        context: Context window size / token count.
        retrievals: Number of database/KB retrievals.
        tool_calls: Number of agentic tool calls.
        latency: Time taken to respond (seconds or scalar).
        corrections: Count/weight of correction overhead (errors, retries).
        recovery: Instability recovery costs (re-routing, state resets).
    """
    tokens: float
    depth: float
    context: float
    retrievals: float
    tool_calls: float
    latency: float
    corrections: float
    recovery: float

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValidationError("runtime cost dimensions must be finite numeric values.")
        if any(value < 0.0 for value in values):
            raise ValidationError("runtime cost dimensions cannot be negative.")

    def weighted_cost(self, weights: dict[str, float], quality_signal: float = 0.0) -> float:
        """
        Projects the cost vector into a single scalar observable cost.

        Preserves a quality floor: cost components that improve correctness, safety,
        coherence, or user value (e.g., depth, retrievals, tool_calls) are discounted
        by a factor related to the quality_signal, so that high-quality useful work
        is not penalized.

        Args:
            weights: Dictionary mapping string field names to float weights.
            quality_signal: Float representing the quality/correctness of response in [0, 1].

        Returns:
            The projected scalar cost.
        """
        if not math.isfinite(quality_signal) or not 0.0 <= quality_signal <= 1.0:
            raise ValidationError("quality_signal must be finite and in [0, 1].")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValidationError("cost weights must be finite and non-negative.")
        # Define fields that are "quality-improving" (productive/useful) vs "overhead" (waste/inefficiency)
        productive_fields = {"depth", "retrievals", "tool_calls", "context"}

        total_cost = 0.0
        for field_name in [
            "tokens", "depth", "context", "retrievals",
            "tool_calls", "latency", "corrections", "recovery"
        ]:
            val = getattr(self, field_name)
            w = weights.get(field_name, 1.0)

            if field_name in productive_fields:
                # Discount productive cost when quality_signal is high
                discount = 1.0 / (1.0 + quality_signal)
                total_cost += w * val * discount
            else:
                # Overhead is not discounted (or could be penalized more if quality is low)
                # Let's say if quality is low, correction/recovery is highly penalized
                if field_name in {"corrections", "recovery"}:
                    penalty = 1.0 + (1.0 - quality_signal)
                    total_cost += w * val * penalty
                else:
                    total_cost += w * val

        return total_cost


def reduce_avoidable_cost(
    cost_vector: RuntimeCostVector,
    quality_signal: float,
    min_retention: float = 0.2
) -> RuntimeCostVector:
    """
    Reduces/suppresses avoidable costs in the cost vector based on the quality signal.

    Ref: Section 7.2 of Feeney (2025).
    - A high quality_signal (close to 1.0) means the cost is justified, allowing more resources.
    - A low quality_signal (close to 0.0) triggers suppression of:
        * Verbosity (tokens)
        * Redundant context (context)
        * Incoherent routing (retrievals, tool_calls)
        * Repeated correction (corrections, recovery)
        * Low-value expansion (depth, latency)

    This is modeled continuously where each field is multiplied by a retention factor
    interpolated between `min_retention` and 1.0 based on `quality_signal`.

    Args:
        cost_vector: The original RuntimeCostVector.
        quality_signal: Quality signal value in [0, 1].
        min_retention: Minimum retention fraction when quality_signal is 0.0.

    Returns:
        A new RuntimeCostVector with suppressed avoidable costs.
    """
    if not math.isfinite(quality_signal):
        raise ValidationError("quality_signal must be finite.")
    if not math.isfinite(min_retention) or not 0.0 <= min_retention <= 1.0:
        raise ValidationError("min_retention must be finite and in [0, 1].")
    # Clamp quality_signal to [0.0, 1.0]
    q = max(0.0, min(1.0, quality_signal))

    # Compute retention factor: when q=1.0, retention is 1.0. When q=0.0, retention is min_retention.
    retention = min_retention + (1.0 - min_retention) * q

    # Some fields can have stronger/stricter suppression.
    # For example, corrections/recovery can be heavily suppressed when quality is low.
    # Verbosity (tokens) can also be reduced.
    # Let's compute customized multipliers:
    mult_tokens = retention
    mult_depth = retention
    mult_context = min_retention + (1.0 - min_retention) * (q ** 0.5)  # slightly softer suppression
    mult_retrievals = retention
    mult_tool_calls = retention
    mult_latency = retention
    mult_corrections = q  # very strict: if quality is 0, corrections cost is wiped out (suppressed entirely)
    mult_recovery = q  # very strict

    return RuntimeCostVector(
        tokens=cost_vector.tokens * mult_tokens,
        depth=cost_vector.depth * mult_depth,
        context=cost_vector.context * mult_context,
        retrievals=cost_vector.retrievals * mult_retrievals,
        tool_calls=cost_vector.tool_calls * mult_tool_calls,
        latency=cost_vector.latency * mult_latency,
        corrections=cost_vector.corrections * mult_corrections,
        recovery=cost_vector.recovery * mult_recovery
    )
