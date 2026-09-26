# Copyright (c) 2026 Don Michael Feeney Jr.
# Standard MIT License applies.

"""
Perception-action reflection engine for Version 3 Agentic AI.
Provides dual-trigger reflection on tool failures and membrane instability thresholds.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from radial_membrane_ai.agentic.tools import ToolCallResult
from radial_membrane_ai.exceptions import ValidationError
from radial_membrane_ai.residuals import ResidualLedger


@dataclass
class ReflectionRecord:
    """Record of a governed reflection event."""

    trigger_type: str  # TOOL_FAILURE, MEMBRANE_INSTABILITY, DUAL_TRIGGER, ROUTINE
    step_description: str
    success: bool
    output: str
    tension: float
    curvature: float
    reflection_text: str
    action_taken: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforces reflection record metrics invariants."""
        if not all(math.isfinite(value) for value in (self.tension, self.curvature, self.timestamp)):
            raise ValidationError("reflection metrics and timestamp must be finite.")
        if self.tension < 0.0:
            raise ValidationError(f"tension cannot be negative, got {self.tension}.")
        if self.curvature < 0.0:
            raise ValidationError(f"curvature cannot be negative, got {self.curvature}.")


class ReflectionEngine:
    """Monitors agent execution and membrane physics to trigger self-correction and logging."""

    def __init__(
        self,
        tension_threshold: float = 2.5,
        curvature_threshold: float = 8.0,
        residual_ledger: Optional[ResidualLedger] = None,
    ) -> None:
        if tension_threshold <= 0.0:
            raise ValidationError(f"tension_threshold must be positive, got {tension_threshold}.")
        if curvature_threshold <= 0.0:
            raise ValidationError(f"curvature_threshold must be positive, got {curvature_threshold}.")

        self.tension_threshold = tension_threshold
        self.curvature_threshold = curvature_threshold
        self.residual_ledger = residual_ledger or ResidualLedger()
        self.reflection_history: List[ReflectionRecord] = []

    def evaluate_and_reflect(
        self,
        step_description: str,
        tool_result: Optional[ToolCallResult],
        current_tension: float,
        current_curvature: float,
    ) -> Optional[ReflectionRecord]:
        """Evaluates tool outcomes and membrane metrics for reflection triggers."""

        if not all(math.isfinite(value) and value >= 0.0 for value in (current_tension, current_curvature)):
            raise ValidationError("current tension and curvature must be finite and non-negative.")
        tool_failed = tool_result is not None and not tool_result.success
        high_instability = (current_tension >= self.tension_threshold) or (current_curvature >= self.curvature_threshold)

        if not tool_failed and not high_instability:
            return None  # No reflection trigger active

        if tool_failed and high_instability:
            trigger_type = "DUAL_TRIGGER"
        elif tool_failed:
            trigger_type = "TOOL_FAILURE"
        else:
            trigger_type = "MEMBRANE_INSTABILITY"

        output_str = tool_result.output if tool_result else "N/A"
        reflection_text = (
            f"Governed reflection triggered [{trigger_type}] during step '{step_description}'. "
            f"Tension={current_tension:.2f} (limit {self.tension_threshold:.2f}), "
            f"Curvature={current_curvature:.2f} (limit {self.curvature_threshold:.2f})."
        )

        if trigger_type in ("TOOL_FAILURE", "DUAL_TRIGGER"):
            action_taken = "Initiate step replanning and substitute safer execution params."
        else:
            action_taken = "Apply governor suppression and contract membrane radius to stabilize tension."

        record = ReflectionRecord(
            trigger_type=trigger_type,
            step_description=step_description,
            success=not tool_failed,
            output=output_str,
            tension=current_tension,
            curvature=current_curvature,
            reflection_text=reflection_text,
            action_taken=action_taken,
            timestamp=float(len(self.reflection_history) + 1),
        )

        self.reflection_history.append(record)

        # Log into residual ledger for persistent preservation
        self.residual_ledger.log_failure(
            record_id=f"REFL_{len(self.reflection_history):08d}",
            error_type=f"Reflection_{trigger_type}",
            shard_id="agentic_reflection",
            severity="MEDIUM" if trigger_type == "TOOL_FAILURE" else "HIGH",
            details={
                "step": step_description,
                "output": output_str,
                "tension": current_tension,
                "curvature": current_curvature,
                "action": action_taken,
            },
        )

        return record

    def get_suggested_activation_deltas(self, record: ReflectionRecord) -> Dict[int, float]:
        """Generates suggested string activation adjustments based on reflection record."""
        deltas: Dict[int, float] = {}
        if record.trigger_type in ("MEMBRANE_INSTABILITY", "DUAL_TRIGGER"):
            # Suppress high-cost generative/exploration strings (strings 4, 5) and boost conciseness (string 11)
            deltas[4] = -0.2  # Exploration
            deltas[5] = -0.2  # Creativity
            deltas[11] = 0.15  # Conciseness
            deltas[1] = 0.1    # Precision
        elif record.trigger_type == "TOOL_FAILURE":
            # Boost Precision (string 1) and Structural Rigor (string 9)
            deltas[1] = 0.2
            deltas[9] = 0.15
        return deltas
