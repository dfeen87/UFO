# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
LDEWorkload integrates the Letter‑Depth Encoding (L.D.E.) subsystem as a workload family.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radial_membrane_ai.workloads.engine import WorkloadTrace

from radial_membrane_ai.workloads.workload import (
    Workload,
    SimulationTarget,
    StabilityBand
)
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.lde.models import LDEConfig


class LDEWorkload(Workload):
    """
    Workload that processes a raw text stream through the L.D.E. pipeline,
    producing an LDEState snapshot wrapped in a WorkloadTrace.
    """

    def __init__(self, config: Optional[LDEConfig] = None) -> None:
        self.config = config or LDEConfig()
        super().__init__(
            name="L.D.E. Text Processing Workload",
            description="Workload for running the L.D.E. pipeline on a raw text stream.",
            target=SimulationTarget.SINGLE_AGENT,
            regime_expectation=KernelRegimeType.BALANCED,
            stability_expectation=StabilityBand.GREEN,
            coherence_expectation=1.0,
            envelope_expectation=PolicyEnvelope(),
            steps=[]
        )

    def run(self, text: str) -> WorkloadTrace:
        """
        Runs the full L.D.E. pipeline on the input raw text,
        returning a WorkloadTrace containing a single WorkloadFrame with the LDEState.
        """
        from radial_membrane_ai.lde.pipeline import lde_encode
        from radial_membrane_ai.workloads.engine import (
            WorkloadTrace,
            WorkloadFrame,
            WorkloadFrameMetrics,
            SAOLevel,
            EnvelopeState,
            StabilityBand as EngineStabilityBand
        )

        state = lde_encode(text, self.config)

        # Compute aggregate metrics
        curvature = float(max(state.boundary.curvature_map)) if state.boundary.curvature_map else 0.0
        tensions = [s.tension for s in state.strings.values() if s.activation > 0]
        tension = float(np.mean(tensions)) if tensions else 0.0
        coherences = [c.coherence for c in state.channels]
        coherence = float(np.mean(coherences)) if coherences else 1.0

        metrics = WorkloadFrameMetrics(
            step_index=0,
            curvature=curvature,
            tension=tension,
            coherence=coherence,
            regime="balanced",
            stability_band=EngineStabilityBand.GREEN,
            sao_level=SAOLevel.NONE,
            envelope_state=EnvelopeState.ADMIT
        )

        frame = WorkloadFrame(
            metrics=metrics,
            membrane_geometry=state.boundary,
            vchannels=state.channels,
            sao_events=[]
        )

        # Store extra L.D.E. attributes for visualization
        setattr(frame, "lde_state", state)

        return WorkloadTrace(frames=[frame])
