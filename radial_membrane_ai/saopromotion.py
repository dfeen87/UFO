"""
Symmetric Ascension Operator (SAO) module.

This module implements the layer-promotion operator chain (SAO), promoting representations
from lower layers to higher layers.

SAO Steps:
1. Symmetry alignment
2. Collapse-boundary activation
3. Admissibility test
4. Projection into receiving layer
5. Residual separation (p_SAO)
6. Beta-style verdict
7. Ascension into higher layer

Verdicts: admit, constrain, block, reproject, ascend
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Dict, Any, Tuple, List

if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.boundary import BoundaryGeometry

from radial_membrane_ai.projection import closure_ratio, project_to_admissible, residual_deformation
from radial_membrane_ai.admissibility import angular_projections, dynamic_capacity_boundary


import numpy as np


class SAOPromotor:
    """
    Symmetric Ascension Operator (SAO) executing layer promotions.
    """

    def __init__(self, promotion_threshold: float = 0.5) -> None:
        self.promotion_threshold = promotion_threshold
        # Log of promotions executed
        self.promotion_ledger: List[Dict[str, Any]] = []

    def align_symmetry(self, membrane: RadialMembrane) -> np.ndarray:
        """
        Symmetric alignment step: Projects activation vector to a symmetrical space
        where opposite components (theta and theta + pi) are aligned to a balanced mean.
        """
        aligned = np.zeros(12, dtype=np.float64)
        for i in range(12):
            opp_idx = (i + 6) % 12
            a1 = membrane.strings[i].activation
            a2 = membrane.strings[opp_idx].activation
            mean_val = (a1 + a2) / 2.0
            aligned[i] = mean_val
        return aligned

    def promote(
        self,
        membrane: RadialMembrane,
        boundary: BoundaryGeometry,
        target_layer: str
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Promotes representation from local membrane to receiving layers.

        Receiving Layers:
        - Holistic Governor
        - identity core
        - semantic memory
        - execution layer
        - policy layer
        - agent orchestration

        Verdicts:
        - admit: Admissible for local layer but not high enough for promotion.
        - constrain: Elevated load, restricted promotion.
        - block: Violates constraints heavily, promotion blocked.
        - reproject: Re-projection required to handle residues before promotion.
        - ascend: Successfully promotes representation to receiving layer.
        """
        valid_layers = {
            "Holistic Governor", "identity core", "semantic memory",
            "execution layer", "policy layer", "agent orchestration"
        }
        if target_layer not in valid_layers:
            raise ValueError(f"Invalid target layer: {target_layer}. Must be one of {valid_layers}")

        # 1. Symmetry alignment
        aligned_activations = self.align_symmetry(membrane)

        # 2. Collapse-boundary activation (mean of aligned activations)
        collapse_activation = float(np.mean(aligned_activations))

        # 3. Admissibility test
        # We check the worst case admissibility on the membrane
        max_i = 0.0
        total_p_sao_mag = 0.0

        for i, s in enumerate(membrane.strings):
            # Using aligned activation
            a_orig, b_orig = angular_projections(membrane, s.theta, samples=64)
            # Scale legs by aligned activation ratio
            scale_fac = aligned_activations[i] / (s.activation if s.activation > 1e-9 else 1.0)
            a_aligned = a_orig * scale_fac
            b_aligned = b_orig * scale_fac

            c = dynamic_capacity_boundary(boundary, s.theta)
            i_val = closure_ratio(a_aligned, b_aligned, c)
            if i_val > max_i:
                max_i = i_val

            # 4. Projection into receiving layer
            a_proj, b_proj = project_to_admissible(a_aligned, b_aligned, c, metric="euclidean")

            # 5. Residual separation (p_SAO)
            res_a, res_b = residual_deformation(a_aligned, b_aligned, a_proj, b_proj)
            total_p_sao_mag += math.sqrt(res_a**2 + res_b**2)

        p_sao = total_p_sao_mag / 12.0

        # 6. Beta-style verdict
        beta = 1 if p_sao > 1e-6 else 0

        # Determine verdict based on threshold and residuals
        if p_sao > 0.4 or max_i > 1.5:
            verdict = "block"
        elif p_sao > 0.15:
            verdict = "reproject"
        elif max_i > 1.0:
            verdict = "constrain"
        elif collapse_activation >= self.promotion_threshold:
            verdict = "ascend"
            # Log ascension
            self.promotion_ledger.append({
                "layer": target_layer,
                "activation": collapse_activation,
                "p_sao": p_sao,
                "verdict": "ascend"
            })
        else:
            verdict = "admit"

        # Temporal Promotion Gates Evaluation (Enforced when history is populated, e.g., >= 5 ticks)
        t_state = getattr(membrane, "temporal_state", None)
        temporal_gate_verdict = "ascend"
        if t_state is not None and len(t_state.admissibility_history) >= 5:
            # Short-range gate: >=5 ticks admissible, tension below yellow threshold (e.g. < 2.0)
            # Mid-range gate: >=20 ticks admissible, tension mostly in green (< 1.0)
            # Long-range gate: >=50 ticks admissible, tension predominantly green, no red
            consec = t_state.consecutive_admissible_ticks
            avg_i, avg_t = t_state.compute_exponential_decay_averages(eta=0.7)

            if target_layer == "semantic memory":
                # Requires mid-range gate: N2 = 20 ticks
                if consec >= 20 and avg_t < 1.0:
                    temporal_gate_verdict = "ascend"
                else:
                    temporal_gate_verdict = "constrain"
            elif target_layer in ("identity core", "policy layer", "agent orchestration"):
                # Requires long-range gate: N3 = 50 ticks
                if consec >= 50 and avg_t < 1.0:
                    temporal_gate_verdict = "ascend"
                else:
                    temporal_gate_verdict = "block"
            else:
                # Requires short-range gate: N1 = 5 ticks (e.g. Holistic Governor, execution layer)
                if consec >= 5 and avg_t < 2.0:
                    temporal_gate_verdict = "ascend"
                else:
                    temporal_gate_verdict = "admit"

        # Blend original and temporal verdicts
        if verdict == "ascend" and temporal_gate_verdict != "ascend":
            verdict = temporal_gate_verdict

        return verdict, p_sao, {
            "collapse_activation": collapse_activation,
            "max_closure_ratio": max_i,
            "beta": beta,
            "temporal_gate_verdict": temporal_gate_verdict
        }
