"""
Bounded Compute Envelope module.

Implements the Brim outer control architecture including:
- Admissible Geometry G
- Projection II_G
- Residual deformation p(t) and normalized residual p_n(t)
- Binary-state conversion beta
- Operator chain mapping AAG to Av
- Brim energy calculations V_B(t)
- Falsification stack
- Frozen schemas and claim-state ledgers
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Dict, Any, List, Tuple
if TYPE_CHECKING:
    from radial_membrane_ai.membrane import RadialMembrane
    from radial_membrane_ai.boundary import BoundaryGeometry

from radial_membrane_ai.projection import project_to_admissible, residual_deformation
from radial_membrane_ai.admissibility import angular_projections, dynamic_capacity_boundary


class ClaimStateLedger:
    """
    Ledger storing execution claims, validating them, and tracking claim status.
    """

    def __init__(self) -> None:
        self.claims: List[Dict[str, Any]] = []

    def record_claim(self, claim_id: str, assertion: str, status: str, metadata: Dict[str, Any]) -> None:
        """
        Records an assertion/claim with its metadata and state status.
        """
        self.claims.append({
            "claim_id": claim_id,
            "assertion": assertion,
            "status": status,
            "metadata": metadata
        })

    def update_status(self, claim_id: str, status: str) -> None:
        for claim in self.claims:
            if claim["claim_id"] == claim_id:
                claim["status"] = status
                break


class FalsificationStack:
    """
    Falsification stack to register and pop constraint violation events.
    """

    def __init__(self) -> None:
        self.violations: List[Dict[str, Any]] = []

    def push_violation(self, angle: float, val: float, boundary: float, type_violation: str) -> None:
        self.violations.append({
            "angle": angle,
            "value": val,
            "boundary": boundary,
            "type": type_violation
        })

    def pop_violation(self) -> Dict[str, Any] | None:
        if self.violations:
            return self.violations.pop()
        return None

    def is_violated(self) -> bool:
        return len(self.violations) > 0


class BrimEnvelope:
    """
    Brim outer control architecture enforcing Bounded Compute Envelope.
    Collapse activation commands into: admit, constrain, block, rest, re-project.
    """

    def __init__(
        self,
        energy_threshold: float = 1.0,
        falsification_stack: FalsificationStack | None = None,
        claim_ledger: ClaimStateLedger | None = None
    ) -> None:
        self.energy_threshold = energy_threshold
        self.falsification_stack = falsification_stack or FalsificationStack()
        self.claim_ledger = claim_ledger or ClaimStateLedger()
        self.frozen_schemas: Dict[str, Any] = {}

    def freeze_schema(self, schema_id: str, data: Dict[str, Any]) -> None:
        """
        Freezes a dynamic configuration schema for governance consistency.
        """
        self.frozen_schemas[schema_id] = data

    def compute_brim_energy(self, membrane: RadialMembrane, boundary: BoundaryGeometry, samples: int = 32) -> float:
        """
        Computes the brim energy V_B(t) as the integrated residual square deformation.
        """
        total_energy = 0.0
        dtheta = (2.0 * math.pi) / samples
        for k in range(samples):
            theta = k * dtheta
            a, b = angular_projections(membrane, theta, samples=64)
            c = dynamic_capacity_boundary(boundary, theta)
            a_proj, b_proj = project_to_admissible(a, b, c, metric="euclidean")
            p_a, p_b = residual_deformation(a, b, a_proj, b_proj)
            total_energy += (p_a**2 + p_b**2) * dtheta
        return total_energy

    def evaluate_envelope(
        self,
        membrane: RadialMembrane,
        boundary: BoundaryGeometry,
        samples: int = 12
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluates the membrane against the envelope, returning a collapsed status verdict
        and residual metrics dictionary.

        Verdicts:
        - admit: Fully admissible, no violations or residues.
        - constrain: Partial boundary violations but within threshold, scale down activations.
        - block: Heavy boundary violations or energy exceeds critical threshold.
        - rest: Zero activation or negative states.
        - re-project: Significant residues but repairable; triggers a re-projection cycle.
        """
        # Calculate brim energy
        energy = self.compute_brim_energy(membrane, boundary, samples=samples)

        # Calculate max normalized residual and check for violations
        max_res_norm = 0.0
        has_violations = False

        for s in membrane.strings:
            a, b = angular_projections(membrane, s.theta, samples=64)
            c = dynamic_capacity_boundary(boundary, s.theta)
            a_proj, b_proj = project_to_admissible(a, b, c, metric="euclidean")
            p_a, p_b = residual_deformation(a, b, a_proj, b_proj)
            res_mag = math.sqrt(p_a**2 + p_b**2)

            # Normalization factor (avoid division by zero)
            c_norm = max(1e-9, c)
            p_n = res_mag / c_norm
            if p_n > max_res_norm:
                max_res_norm = p_n

            # Check for binary state conversion beta (is residual strictly positive?)
            beta = 1 if res_mag > 1e-6 else 0
            if beta == 1:
                has_violations = True
                self.falsification_stack.push_violation(s.theta, math.sqrt(a**2 + b**2), c, "boundary_violation")
                self.claim_ledger.record_claim(
                    f"claim_{s.name}_{s.index}",
                    f"{s.name} is admissible",
                    "falsified",
                    {"residual": res_mag}
                )
            else:
                self.claim_ledger.record_claim(
                    f"claim_{s.name}_{s.index}",
                    f"{s.name} is admissible",
                    "verified",
                    {"residual": 0.0}
                )

        # AAG -> Av operator chain analysis (Average Activation to Average Velocity style constraint)
        avg_act = sum(s.activation for s in membrane.strings) / 12.0
        if avg_act < 1e-9:
            return "rest", {"brim_energy": energy, "max_normalized_residual": 0.0}

        if energy > self.energy_threshold * 2.0 or max_res_norm > 1.0:
            return "block", {"brim_energy": energy, "max_normalized_residual": max_res_norm}
        elif energy > self.energy_threshold:
            return "re-project", {"brim_energy": energy, "max_normalized_residual": max_res_norm}
        elif has_violations or max_res_norm > 0.1:
            return "constrain", {"brim_energy": energy, "max_normalized_residual": max_res_norm}
        else:
            return "admit", {"brim_energy": energy, "max_normalized_residual": max_res_norm}
