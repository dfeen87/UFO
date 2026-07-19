"""
Unit tests for extended governance layers and kernels.
"""

from __future__ import annotations
import math

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.admissibility import (
    radial_depth_contribution,
    dynamic_capacity_boundary,
    angular_projections,
    local_closure_test_at_angle,
    global_closure_aggregation
)
from radial_membrane_ai.kernels import KernelRegimeManager
from radial_membrane_ai.holistic import HolisticGovernorField


def test_admissibility_additions() -> None:
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()

    # Excite strings
    membrane.strings[0].activation = 0.5
    membrane.strings[0].radius = 1.5

    # Check radial depth contribution
    contrib = radial_depth_contribution(membrane, 0.0)
    assert contrib > 0.0

    # Check dynamic capacity boundary
    cap = dynamic_capacity_boundary(boundary, 0.0)
    assert cap == boundary.get_radius(0.0)

    # Check angular projections
    a, b = angular_projections(membrane, 0.0)
    assert isinstance(a, float)
    assert isinstance(b, float)

    # Check local closure test at angle
    res = local_closure_test_at_angle(membrane, boundary, 0.0)
    assert isinstance(res, bool)

    # Check global closure aggregation
    agg = global_closure_aggregation(membrane, boundary)
    assert "average_ratio" in agg
    assert "max_ratio" in agg
    assert "is_admissible" in agg


def test_kernel_regime_manager() -> None:
    mgr = KernelRegimeManager()

    # Step kernels
    k, k_hlv = mgr.step_kernels(input_excitation=1.0, v_channel_activity=0.5)
    assert k > 0.0
    assert k_hlv > 0.0

    # Compute effective kernel
    k_eff = mgr.compute_effective_kernel(0.0, A_0=1.2, u_t_theta=0.8)
    assert k_eff > 0.0

    # Select regimes
    r1 = mgr.select_regime(0.1, 0.9)
    assert r1 == (1.0, 0.0, 0.0)

    r2 = mgr.select_regime(0.5, 0.5)
    assert r2 == (0.0, 1.0, 0.0)

    r3 = mgr.select_regime(0.8, 0.2)
    assert r3 == (0.0, 0.0, 1.0)

    # Angular phase alignment and coherence ratio
    align = mgr.angular_phase_alignment(0.0, math.pi)
    assert math.isclose(align, 0.0, abs_tol=1e-9)

    ratio = mgr.coherence_ratio([0.5, 0.7, 0.9])
    assert math.isclose(ratio, 0.7)


def test_holistic_governor_field() -> None:
    field = HolisticGovernorField()

    H = field.compute_H_field(p_avg=0.1, o_avg=0.2, c_avg=0.1, T_avg=0.05, K_avg=0.15)
    assert isinstance(H, float)

    score = field.get_coherence_score(H)
    assert 0.0 <= score <= 1.0

    band = field.evaluate_stability_band(score)
    assert band in ("green", "yellow", "red")

    # Drift
    assert field.detect_drift(0.9) is False
    assert field.detect_drift(0.8) is False
    assert field.detect_drift(0.7) is False
    assert field.detect_drift(0.6) is False
    # Drop from 0.9 to 0.4 over 5 steps exceeds 0.15 threshold
    assert field.detect_drift(0.4) is True

    # Overload
    assert field.detect_overload(0.9) is True
    assert field.detect_overload(0.2) is False

    # Policy tension
    assert field.detect_policy_tension(p_avg=0.8, pi_avg=0.8) is True
    assert field.detect_policy_tension(p_avg=0.1, pi_avg=0.2) is False

    # Modulate kernel
    mod = field.modulate_kernel(0.5)
    assert 0.1 <= mod <= 1.0

    # Reconfiguration trigger
    assert field.check_reconfiguration_trigger(0.2, False, False) is True
    assert field.check_reconfiguration_trigger(0.8, True, True) is True
