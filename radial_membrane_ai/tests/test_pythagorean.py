# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Unit and integration tests for the unified math / Pythagorean projection layer (Feeney, 2026).
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.projection import (
    closure_ratio,
    admissibility_test,
    project_to_admissible,
    residual_deformation
)
from radial_membrane_ai.admissibility import (
    angular_decomposition,
    local_closure_test,
    KernelEvolution,
    apply_lyapunov_dissipation,
    phase_smoothing,
    boundary_loop_trace
)
from radial_membrane_ai.coherence import closure_coherence
from radial_membrane_ai.facet import (
    TensionState,
    FacetVector,
    TensionAutomaton,
    route_signal
)
from radial_membrane_ai.holistic import compute_holistic_field
from radial_membrane_ai.simulation import RainbowSimulation


def test_projection_math() -> None:
    """
    Test the fundamental Pythagorean Projection Engine functions.
    """
    # Closure Ratio
    assert closure_ratio(3.0, 4.0, 5.0) == 1.0
    assert closure_ratio(1.0, 1.0, 0.0) == float("inf")

    # Admissibility Test
    assert admissibility_test(1.0) is True
    assert admissibility_test(1.05) is False

    # Euclidean Projection
    # Within bounds -> no change
    a_p, b_p = project_to_admissible(1.0, 1.0, 2.0, metric="euclidean")
    assert a_p == 1.0 and b_p == 1.0

    # Violates bounds -> project to circle of radius 2
    a_p, b_p = project_to_admissible(3.0, 4.0, 2.0, metric="euclidean")
    assert math.isclose(a_p, 1.2)
    assert math.isclose(b_p, 1.6)

    # Edge cases
    assert project_to_admissible(3.0, 4.0, 0.0) == (0.0, 0.0)
    assert project_to_admissible(0.0, 0.0, 2.0) == (0.0, 0.0)

    # Weighted Projection
    weights = {"w_a": 2.0, "w_b": 0.5}
    a_p_w, b_p_w = project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights=weights)
    assert math.isclose(a_p_w**2 + b_p_w**2, 4.0)

    # Weighted Projection with small/zero weights
    weights_small = {"w_a": 0.0, "w_b": 0.0}
    a_p_w_s, b_p_w_s = project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights=weights_small)
    assert math.isclose(a_p_w_s**2 + b_p_w_s**2, 4.0)

    # Fallback / Invalid metric
    a_p_f, b_p_f = project_to_admissible(3.0, 4.0, 2.0, metric="invalid")
    assert math.isclose(a_p_f**2 + b_p_f**2, 4.0)

    # Residual Deformation
    p_a, p_b = residual_deformation(3.0, 4.0, 1.2, 1.6)
    assert p_a == 1.8 and p_b == 2.4


def test_angular_decomposition_and_tests() -> None:
    """
    Test field decomposition, local closure, and stability pipeline.
    """
    membrane = RadialMembrane()
    # Excite string 1 (theta_1 = pi/6)
    membrane.strings[0].activation = 1.0

    a, b = angular_decomposition(membrane, 0.0, samples=64)
    assert abs(a) > 0.0 or abs(b) > 0.0

    # Local closure test
    assert local_closure_test(a, b, 1.5) is True
    assert local_closure_test(10.0, 10.0, 1.0) is False

    # Kernel Leg Evolution
    evo = KernelEvolution()
    initial_K = evo.K
    initial_K_HLV = evo.K_HLV
    evo.step(input_excitation=1.0, v_channel_activity=0.5)
    assert evo.K != initial_K
    assert evo.K_HLV != initial_K_HLV
    assert evo.get_effective_kernel(0.0, 0.5) > 0.0


def test_dissipation_and_smoothing() -> None:
    """
    Test stability dissipation, circular phase smoothing, and loop trace.
    """
    membrane = RadialMembrane()
    for s in membrane.strings:
        s.activation = 0.8

    # Circular phase smoothing
    phase_smoothing(membrane, window_size=3)
    for s in membrane.strings:
        assert math.isclose(s.activation, 0.8)  # uniform remains unchanged

    # Small window size
    phase_smoothing(membrane, window_size=1)
    for s in membrane.strings:
        assert math.isclose(s.activation, 0.8)

    membrane.strings[0].activation = 1.0
    phase_smoothing(membrane, window_size=3)
    assert membrane.strings[0].activation < 1.0  # smoothed down

    # Boundary Loop Trace
    boundary = BoundaryGeometry(base_radius=1.5)
    length = boundary_loop_trace(boundary, samples=64)
    assert math.isclose(length, 2.0 * math.pi * 1.5, rel_tol=1e-3)


def test_closure_coherence_and_facets() -> None:
    """
    Test Q_ij and Facet-state automaton & routing.
    """
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()

    # Closure Coherence
    q_val = closure_coherence(membrane, boundary, 1, 2, samples=16)
    assert 0.0 <= q_val <= 1.0

    # Single sample
    q_val_single = closure_coherence(membrane, boundary, 1, 2, samples=1)
    assert 0.0 <= q_val_single <= 1.0

    # Test indices exception
    with pytest.raises(ValueError):
        closure_coherence(membrane, boundary, 0, 13)

    # Tension Automaton
    auto = TensionAutomaton()
    assert auto.transition(TensionState.RELAXED, 0.1, 0.0) == TensionState.RELAXED
    assert auto.transition(TensionState.RELAXED, 0.5, 0.1) == TensionState.ADMISSIBLE
    assert auto.transition(TensionState.RELAXED, 0.8, 0.1) == TensionState.STRETCHED
    assert auto.transition(TensionState.RELAXED, 1.0, 0.1) == TensionState.CRITICAL
    assert auto.transition(TensionState.RELAXED, 1.3, 0.1) == TensionState.BOUNDARY_COLLAPSE
    assert auto.transition(TensionState.RELAXED, 0.5, 0.6) == TensionState.BOUNDARY_COLLAPSE

    # Route Signal Rules
    f_relaxed = FacetVector("f1", TensionState.RELAXED, 0.5, 1.0, 0.0, 1.0)
    f_admissible = FacetVector("f2", TensionState.ADMISSIBLE, 0.5, 1.0, 0.0, 1.0)
    f_stretched = FacetVector("f3", TensionState.STRETCHED, 0.5, 1.0, 0.0, 1.0)
    f_critical = FacetVector("f4", TensionState.CRITICAL, 0.5, 1.0, 0.0, 1.0)
    f_collapse = FacetVector("f5", TensionState.BOUNDARY_COLLAPSE, 0.5, 1.0, 0.0, 1.0)

    # State routing logic tests
    assert route_signal(f_relaxed, f_collapse, 1.0) == 0.0
    assert route_signal(f_relaxed, f_critical, 1.0) == 0.2
    assert route_signal(f_relaxed, f_stretched, 1.0) == 0.6
    assert route_signal(f_stretched, f_admissible, 1.0) == 1.2
    assert route_signal(f_relaxed, f_admissible, 1.0) == 1.0


def test_holistic_and_simulation_integration() -> None:
    """
    Test holistic governor field and full RainbowSimulation integration.
    """
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()
    facets = [
        FacetVector(s.name, TensionState.ADMISSIBLE, s.activation, 1.0, 0.0, 1.0)
        for s in membrane.strings
    ]
    q_matrix = np.ones((12, 12), dtype=np.float64) * 0.9

    h_val = compute_holistic_field(membrane, boundary, facets, q_matrix)
    assert h_val > 0.0

    # RainbowSimulation full integration run
    sim = RainbowSimulation()
    sim.run_step("planning")
    assert len(sim.holistic_history) == 1
    assert sim.holistic_history[0] > 0.0

    # Cycle sequence
    sim.run(3, ["technical_deep_analysis", "supportive_concise_reply"])
    sim.render_summary()


def test_dissipation_edge_cases() -> None:
    """
    Test Lyapunov dissipation and membrane quadrant edges.
    """
    from radial_membrane_ai.governor import Governor
    membrane = RadialMembrane()
    gov = Governor()
    # High activations
    for s in membrane.strings:
        s.activation = 0.9
    # Run dissipation with None (no history) -> should do nothing
    apply_lyapunov_dissipation(membrane, gov, target_energy=None, dissipation_rate=0.05)
    for s in membrane.strings:
        assert math.isclose(s.activation, 0.9)

    # Set history with low energy to trigger dissipation
    gov.energy_history = [0.1, 0.2]
    # Current energy is ~0.36, which is > target_energy (0.15), so it scales down activations
    apply_lyapunov_dissipation(membrane, gov, target_energy=None, dissipation_rate=0.05)
    for s in membrane.strings:
        assert s.activation < 0.9

    # Membrane quadrant invalid
    with pytest.raises(ValueError):
        membrane.get_quadrant_activation("unknown")
