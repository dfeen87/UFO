"""
Unit and integration tests for the UFO governed deformable radial membrane model.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from radial_membrane_ai.membrane import RadialMembrane, BehavioralString
from radial_membrane_ai.channels import (
    phase_alignment,
    activation_weight,
    base_propagation,
    cost_aware_propagation,
    update_radius_along_channel,
    channel_coherence,
    max_field_activation,
    ChannelDiagnostics
)
from radial_membrane_ai.governor import Governor, GovernorConfig, compute_local_cost, governed_propagation
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.cost import RuntimeCostVector, reduce_avoidable_cost
from radial_membrane_ai.simulation import RainbowSimulation


def test_membrane_initialization() -> None:
    """
    Test that the radial membrane initializes with 12 strings,
    correct default values, correct phase angles, and correct quadrant mappings.
    """
    membrane = RadialMembrane(c_baseline=0.15)
    assert len(membrane.strings) == 12

    # Check first string: "depth"
    s1 = membrane.strings[0]
    assert s1.name == "depth"
    assert s1.index == 1
    assert math.isclose(s1.theta, (2.0 * math.pi * 1) / 12.0)
    assert s1.activation == 0.0
    assert s1.radius == 0.0
    assert s1.cost == 0.15
    assert s1.tension == 0.0
    assert s1.stiffness == 0.0

    # Check last string: "conciseness"
    s12 = membrane.strings[11]
    assert s12.name == "conciseness"
    assert s12.index == 12
    assert math.isclose(s12.theta, 2.0 * math.pi)

    # Check quadrant mapping activations
    assert membrane.get_quadrant_activation("analytical") == 0.0
    assert membrane.get_quadrant_activation("contextual") == 0.0
    assert membrane.get_quadrant_activation("generative") == 0.0
    assert membrane.get_quadrant_activation("interpersonal") == 0.0

    with pytest.raises(ValueError):
        membrane.get_quadrant_activation("invalid_quadrant")


def test_membrane_activation_updates_and_field() -> None:
    """
    Test activation update mechanics and basis field values.
    """
    membrane = RadialMembrane()
    delta = np.zeros(12, dtype=np.float64)
    # Excite depth (idx 1) and precision (idx 2)
    delta[0] = 0.5
    delta[1] = 0.8
    membrane.update_activation(delta)

    assert membrane.strings[0].activation == 0.5
    assert membrane.strings[1].activation == 0.8

    # Assert clamping: activation cannot exceed 1.0 or go below 0.0
    delta_overflow = np.zeros(12, dtype=np.float64)
    delta_overflow[0] = 1.0
    delta_overflow[1] = -2.0
    membrane.update_activation(delta_overflow)
    assert membrane.strings[0].activation == 1.0
    assert membrane.strings[1].activation == 0.0

    with pytest.raises(ValueError):
        membrane.update_activation(np.zeros(10))

    # Test field value calculations using default Gaussian basis
    # Peak at theta_1 (index 1) should be relatively high
    theta_1 = (2.0 * math.pi * 1) / 12.0
    val_at_theta_1 = membrane.field_value(theta_1, basis_type="gaussian")
    assert val_at_theta_1 > 0.5

    # Test field value with Cosine basis
    val_at_theta_1_cos = membrane.field_value(theta_1, basis_type="cosine")
    assert val_at_theta_1_cos > 0.5

    with pytest.raises(ValueError):
        membrane.field_value(theta_1, basis_type="unknown")


def test_channels_propagation_and_coherence() -> None:
    """
    Test V-channel propagation math and coherence diagnostics.
    """
    membrane = RadialMembrane()
    # Excite string 1 (depth) to 0.5 and string 2 (precision) to 0.5
    membrane.strings[0].activation = 0.5
    membrane.strings[1].activation = 0.5

    # Test basic phase alignment
    # Identical angles -> alignment 1.0
    assert math.isclose(phase_alignment(0.5, 0.5), 1.0)
    # Opposite angles -> alignment 0.0
    assert math.isclose(phase_alignment(0.0, math.pi), 0.0)

    # Test activation sigmoid weight
    # 0.5 activation weight
    w_05 = activation_weight(0.5)
    assert 0.0 < w_05 < 1.0

    # Test base propagation
    p_base = base_propagation(membrane.strings[0], membrane.strings[1])
    assert p_base > 0.0

    # Test cost-aware propagation
    membrane.strings[1].cost = 0.5
    p_cost_simple = cost_aware_propagation(
        membrane.strings[0], membrane.strings[1], lambda_=1.0, cost_variant="simple"
    )
    p_cost_exp = cost_aware_propagation(
        membrane.strings[0], membrane.strings[1], lambda_=1.0, cost_variant="exponential"
    )
    assert p_cost_simple < p_base
    assert p_cost_exp < p_base

    with pytest.raises(ValueError):
        cost_aware_propagation(membrane.strings[0], membrane.strings[1], cost_variant="unknown")

    # Test bounded depth update
    # Source radius is 1.5
    membrane.strings[0].radius = 1.5
    new_r = update_radius_along_channel(
        source=membrane.strings[0], target=membrane.strings[1], r_max=2.0
    )
    assert new_r <= 2.0
    assert new_r > 0.0

    # Custom h_func
    new_r_custom = update_radius_along_channel(
        source=membrane.strings[0], target=membrane.strings[1], r_max=2.0, h_func=lambda p: p * 0.5
    )
    assert new_r_custom <= new_r

    # Test channel coherence calculation
    coher = channel_coherence(membrane, 1, 2, samples=10)
    assert 0.0 <= coher <= 1.0

    coher_1sample = channel_coherence(membrane, 1, 2, samples=1)
    assert 0.0 <= coher_1sample <= 1.0

    with pytest.raises(ValueError):
        channel_coherence(membrane, 0, 2)

    # Test max field activation helper
    max_act = max_field_activation(membrane, samples=30)
    assert max_act > 0.0

    # Test Channel Diagnostics helper
    diag = ChannelDiagnostics(membrane)
    matrix = diag.compute_all_coherences(samples=8)
    assert matrix.shape == (12, 12)
    assert np.all(matrix >= 0.0)

    hi_channels = diag.get_highest_coherence_channels()
    assert len(hi_channels) >= 0


def test_governor_costs_and_stability() -> None:
    """
    Test governor local cost computations, updates, and Lyapunov stability tracking.
    """
    config = GovernorConfig(w_d=0.3, w_a=0.2, w_l=0.4, w_c=0.1, w_T=0.1, w_K=0.1)
    gov = Governor(config=config)

    s = BehavioralString(
        name="depth", index=1, theta=0.0, activation=0.5,
        radius=1.0, cost=0.1, tension=0.2, stiffness=0.1
    )

    # Local cost calculation
    cost_i = compute_local_cost(s, tool_load=0.5, context_load=10.0, config=config)
    # Expected: 0.3*(1^2) + 0.2*0.5 + 0.4*0.5 + 0.1*log(11) + 0.1*0.2 + 0.1*0.1
    expected_cost = 0.3 * 1.0 + 0.2 * 0.5 + 0.4 * 0.5 + 0.1 * math.log(11.0) + 0.1 * 0.2 + 0.1 * 0.1
    assert math.isclose(cost_i, expected_cost)

    # Governed propagation
    s_target = BehavioralString(
        name="precision", index=2, theta=math.pi / 6, activation=0.4,
        radius=0.5, cost=0.2, tension=0.0, stiffness=0.0
    )
    p_gov = governed_propagation(s, s_target, config=config, tool_load=0.0, context_load=0.0)
    assert p_gov > 0.0

    p_gov_simple = governed_propagation(s, s_target, config=config, cost_variant="simple")
    assert p_gov_simple > 0.0

    with pytest.raises(ValueError):
        governed_propagation(s, s_target, config=config, cost_variant="invalid")

    # Governor membrane update and tracking Lyapunov energy
    membrane = RadialMembrane()
    # Initially energy depends on baseline costs: 12 strings * beta * c_baseline
    e_init = gov.compute_lyapunov_energy(membrane)
    expected_init_energy = 12.0 * gov.config.lyapunov_beta * membrane.c_baseline
    assert math.isclose(e_init, expected_init_energy)

    gov.update_membrane(membrane, task_value=0.8)
    assert len(gov.energy_history) == 1
    assert gov.is_stable()

    # Create artificial energy history
    gov.energy_history = [10.0, 8.0, 6.0, 5.0, 4.0]
    assert gov.is_stable()  # trending downwards

    gov.energy_history = [1.0, 1.0001, 0.9999, 1.0, 1.0001]
    assert gov.is_stable()  # bounded/stable (low variance)

    gov.energy_history = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert not gov.is_stable()  # growing and unstable


def test_boundary_geometry() -> None:
    """
    Test dynamic boundary radius, updates, and geometric diagnostics.
    """
    boundary = BoundaryGeometry(base_radius=1.5)
    assert boundary.get_radius(0.0) == 1.5

    # Check curvature of uniform circle is 0 (first/second derivative is zero)
    assert math.isclose(boundary.curvature(0.0), 0.0, abs_tol=1e-5)
    assert math.isclose(boundary.asymmetry(0.0), 0.0, abs_tol=1e-5)
    assert math.isclose(boundary.tangent(0.0), 0.0, abs_tol=1e-5)

    # Let's excite a deviation at string index 1 (theta = 2*pi/12)
    boundary.radius_deviation[1] = 0.5
    # Radius at theta = 2*pi/12 should be larger than 1.5
    r_peak = boundary.get_radius((2.0 * math.pi) / 12.0)
    assert r_peak > 1.5

    # Curvature at the peak should be negative (bending downwards)
    curv = boundary.curvature((2.0 * math.pi) / 12.0)
    assert curv < 0.0

    # Asymmetry across from the peak should be positive
    asym = boundary.asymmetry((2.0 * math.pi) / 12.0)
    assert asym > 0.0

    # Test update boundary
    membrane = RadialMembrane()
    # Excite string 1
    membrane.strings[0].activation = 0.8
    membrane.strings[0].radius = 1.0
    boundary.update_boundary(membrane, task_value=0.9)
    # String index 1 deviation should have increased due to stretch pressure
    assert boundary.radius_deviation[1] > 0.5


def test_cost_taxonomy() -> None:
    """
    Test runtime cost vector projections and quality-preserving reductions.
    """
    v = RuntimeCostVector(
        tokens=100.0, depth=2.0, context=1000.0, retrievals=5.0,
        tool_calls=2.0, latency=1.5, corrections=4.0, recovery=0.0
    )

    weights = {
        "tokens": 0.1, "depth": 0.5, "context": 0.01, "retrievals": 0.3,
        "tool_calls": 0.4, "latency": 1.0, "corrections": 0.8, "recovery": 1.2
    }

    # Under high quality (quality_signal = 1.0), productive costs are discounted
    cost_high_q = v.weighted_cost(weights, quality_signal=1.0)
    # Under low quality (quality_signal = 0.0), no discounts and penalties on corrections
    cost_low_q = v.weighted_cost(weights, quality_signal=0.0)

    assert cost_high_q < cost_low_q

    # Test reduce_avoidable_cost function
    v_reduced_low_q = reduce_avoidable_cost(v, quality_signal=0.0, min_retention=0.2)
    v_reduced_high_q = reduce_avoidable_cost(v, quality_signal=1.0, min_retention=0.2)

    assert v_reduced_low_q.tokens < v_reduced_high_q.tokens
    assert v_reduced_low_q.corrections < v_reduced_high_q.corrections


def test_rainbow_simulation() -> None:
    """
    Test Project Rainbow-style multi-step simulation.
    """
    sim = RainbowSimulation()
    assert len(sim.activation_history) == 1
    assert len(sim.radius_history) == 1

    # Run single step of planning
    sim.run_step("planning")
    assert len(sim.activation_history) == 2
    assert len(sim.radius_history) == 2
    assert len(sim.cost_history) == 1
    assert len(sim.observable_cost_history) == 1

    # Run multi-step sequence
    sim.run(n_steps=5, task_sequence=["technical_deep_analysis", "supportive_concise_reply"])
    assert len(sim.activation_history) == 7

    # Verify history is tracked correctly
    acts = sim.get_activation_history()
    assert len(acts) == 7
    assert acts[0].shape == (12,)

    energies = sim.get_energy_history()
    assert len(energies) == 6

    snaps = sim.get_boundary_snapshots()
    assert len(snaps) == 7

    with pytest.raises(ValueError):
        sim.run_step("unknown_task")

    with pytest.raises(ValueError):
        sim.run(2, [])

    # Render summary to test diagnostic output printing
    sim.render_summary()
