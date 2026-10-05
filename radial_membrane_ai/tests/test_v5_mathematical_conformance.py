"""Focused executable boundaries for the v5 mathematical conformance audit."""

import math

import pytest

from radial_membrane_ai.admissibility import KernelEvolution, boundary_loop_trace
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.channels import inverse_cost_weight, phase_alignment
from radial_membrane_ai.kernels import KernelRegimeManager
from radial_membrane_ai.projection import closure_ratio, project_to_admissible


def test_local_closure_is_the_published_squared_leg_ratio() -> None:
    assert closure_ratio(3.0, 4.0, 5.0) == pytest.approx(1.0)
    assert closure_ratio(3.0, 4.0, 10.0) == pytest.approx(0.25)


@pytest.mark.parametrize("metric", ["euclidean", "angular", "radial"])
def test_documented_metric_labels_are_intentional_radial_scaling_aliases(metric: str) -> None:
    projected = project_to_admissible(6.0, 8.0, 5.0, metric=metric)
    assert projected == pytest.approx((3.0, 4.0))
    assert math.hypot(*projected) == pytest.approx(5.0)


def test_phase_alignment_and_inverse_cost_keep_their_separate_domains() -> None:
    assert phase_alignment(0.0, 0.0) == pytest.approx(1.0)
    assert phase_alignment(0.0, math.pi) == pytest.approx(0.0)
    assert inverse_cost_weight(0.0, variant="simple") > inverse_cost_weight(1.0, variant="simple")
    assert inverse_cost_weight(1.0, variant="simple") > inverse_cost_weight(10.0, variant="simple")


def test_effective_kernel_surfaces_preserve_distinct_declared_equations() -> None:
    paper_kernel = KernelRegimeManager(initial_K_HLV=2.0)
    operational_kernel = KernelEvolution(initial_K=3.0, initial_K_HLV=2.0)

    assert paper_kernel.compute_effective_kernel(0.7, A_0=0.5, u_t_theta=0.25) == pytest.approx(0.25)
    assert operational_kernel.get_effective_kernel(0.7, cost_pressure=1.0) == pytest.approx(3.0)


def test_boundary_loop_trace_is_snapshot_arc_length_not_temporal_memory() -> None:
    boundary = BoundaryGeometry(base_radius=2.0)
    before = tuple(boundary.radius_deviation.items())
    assert boundary_loop_trace(boundary, samples=512) == pytest.approx(4.0 * math.pi, rel=1e-4)
    assert tuple(boundary.radius_deviation.items()) == before
