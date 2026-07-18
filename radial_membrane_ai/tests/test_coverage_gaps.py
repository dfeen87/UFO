"""
Extra test module to ensure 100% test coverage across all lines.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.projection import project_to_admissible
from radial_membrane_ai.governor import Governor


def test_final_coverage_gaps() -> None:
    """
    Cover the remaining missing lines from the coverage report.
    """
    # 1. boundary.py: update_boundary clamps, i_ratio > 1.0, asymmetry, tangent
    membrane = RadialMembrane()
    membrane.strings[0].activation = 1.0
    membrane.strings[0].radius = 1.0
    boundary = BoundaryGeometry()

    # Trigger i_ratio > 1.0 in update_boundary (set all deviations to make capacity small)
    boundary.radius_deviation = {i: -0.9 for i in range(1, 13)}
    boundary.update_boundary(membrane, task_value=0.9)

    # Huge stretch to hit max_dev limit
    boundary.update_boundary(membrane, task_value=1000.0, learning_rate=1.0)
    # Huge collapse to hit min_dev limit
    membrane.strings[0].cost = 1000.0
    boundary.update_boundary(membrane, task_value=0.0, learning_rate=1.0)

    # get_radius total_weight <= 0.0
    boundary.basis_sigma = 1e-12
    boundary.get_radius(0.5)
    boundary.basis_sigma = math.pi / 6

    # asymmetry and tangent
    assert boundary.asymmetry(1.0) == (boundary.get_radius(1.0) - boundary.get_radius(1.0 + math.pi))
    assert boundary.tangent(1.0, delta=0.01) == ((boundary.get_radius(1.01) - boundary.get_radius(0.99)) / 0.02)

    # 2. governor.py: extra clamp logic, is_stable cases
    gov = Governor()
    # Trigger c_i > task_value and task_value < 0.3 clamp
    membrane_gov = RadialMembrane()
    for s in membrane_gov.strings:
        s.radius = 10.0
    gov.update_membrane(membrane_gov, task_value=0.1)

    # is_stable when energy history is small
    gov.energy_history = [1.0]
    assert gov.is_stable() is True
    gov.energy_history = [1.0, 2.0]
    assert gov.is_stable(window=1) is True

    # is_stable trending upward (returns False)
    gov.energy_history = [1.0, 2.0, 3.0]
    assert gov.is_stable() is False

    # 3. membrane.py: quadrant average with empty target strings
    membrane_empty = RadialMembrane()
    membrane_empty._quadrant_map["analytical"] = {"non_existent"}
    assert membrane_empty.get_quadrant_activation("analytical") == 0.0

    # 4. projection.py: weighted with None weights and edge cases
    assert project_to_admissible(0.0, 0.0, 1e-10, metric="euclidean") == (0.0, 0.0)

    # Weighted Projection with None weights
    a_p, b_p = project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights=None)
    assert math.isclose(a_p**2 + b_p**2, 4.0)

    # Bisection search precision
    assert len(project_to_admissible(3.0, 4.0, 2.0, metric="weighted", weights={"w_a": 1.0, "w_b": 1.0})) == 2
