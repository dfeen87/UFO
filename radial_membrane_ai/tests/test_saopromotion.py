"""
Unit and integration tests for Symmetric Ascension Operator (SAO).
"""

from __future__ import annotations
import math
import pytest
from radial_membrane_ai.membrane import RadialMembrane
from radial_membrane_ai.boundary import BoundaryGeometry
from radial_membrane_ai.saopromotion import SAOPromotor


def test_sao_promotor() -> None:
    membrane = RadialMembrane()
    boundary = BoundaryGeometry()
    promotor = SAOPromotor(promotion_threshold=0.3)

    # 1. Align symmetry
    membrane.strings[0].activation = 0.8
    membrane.strings[6].activation = 0.2
    aligned = promotor.align_symmetry(membrane)
    assert math.isclose(aligned[0], 0.5)
    assert math.isclose(aligned[6], 0.5)

    # 2. Promote representation - Target Layer: Holistic Governor
    verdict, p_sao, meta = promotor.promote(membrane, boundary, "Holistic Governor")
    assert verdict in ("ascend", "admit", "constrain", "reproject", "block")

    # Ensure invalid target layer raises ValueError
    with pytest.raises(ValueError):
        promotor.promote(membrane, boundary, "Invalid Layer")

    # Promote on high alignment
    for s in membrane.strings:
        s.activation = 0.8
    verdict, p_sao, meta = promotor.promote(membrane, boundary, "Holistic Governor")
    assert verdict in ("ascend", "constrain", "block")
