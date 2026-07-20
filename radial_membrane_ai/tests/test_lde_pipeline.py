"""
Unit tests for the L.D.E. Pipeline (Algorithm 1).
"""

import pytest
from radial_membrane_ai.lde.models import LDEConfig, LDEState
from radial_membrane_ai.lde.pipeline import lde_encode


def test_basic_pipeline_encoding():
    text = "Read from left to right, the U.S. flag becomes a story — beginnings, grounding, and the horizon ahead."
    config = LDEConfig()
    state = lde_encode(text, config)

    assert isinstance(state, LDEState)
    assert len(state.strings) == 26
    assert len(state.boundary.radius_map) == 100
    assert len(state.boundary.tangent_map) == 100
    assert len(state.boundary.curvature_map) == 100
    assert state.boundary.asymmetry >= 0.0

    # Verify specific properties of letters
    e_string = state.strings["e"]
    assert e_string.activation > 0.0
    assert len(e_string.positions) == 8  # 8 occurrences in the text
    assert e_string.depth > 0.0

    # Verify a letter that does not exist in the text (e.g., 'q')
    q_string = state.strings["q"]
    assert q_string.activation == 0.0
    assert q_string.spread == 0.0
    assert q_string.depth == 0.0
    assert q_string.tension == 0.0
    assert q_string.stiffness == 1.0


def test_edge_cases_and_gaps():
    # 1 occurrence of 'x'
    text_1 = "x"
    state_1 = lde_encode(text_1)
    assert state_1.strings["x"].activation == 1.0
    assert state_1.strings["x"].spread == 0.0
    assert state_1.strings["x"].tension == 0.0

    # 2 occurrences of 'x' (1 gap)
    text_2 = "xax"
    state_2 = lde_encode(text_2)
    assert state_2.strings["x"].activation > 0.0
    assert state_2.strings["x"].spread > 0.0
    assert state_2.strings["x"].tension == 0.5  # Since gaps < 2, tension = repetition_pressure = 1/2 = 0.5

    # 3 occurrences of 'x' (2 gaps)
    text_3 = "xaxbx"
    state_3 = lde_encode(text_3)
    assert state_3.strings["x"].tension >= 0.0


def test_empty_and_punctuation_only_text():
    # Empty string
    state_empty = lde_encode("")
    assert state_empty.strings["a"].activation == 0.0
    assert state_empty.boundary.asymmetry == 0.0

    # Punctuation-only string
    state_punc = lde_encode("!!! --- ???")
    assert state_punc.strings["a"].activation == 0.0
    assert state_punc.boundary.asymmetry == 0.0


def test_reconstruction_failure_verification():
    # Kelvin sign ('K') has Kelvin as uppercase and 'k' as lowercase,
    # but 'k'.upper() is 'K'. So the reconstructed string will be 'K' which mismatch 'K'.
    kelvin_text = "\u212a"
    config = LDEConfig(rho="full-reconstruction")

    with pytest.raises(ValueError, match="Reconstruction verification failed!"):
        lde_encode(kelvin_text, config)

    # If rho is compressed, it should not verify and should pass successfully
    config_compressed = LDEConfig(rho="compressed")
    state = lde_encode(kelvin_text, config_compressed)
    assert isinstance(state, LDEState)
