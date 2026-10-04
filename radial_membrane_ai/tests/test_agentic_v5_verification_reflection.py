import pytest

from radial_membrane_ai.agentic.contracts import (
    ActionProposal, ExpectedPostcondition, Observation, SideEffectClass, VerificationStatus,
)
from radial_membrane_ai.agentic.engine import AgenticEngine
from radial_membrane_ai.agentic.verification import VerificationRegistry


def proposal(postconditions):
    return ActionProposal(
        "i", "r", "a", "SearchTool", {"query": "x"}, tuple(postconditions), (0.1,) * 8,
        SideEffectClass.OBSERVATIONAL, frozenset({"search"}), "test",
    )


def test_verification_states_do_not_follow_tool_success():
    registry = VerificationRegistry()
    verified = proposal((ExpectedPostcondition("fields_equal", {"x": 1}),))
    obs = Observation("o", "e", {"data": {"x": 1}}, True)
    assert registry.verify(verified, obs).status is VerificationStatus.VERIFIED
    failed_obs = Observation("o2", "e", {"data": {"x": 2}}, True)
    assert registry.verify(verified, failed_obs).status is VerificationStatus.FAILED
    partial = proposal((ExpectedPostcondition("fields_equal", {"x": 1}),
                        ExpectedPostcondition("fields_equal", {"missing": 2})))
    assert registry.verify(partial, obs).status is VerificationStatus.PARTIAL
    assert registry.verify(verified, None).status is VerificationStatus.UNKNOWN


def test_late_invalid_reflection_candidate_never_mutates_authoritative_state():
    engine = AgenticEngine()
    before = tuple(string.activation for string in engine.engine.membrane.strings)
    with pytest.raises(Exception):
        engine.reflection_engine.build_candidate("p", list(before), {1: 0.2, 11: float("nan")}, "adversarial")
    assert tuple(string.activation for string in engine.engine.membrane.strings) == before
