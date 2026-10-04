"""Independent deterministic postcondition verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from radial_membrane_ai.agentic.contracts import (
    ActionProposal, Observation, VerificationResult, VerificationStatus,
)


Verifier = Callable[[dict[str, Any], dict[str, Any]], bool | None]


class VerificationRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {
            "fields_equal": self._fields_equal,
            "output_equals": self._output_equals,
        }

    def register(self, name: str, verifier: Verifier) -> None:
        if not isinstance(name, str) or not name or not callable(verifier):
            raise ValueError("verifier registration requires a name and callable.")
        self._verifiers[name] = verifier

    @staticmethod
    def _fields_equal(expected: dict[str, Any], evidence: dict[str, Any]) -> bool | None:
        source = evidence.get("data")
        if not isinstance(source, Mapping):
            return None
        if any(key not in source for key in expected):
            return None
        return all(source[key] == value for key, value in expected.items())

    @staticmethod
    def _output_equals(expected: dict[str, Any], evidence: dict[str, Any]) -> bool | None:
        if "value" not in expected or "output" not in evidence:
            return None
        return evidence["output"] == expected["value"]

    def verify(self, proposal: ActionProposal, observation: Observation | None) -> VerificationResult:
        if observation is None:
            return VerificationResult(
                VerificationStatus.UNKNOWN, proposal.proposal_digest, None, 0, 0, ("NO_OBSERVATION",)
            )
        if not observation.complete:
            return VerificationResult(
                VerificationStatus.UNKNOWN, proposal.proposal_digest, observation.observation_id,
                len(proposal.expected_postconditions), 0, ("INCOMPLETE_OBSERVATION",),
            )
        outcomes: list[bool | None] = []
        evidence = dict(observation.evidence)
        for condition in proposal.expected_postconditions:
            verifier = self._verifiers.get(condition.verifier)
            outcomes.append(None if verifier is None else verifier(dict(condition.expected), evidence))
        satisfied = sum(outcome is True for outcome in outcomes)
        if any(outcome is False for outcome in outcomes):
            status, reason = VerificationStatus.FAILED, "CONTRADICTORY_EVIDENCE"
        elif outcomes and all(outcome is True for outcome in outcomes):
            status, reason = VerificationStatus.VERIFIED, "ALL_POSTCONDITIONS_SATISFIED"
        elif satisfied:
            status, reason = VerificationStatus.PARTIAL, "PARTIAL_POSTCONDITION_EVIDENCE"
        else:
            status, reason = VerificationStatus.UNKNOWN, "INSUFFICIENT_EVIDENCE"
        return VerificationResult(status, proposal.proposal_digest, observation.observation_id,
                                  len(outcomes), satisfied, (reason,))
