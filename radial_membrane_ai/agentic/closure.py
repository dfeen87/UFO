"""Reason-coded Paper XII Agentic Closure operator."""

from radial_membrane_ai.agentic.contracts import (
    AdmissionDecision,
    AdmissionVerdict,
    AgenticResidual,
    ClosureDecision,
    ClosureResult,
    ExecutionState,
    IntentSatisfaction,
    ReflectionResult,
    ResourceBudget,
    VerificationResult,
    VerificationStatus,
)


class AgenticClosure:
    def evaluate(
        self,
        *,
        admission: AdmissionVerdict,
        execution_state: ExecutionState,
        verification: VerificationResult,
        intent_satisfaction: IntentSatisfaction,
        residual: AgenticResidual,
        remaining_budget: ResourceBudget,
        authority_valid: bool,
        reflection: ReflectionResult,
        hard_budget_breach: bool = False,
    ) -> ClosureResult:
        if admission.decision is AdmissionDecision.ESCALATE:
            return ClosureResult(ClosureDecision.ESCALATE, ("ADMISSION_ESCALATED",))
        if admission.decision is AdmissionDecision.BLOCK:
            return ClosureResult(ClosureDecision.HALT_FAILURE, ("ADMISSION_BLOCKED",))
        if admission.decision is AdmissionDecision.CONSTRAIN:
            return ClosureResult(ClosureDecision.CONSTRAIN, ("ADMISSION_CONSTRAINED",))
        if admission.decision is AdmissionDecision.REPROJECT:
            return ClosureResult(ClosureDecision.REPLAN, ("ADMISSION_REPROJECT_REQUIRED",))
        if execution_state is ExecutionState.OUTCOME_UNKNOWN:
            return ClosureResult(ClosureDecision.ESCALATE, ("EXECUTION_OUTCOME_UNKNOWN",))
        if not authority_valid or residual.policy_authority:
            return ClosureResult(ClosureDecision.CONSTRAIN, ("AUTHORITY_UNRESOLVED",))
        if hard_budget_breach:
            return ClosureResult(ClosureDecision.HALT_FAILURE, ("HARD_BUDGET_BREACH",))
        if reflection.attempted and not reflection.committed:
            return ClosureResult(ClosureDecision.REFLECT, ("REFLECTION_REJECTED",))
        if execution_state is ExecutionState.FAILED or verification.status is VerificationStatus.FAILED:
            return ClosureResult(ClosureDecision.REPLAN, ("ACTION_OR_VERIFICATION_FAILED",))
        if intent_satisfaction.status is VerificationStatus.VERIFIED:
            return ClosureResult(ClosureDecision.HALT_SUCCESS, ("INTENT_POSTCONDITIONS_VERIFIED",))
        if not any(remaining_budget.limits):
            return ClosureResult(ClosureDecision.HALT_FAILURE, ("BUDGET_EXHAUSTED",))
        return ClosureResult(ClosureDecision.CONTINUE, ("FUTURE_ACTION_PERMISSIBLE_NOT_AUTHORIZED",))
