# Retry-unsafe known failures: deferred remediation

The current hardening branch still permits a known failed, non-idempotent tool
to be invoked again after an observational recovery step, despite trusted
`ToolCapability(..., idempotent=False, retry_safe=False)` metadata. This differs
from an unknown outcome: the repaired unknown path escalates without recurrence.

## Confirmed bounded reproduction

From the installed checkout, run the following with its Python interpreter.
It uses a simulated counter, no external service, and at most three cycles.
The final assertion **fails** on the current branch: `tool.calls` is **2**.

```python
from dataclasses import replace
from radial_membrane_ai.agentic.contracts import ResourceBudget, SideEffectClass
from radial_membrane_ai.agentic.tools import ToolCallResult
from radial_membrane_ai.tests.test_bedrock_1_1_hardening import scenario

engine, tool, intent, plan = scenario()

def known_failure(params):
    tool.calls += 1
    return ToolCallResult(
        tool.name, False, "known partial failure", data={"ok": False},
        cost_vector=[0.1] * 8,
    )

tool.execute = known_failure
intent = replace(
    intent,
    delegated_authority=frozenset({"write", "memory.read"}),
    initial_budget=ResourceBudget((3.0,) * 8),
    permitted_side_effects=frozenset({
        SideEffectClass.IRREVERSIBLE, SideEffectClass.OBSERVATIONAL,
    }),
)
assert engine.tool_registry.get_metadata(tool.name).retry_safe is False
result = engine.run_governed_intent(intent, plan=plan, max_cycles=3)
assert [r.execution.state.value for r in result.receipts] == [
    "FAILED", "EXECUTED", "FAILED",
]
assert tool.calls == 1, "retry-unsafe write was invoked twice after recovery"
```

Granting both observational and irreversible side-effect classes matters:
otherwise the existing authority gate correctly blocks the recovery read before
the second write. This probe does not bypass admission. It demonstrates that
fresh authorization alone does not establish retry safety.

## Root cause and next correction

`AgenticClosure.evaluate` returns `REPLAN` for a known failure. The v5
`GoalPlanner.replan` retains the failed step after adding a memory-recovery step.
The engine never uses the trusted tool's `retry_safe`/`idempotent` metadata to
decide whether that failed operation may recur. A known failure does not imply
that partial effects were undone, and an observational recovery does not create
retry authority or idempotence.

The next focused correction should capture retry semantics before invocation
and prohibit automatic recurrence of a retry-unsafe failed operation. Preserve
its truthful `FAILED` state, real modeled cost, residuals, and lack of external
rollback evidence. Escalation or an explicitly authorized compensation protocol
must be designed before repeating it. Validate known failure, unknown outcome,
retry-safe failure, partial effects, broad/narrow intent envelopes, memory
qualification and cross-cycle continuity.

No fourth runtime repair is added in this pass because its three selected
consequential fixes are complete. This bounded adversarial assertion remains
failing and is explicitly reported alongside the passing existing quality
suite; it is not presented as fixed, skipped, or an expected-failure CI test.
It is an additional **NO-GO** release reason. No new release is authorized.
