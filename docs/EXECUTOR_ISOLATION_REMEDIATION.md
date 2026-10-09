# Python executor isolation: separate remediation proposal

`PythonCodeExecutorTool` evaluates AST-filtered source in the caller's Python
process. Forbidden imports/attributes/builtins limit syntax; they do not bound
computation or create a security sandbox. A governed model-cost reservation does
not limit CPU time, heap allocation, or serialization output.

The BEDROCK 1.1 probe accepted `while True: pass` and stopped only at an external
worker's CPU limit. A large list allocation failed only at that harness's
address-space cap. A one-million-character string returned 1,000,014 output
bytes. These fixed payloads were run in disposable workers with CPU, memory,
file-output and core limits, plus a parent deadline and process-group cleanup.
They were never executed unbounded in the host test process.

Until a complete isolation design is implemented, use only trusted, bounded
simulation source. The current declaration remains an in-process evaluator.

## Required design work

1. Define a narrow source/result protocol with primitive JSON data and explicit
   source, AST, result-depth, byte-size, and output bounds. Validate both ends;
   do not use pickle or expose host objects in the worker.
2. Execute in a fresh child/container with explicitly chosen filesystem/network
   authority. Restricted builtins alone cannot establish this boundary. Decide
   whether this operation is pure local computation and enforce that decision.
3. Enforce CPU and memory limits before source parsing/execution. The parent must
   own a wall-clock deadline, cancellation, bounded IPC reads, and termination
   of the entire worker/process group. Cleanup must survive exceptions and
   invalid/truncated responses. Fail closed on unsupported platforms/limits.
4. Distinguish known local failure from unknown external effects. Killing a
   worker is not evidence that remote side effects were rolled back. Retain the
   reservation whenever the actual modeled cost/outcome cannot be established.
5. Preserve deterministic semantic receipts and finite model costs; timing is
   diagnostic. Validate bounded successful programs, loops, large allocations,
   output floods, malformed/cyclic values, worker crashes, cancellation,
   truncation, cleanup, and lack of host mutation in disposable test workers.
6. Review public compatibility once the limits, platforms, and execution
   semantics are concrete. A change to accepted programs/resource contracts may
   warrant a major release, but the codename or defect severity cannot authorize
   v6 in advance.

This is an implementation proposal, not a claim that those controls exist.
Rejecting only `while` or adding a thread timeout would leave large finite
expressions, comprehensions, allocation, and serialization hazards unresolved.
No such partial correction is presented as a sandbox in this pass.
