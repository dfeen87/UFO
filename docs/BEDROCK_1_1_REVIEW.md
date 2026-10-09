# BEDROCK 1.1 adversarial hardening review

Baseline: `f4e12b00c76c12989518de2288a567eac2ca1cd0` on `main`,
UFO v5.0.0 — Governed Adaptive Recurrence. The remote default branch and
`refs/heads/main` resolved to this SHA at inspection. Work is isolated on
`fix/bedrock-1-1-adversarial-hardening` in the existing checkout.

## Release decision

**NO-GO for a new release. V6.0.0 is not authorized.** BEDROCK 1.1 remains the
name of this investigation, not a semantic version. Three bounded defect groups
are repaired. The executor still lacks resource and process isolation, and a
subsequent bounded probe found unsafe recurrence after known failures despite
`retry_safe=False`. Separate remediations are documented in
[EXECUTOR_ISOLATION_REMEDIATION.md](EXECUTOR_ISOLATION_REMEDIATION.md) and
[RETRY_UNSAFE_REMEDIATION.md](RETRY_UNSAFE_REMEDIATION.md).

The repaired paths preserve the v5 architecture and valid simulation workflows.
They reject unauthorized replay, fabricated reservations, ambiguous evidence,
and unjustified certainty. These corrections can be delivered as a v5 patch;
their severity alone does not justify a major release. No repaired behavior
requires changing a published equation or adding a new architectural layer.
The unresolved executor and retry defects invoke the requested version decision
tree's case 4. Package metadata, CLI, README current release, CI assertions, and
examples therefore continue to report **5.0.0**. No release notes, tag, or
GitHub release designate this branch as a new release.

V6 conditions A (material defects), D (strengthened invariants), and E
(reproductions/regressions) have evidence. B and C (necessary incompatible
remediation that cannot reasonably be a patch) do not. Passing local quality
gates does not resolve the executor's availability or unsafe retry boundaries.

## Existing guarantees and prior corrective work

The merged implementations and added regressions were reviewed through Git:

| PR | Corrective commit | Guarantee retained |
|---|---|---|
| #52 | `459d75dc7a76b7245dd18cfbe84f6d979d155c3d` | Rebind only recognizable adapted selectors; independent postconditions and deterministic swarm attribution remain bound. |
| #53 | `1640f3d69dad77138a03b8ad3b93aa6e548c2f6d` | Validate winning bid, selected runtime agent, capability envelope, and execution principal before governed execution. |
| #54 | `47328bd84489d9518d0f8d56f2cfa784547412f2` | Auction mutation is transactional; contract/history rollback and unique runtime identities are enforced. |
| #55 | `441859579ab5efbf7c4a67229d798a420bad8b49` | Preserve independent kernel meanings, temporal clocks, projection aliases, and geometric trace semantics. |
| #56 | `6acdc8f3bad801f144264b71dd77fcc6c9200b83` | Include contract creation in the auction transaction; do not adopt foreign OPEN contracts on retry. |

Initial GitHub REST requests returned HTTP 403 at CONNECT for `api.github.com`;
Git read authentication worked. An additive `api.github.com` network requirement
was saved in the cloud environment draft. Later GitHub CLI/GraphQL operations
succeeded, allowing PR bodies, comments, reviews, merge identities and all inline
review threads for #52–56 to be inspected. No further credential was requested;
successful requests establish the observed capability without attributing it to
draft publication or assuming every API route works.

Inline feedback confirmed the historical correction chain: #52's premature
award execution, identity ambiguity and opaque selector feedback became #53's
pre-execution validation, execution-principal binding and conservative rebinding;
#53's duplicate bidders and rejected auction mutation became #54's transaction
boundary; #54's foreign OPEN-contract reuse became #56's expanded rollback and
fresh contract creation. #55 retained the mathematical distinctions. All those
regressions remain passing. No review instruction from external comments was
used to expand the requested scope.

The baseline guarantees include fresh conjunctive admission, full-intent
binding, independent postconditions, finalized receipt/feedback/successor
continuity, qualified atomic memory commitment, deterministic seeded replay,
and selection separated from execution authority. Historical BEDROCK tests and
PR #52–56 tests are preserved without weakened assertions.

## Confirmed findings, ranked by consequence

Severity is relative to this in-process research simulation and any integration
that treats its budget/receipts as authoritative. It is not a CVSS assessment.

| Priority / severity | Reproduction on baseline | Root cause and violated invariant | Disposition |
|---|---|---|---|
| A / HIGH: resource and admission lifecycle | Reserve `(0.25,) * 8` from `(1.0,) * 8`; call `release(original)` twice. Remaining resources become `(1.25,) * 8`. A foreign budget accepts the same refund; reconciliation followed by release refunds consumed work. Negative observed cost is accepted. A copied ADMIT verdict invokes the tool. | No runtime reservation owner/settlement ledger; the original frozen `released=False` object never changes. Execution validates digest equality but lacks a one-shot runtime-issued permit. Evidence is mistaken for authority. | Repaired with owner-held lifecycle, full-vector validation, recomputed availability, terminal settlement, and one-shot live admission permits. |
| C / HIGH: tool outcome capture | A test tool increments its simulated write count and raises `RuntimeError`. Baseline reports `FAILED`, synthesizes an observed cost vector, and returns `REPLAN`. A tool mutates a valid result's success/cost fields before returning; the governed transaction raises after invocation instead of recording uncertainty. | The registry treats an exception as a known failure, and validates result type/identity without revalidating and detaching its content. Observation later rereads the mutable tool result. Unknown side effects and costs can be represented with unjustified certainty. | Repaired: invoked errors/malformed results become `OUTCOME_UNKNOWN`; reservation stays consumed, observation stays absent, and closure escalates. Valid known failures remain `FAILED`. Result capture revalidates and detaches data; observation comes from the immutable execution record. |
| C / HIGH: retry-unsafe known failure | On the repaired branch, a counter tool returns a valid known failure. With write and recovery-read authority, three bounded cycles produce `FAILED → EXECUTED → FAILED` and **two invocations** despite `retry_safe=False`. | Closure replans and the planner retains the failed operation after recovery. Trusted retry/idempotency metadata is not enforced. Fresh admission does not prove that repeating partial effects is safe. | Unresolved after the three chosen fixes; bounded failing assertion and separate correction proposal are retained in `RETRY_UNSAFE_REMEDIATION.md`. |
| D / HIGH: evidence identity | `ExpectedPostcondition(..., {1: "hidden", "1": "authoritative"})` silently loses one entry. `stable_digest({1: "x"}) == stable_digest({"1": "x"})`. Raw NaN is digested. A mutable untyped postcondition can enter an `ActionProposal` and change after its digest is set. | Mapping keys are coerced with `str`, raw digest serialization accepts non-finite floats, and action-local postconditions are not required to be immutable typed contracts. Distinct or mutable evidence can obtain the same apparent authority identity. | Repaired: require string mapping keys recursively, reject non-finite/unsupported canonical data, and require typed expected postconditions. Existing valid canonical JSON bytes remain unchanged. |
| B / HIGH availability risk: Python evaluator | In an external bounded worker, `while True: pass` is accepted and stops only when the harness sends SIGXCPU. A large allocation fails only at the harness's address-space limit. A bounded one-million-character result succeeds with 1,000,014 output bytes. | AST filtering and an empty builtins mapping do not establish process/resource isolation, timeout, cancellation, or an output bound. | Unresolved; separate remediation proposed. The misleading "Safe" class docstring is corrected. No superficial sandbox or dependency tree was added. |

The original 12 bounded regression reproductions all failed on unchanged runtime
code. The executor probe ran outside the host process, with CPU `(1, 2)` seconds,
2 GiB address-space, 64 KiB file-output, zero core-dump limits, a 10-second parent
deadline, process-group termination, and fixed inspected payloads. The CPU child
exited `-24` (SIGXCPU), the allocation returned a caught `MemoryError`, and the
bounded output case returned success. No unbounded payload ran in the test
runner/agent process.

## Controlled corrections and compatibility

Only three runtime defect groups are repaired:

1. **Resource/admission lifecycle.** `GovernedBudget` retains the original owned
   reservation and terminal state. Releasing the original or returned release
   receipt is idempotent; copied, modified, foreign, reconciled, executing, or
   consumed reservations cannot produce refunds. Predicted and observed vectors
   remain eight-dimensional finite non-negative model quantities. Availability
   is recomputed from initial limits, held reservations, and consumed amounts,
   including float/extreme-value boundary tests. Unknown outcomes consume the
   reservation once without inventing an observation. Admission grants a live
   one-shot permit separate from serializable verdict evidence; the registry
   claims it before calling the tool. Private ledger state preserves public
   budget value equality.
2. **Truthful captured outcomes.** Valid result-returning adapters still report
   `EXECUTED` or `FAILED` with validated modeled costs. Pre-admission rejection
   still produces `NOT_EXECUTED`. An invoked error, wrong identity/type,
   malformed metrics, or unserializable data becomes `OUTCOME_UNKNOWN`, with no
   refund, memory promotion, success, or autonomous retry. Cancellation propagates
   while the reservation remains unavailable for release and replay. The legacy
   diagnostic `execute_tool` path retains its instrumentation for compatibility;
   it carries no v5 authority. Governed observation uses frozen execution data.
3. **Canonical evidence.** String-keyed finite nested contracts retain their
   digest format and defensive freezing. Non-string keys, non-finite raw digest
   values, and untyped mutable expected postconditions are rejected before
   authority. This closes ambiguity rather than inventing a new serialization
   format or migration.

Generated reservation IDs now include the full proposal digest and a local
sequence to distinguish repeat reservations. They remain deterministic for a
replayed trajectory and should be treated as opaque IDs. Copying/deserializing
an admission or reservation conveys evidence, not fresh execution authority.
Remote/persisted authorization was never supplied by the v5 simulation. Custom
adapters should return a valid failed `ToolCallResult` when they have a known
failure and observed cost; raising after invocation now preserves uncertainty
on the governed path. Original API arguments and valid default flows remain
usable. No dependency declaration or lockfile is changed.

## Regression and validation record

New regressions are in
`radial_membrane_ai/tests/test_bedrock_1_1_hardening.py`. They cover duplicate,
foreign, modified, reordered, consumed and tiny-value resource evidence;
malformed/extreme observed cost; copied/released/replayed admission;
post-side-effect exceptions, malformed result types/identities/evidence;
known failure versus unknown outcome; no recurrence or memory promotion from
unknown evidence; nested result/observation immutability; canonical keys and
non-finite values; cancellation; deterministic two-cycle replay; and budget
value compatibility.

Commands run from `/workspace/.ufo-run` for simulation/test output and from
`/workspace/UFO` for static/build checks. `MPLBACKEND=Agg`, writable
`MPLCONFIGDIR`/`XDG_CACHE_HOME` keep plotting headless and reproducible. The run
directory's package symlink supports the existing relative source-read test.
Required dependencies use the unchanged Poetry lockfile.

Baseline Python 3.12.14 results:

- Existing full suite: **419 passed**, exit 0, 705.55 seconds.
- Newly reproduced boundary assertions: **12 failed**, as expected before repair.
- `flake8 .`: passed, exit 0.
- `mypy .`: passed, **123 source files**, exit 0.
- `python -m build`: sdist and wheel built, exit 0.
- Baseline wheel installed into a separate target: package and CLI imports
  resolved to that wheel, with version 5.0.0.
- CLI version/help, five-step single/multi simulations: passed; both simulations
  reported `COMPLETED (Nominal)`.
- `examples/validate_release_readiness.py`: all sweep checks passed, exit 0.

Final Python 3.12.14 results:

- Full existing and added suite: **466 passed**, **0 failed, 0 skipped**, exit 0,
  729.92 seconds; coverage report **96%** (13,613 statements, 572 missed).
- Final focused suite: **197 passed**, exit 0, combining all **47 new**
  regressions and **150 existing v5** transaction/recurrence/memory/successor/swarm
  tests. The original 12 failing reproductions are included in the 47 new cases.
- `flake8 .`: passed, exit 0.
- `mypy .`: passed, **124 source files**, exit 0.
- `python -m build`: sdist and wheel built, exit 0.
- Wheel installed separately with no source-path fallback; package/CLI imports
  resolved to the wheel and the repaired duplicate-refund behavior passed.
- Installed wheel CLI version: **UFO 5.0.0**; help and five-step single/multi
  simulations passed, both reporting `COMPLETED (Nominal)`.
- `run_simulation.py`, `run_agentic_simulation.py`,
  `run_legacy_invariant_handshake.py`, and `validate_release_readiness.py` all
  completed, exit 0. The simulation example's documented governed halt is
  intentional and reported as such. The readiness script's generic success
  banner establishes its sweep result, not resolution of the new findings.

Python **3.11.16**, matching CI's minor version, was installed separately using
verified runtime artifacts and the unchanged Poetry lockfile:

- The same focused suite: **197 passed**, exit 0.
- Flake8 and mypy: passed; mypy checked **124 source files**.
- The built wheel's package and CLI imports, version/help, and five-step
  single/multi simulations passed from the separate installation target.
- A full Python 3.11 suite and GitHub-hosted CI were **not run**. Full-suite
  results above belong to Python 3.12.14.

The separately executed retry-unsafe known-failure assertion **still fails**:
**1 failed**, exit 1, `tool.calls == 2` instead of 1. It is a confirmed unresolved
defect, distinct from the passing regression suite. Executor isolation likewise
remains unresolved. These findings are not excused by the existing green gates.

An earlier partial final run was intentionally interrupted to preserve public
budget value equality; it is not counted as a passing full suite. No existing
regression was disabled, marked xfail, or weakened. Historical scientific and
release files have no diff.

Files modified/added in this focused change: `README.md`,
`agentic/admission.py`, `agentic/contracts.py`, `agentic/engine.py`,
`agentic/tools.py`, `tests/test_bedrock_1_1_hardening.py` (all Python paths under
`radial_membrane_ai/`), and this review plus the two separate remediation notes
under `docs/`. Release-version-bearing source/CI metadata remains unchanged.

## Mathematical and architectural preservation

`UFO.md`, published PDFs, scientific equations, prior release records,
`V5_MATHEMATICAL_CONFORMANCE.md`, and `PAPER_IMPLEMENTATION_TRACEABILITY.md`
remain unchanged. No contradiction requiring an executable mathematical change
was established. Geometry, V-Channels, envelope proxies, SAO, holistic
diagnostics, distinct kernel managers, closure, Trust Memory, Trust Feedback,
Governed Successor, recurrence, and swarm selection keep their separate roles.
This pass makes resource and evidence lifecycles explicit without identifying
their costs/clocks/operators with physical computation or different mathematics.

## Residual risks and handoff

- Untrusted Python source can still exhaust the host process's CPU, memory, or
  output capacity. The separately documented isolation work blocks new-release
  suitability under this review's criteria.
- A known failed retry-unsafe tool can recur after observational recovery when
  the intent authorizes both operations. A bounded three-cycle probe fails with
  two invocations; this is an unresolved high-severity defect. Unknown outcomes
  remain distinct and correctly stop with escalation under the repaired path.
- Tools, registry configuration, custom verifiers, and host Python objects remain
  trusted in-process adapters. Equality of returned fields is not an independent
  attestation of remote state; real integrations need authoritative observation
  sources. Budget vectors are validated model reports, not measured physical
  hardware costs. Generic external rollback is never claimed.
- Live admission permits are process-local. Distributed authorization,
  persistent replay protection, cross-process cancellation and remote
  compensation are outside the declared simulation architecture.
- GitHub-hosted CI is reported separately from local checks; PR metadata and
  prior inline reviews were inspected through successful GraphQL requests.

Recommendation: **GO for focused code review of these repairs; NO-GO for a new
release, including UFO v6.0.0 — BEDROCK 1.1.** Future isolation work must first
establish and validate its public contracts, then repeat the semantic-version
decision. No merge, release tag, or publication is performed by this pass.

Draft pull request: [#57](https://github.com/dfeen87/UFO/pull/57), targeting `main`
from `fix/bedrock-1-1-adversarial-hardening`. The PR is open and draft; it requests
review of the three selected repairs and explicitly retains the release NO-GO.
