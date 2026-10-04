# U.F.O. paper-to-code traceability (v4.0.0)

This is the conformance record for the **research simulation**, not a claim of
production safety or a mathematical proof.  A name match is not counted as an
implementation: each row identifies the executable semantic mechanism and a
regression boundary.  Published PDFs remain the primary sources and are not
modified by this pass.

## Corpus inventory and count reconciliation

`docs/papers/` contains **11 PDF filenames and 10 byte-distinct documents**.
`A Governed Deformable Radial Membrane.pdf` and
`Publication Dynamic Radial Membrane.pdf` are byte-for-byte identical (SHA-256
`af0c803d…e9c7`); the latter is therefore an alias, not an eleventh document.
There is no evidence in the checked-in index that a master/foundation paper is
intended to be counted in addition to these files.  The defensible repository
count is consequently “10 unique documents (11 filenames)”, not an 11- or
12-paper claim.  The design note *The Invariant Handshake* names a 2026 paper,
but that paper is **not present** under `docs/papers/`; only the maintained
design note and implementation are present.  The smallest correction is the
clarification in `docs/README.md`, rather than adding, deleting, or editing a
publication.

Dates below are dates exposed by PDF metadata where present, not independently
verified publication dates. “Not exposed” means the file supplied no usable
date metadata; it must not be inferred from code comments.

| Exact indexed title / source filename | Date visible in file | Principal scope and boundary |
|---|---:|---|
| *A Dynamic Radial Membrane Architecture* | not exposed | Foundation/unit radial model and behavioral field; conceptual and simulation-oriented. |
| *A Governed Deformable Radial Membrane* | 2026-06-18 | Deformable geometry, governed cost and stability control; also stored under the `Publication…` alias. |
| *Publication Dynamic Radial Membrane* | 2026-06-18 | Exact duplicate alias of the preceding document, not another paper. |
| *Holistic Governor* | 2026-06-19 | Cross-signal coherence/stability field and intervention triggers; runtime scores do not constitute a control proof. |
| *Twelve Strings as Computational Facets* | 2026-06-19 | Twelve named, phase-positioned behavioral facets and state-aware corridors. |
| *V-Channel Activation and Kernel Regimes* | 2026-06-19 | Alignment/coherence/inverse-cost propagation and selectable regimes. |
| *Unified Math in U.F.O.* | 2026-06-25 | Cross-paper vocabulary and operator synthesis; maps abstraction levels rather than adding a separate runtime. |
| *Letter Depth Encoding (L.D.E.)* | 2026-06-25 | Symbol geometry plus distinct reconstruction metadata; geometry alone is not lossless. |
| *U.F.O. Federated Shard Architecture* | 2026-06-26 | Simulated shard admission, routing, mesh scoring and governance; not a distributed consensus protocol. |
| *The Symmetric Ascension Operator* | 2026-06-29 | Symmetry, projection, residual and promotion verdict chain. |
| *The Bounded Compute Envelope* | not exposed | Configured proxy envelope, brim energy and falsification ledger; not a wall-clock/space complexity guarantee. |

Supporting sources were also considered: `README.md`, the three architecture
notes in `docs/`, `research/README.md`, and the extracted L.D.E. text/search
artifacts in `research/lde-analysis/`.  Those are secondary or analysis sources,
not additional publications.  The invariant-handshake and multi-agent notes are
later architectural extensions and explicitly retain their design/simulation
status.

## Claim → invariant → mechanism → regression matrix

Statuses have their literal meanings: **implemented and tested** refers only to
the stated invariant; **partial** means that adjacent prose or stronger outcomes
remain unimplemented; **simulation-only** means no deployment or empirical
systems guarantee is made.

| Paper/source and claim/operator | Executable invariant | Implementation mechanism | Regression proof | Status and discrepancy/risk |
|---|---|---|---|---|
| Dynamic/Governed Membrane — radial behavioral field | Twelve stable phases; activation `[0,1]`; candidate delta finite before commit | `membrane.RadialMembrane`, `BehavioralString` | `test_bedrock_invariants.py`; `test_paper_conformance.py::test_twelve_facet_basis_has_stable_identity_phase_and_order` | **Implemented and tested.** The early unit/fixed picture is an abstraction; `BoundaryGeometry` is the later deformable surface. |
| Governed Membrane — deformable boundary | Radius clamped to configured positive range; tangent/curvature are finite-difference diagnostics | `boundary.BoundaryGeometry` | `test_model.py`, `test_pre_launch_sweep.py` | **Implemented but weakly tested.** Numerical curvature is diagnostic, not a smoothness theorem; constructor domains are not uniformly hardened. |
| Twelve Strings — computational facets | Exact ordered basis: depth, precision, technical detail, structural rigor, context sensitivity, transparency, initiative, exploration, creativity, tone, emotional warmth, conciseness | `RadialMembrane.strings`; `FacetVector`; `TensionAutomaton` | explicit basis regression in `test_paper_conformance.py` | **Implemented and tested** for identity/order. Persistence and policy fields are later implementation extensions. |
| V-Channel — aligned low-resistance corridor | Propagation multiplies target activation, angular alignment/coherence and inverse-cost weight; higher otherwise-equivalent cost cannot increase weight | `channels.cost_aware_propagation`, `governor.governed_propagation`, `facet.route_signal` | `test_model.py`, `test_pythagorean.py`, `test_architectural_governance.py` | **Partial.** Functions implement deterministic local scores; they do not solve a global optimal-route problem. Zero-cost is allowed, malformed values are not uniformly validated at every low-level helper. |
| Kernel regimes — behavioral physics selection | Enum-limited regimes and deterministic trigger order; multiphase state advances on explicit ticks | `kernel_regimes.RegimeManager`, `KernelRegimeType`; legacy `kernels.KernelRegimeManager` | `test_kernel_regimes.py` | **Implemented but weakly tested.** Threshold switching is implemented; no general hysteresis claim or anti-thrashing guarantee is established. Two manager abstractions remain a mapping risk. |
| Bounded Compute Envelope — geometric/cost bound | Brim energy and sampled capacity violations produce a configured admission verdict and ledger evidence | `envelope.BrimEnvelope`, `FalsificationStack`, `cost.RuntimeCostVector`, `AdmissibilityGate` | `test_envelope.py`, `test_architectural_governance.py` | **Simulation-only / partial.** Bounded quantities are model proxies and configured budgets, not CPU time, memory, or asymptotic compute. |
| Holistic Governor — suppression/control | Task, coherence and local cost influence activation delta; costly low-value expansion is suppressed | `Governor.update_membrane`; `HolisticGovernorField` diagnostics/triggers | `test_model.py`, `test_temporal_governance.py` | **Partial.** The local Governor commits control. Several holistic field outputs are trigger/diagnostic signals and do not independently mutate every execution path (“diagnostic-only,” not hidden control). |
| Governor — Lyapunov-style energy | Non-negative weighted energy for valid non-negative state; history trend/variance heuristic and optional dissipation | `Governor.compute_lyapunov_energy`, `is_stable`; `apply_lyapunov_dissipation` | `test_model.py`, `test_pre_launch_sweep.py` | **Heuristic, implemented and tested.** No proof of positive definiteness about a unique equilibrium or monotonic discrete-time decrease; temporary increases are permitted. Never read “stable” here as formal Lyapunov stability. |
| SAO — symmetric ascension | Opposite facets align; projection precedes verdict; only `ascend` appends promotion ledger evidence | `saopromotion.SAOPromotor.promote`; `collective_sao_promote` | `test_saopromotion.py`, `test_collective_reasoning.py` | **Partial.** Local promotion ledger mutation is atomic for rejection; the receiving “layer” is represented by evidence, not a transactional external datastore. Multi-agent all-party symmetry/policy is separately governed. |
| Federated Shards — workload projection/admission/routing | Evidence is finite/non-negative; only known evidence plus strict routing metadata accepted; terminal shard states override evidence | `routing.FederatedRoutingChannel`; `shard.FederatedShard.evaluate_admissibility`; `mesh.FederatedShardMesh` | projection/admission/routing falsification cases in `test_paper_conformance.py`; `test_federated_mesh.py` | **Implemented and tested** as an in-process simulation. `routed=True` is informational and never attestation. There is no network transport, freshness protocol, Byzantine tolerance, or distributed consistency guarantee. |
| Federated mesh — global coherence | Deterministic aggregate of route/trust/economic positives and residual/policy/latency/failure penalties | `FederatedShardMesh.compute_mesh_coherence`; collective mesh coherence | `test_federated_mesh.py`, `test_collective_reasoning.py` | **Partial.** Quarantine gates routing; the scalar coherence report is not itself global authorization and currently includes all shards in some averages. |
| L.D.E. — geometry-enriched representation | Dimensions match; generated geometry finite for validated normal configs; directed symbol transitions preserved | `lde_encode`, `LDEState`, `LDEBoundaryGeometry` | `test_lde_pipeline.py`; dimensional falsification regression | **Implemented and tested** as symbolic simulation. Depth is a weighted descriptive feature, not semantic understanding. |
| L.D.E. — full reconstruction | Full mode retains ordering, casing, punctuation/spacing/boundaries and satisfies `reconstruct(encode(T)) == T`; compressed mode refuses lossless API | `lde_encode`, new `lde_reconstruct` | adversarial round-trip and compressed-mode regressions in `test_paper_conformance.py` | **Implemented and tested.** Losslessness comes from reconstruction metadata, explicitly not from depth/boundary geometry alone. Alphabet normalization has known Unicode edge cases which fail rather than corrupt silently. |
| Temporal governance — short/durable horizons | Bounded deques, explicit tick order, decay/recovery, and promotion gates distinguish transient from durable evidence | `temporal.TemporalMembraneState`; SAO temporal gates | `test_temporal_governance.py` | **Implemented simulation.** No durable user-profile store, event-time clock, stale/future timestamp protocol, or production personalization claim. |
| Multi-agent design note — governed coupling | Policy envelopes intersect and collective admissibility precedes promotion; correction can roll back simulation state | `multi_agent`, `collective_reasoning`, `MeshCorrectness` | `test_multi_agent.py`, `test_multi_cluster.py`, `test_collective_reasoning.py` | **Later architectural extension / simulation-only.** Not a federated deployment protocol. |
| Semantic memory | Writes/reads/promotions use policy/admissibility contexts and curvature state is bounded by its operator rules | `semantic_memory` package and integration hooks | `test_semantic_memory.py` | **Later architectural extension.** In-memory governed records are not a secure persistent memory service. |
| Agentic planning/swarm/tooling | Planner budgets and tool policy gates constrain simulated execution; result validation is explicit | `agentic.AgenticEngine`, `GoalPlanner`, `SwarmAuctioneer`, tools | `test_agentic.py`, `test_agentic_hardening.py` | **Later architectural extension / experimental.** Not all generic tool side effects can be transactionally rolled back; it is not direct evidence for the original papers. |
| Invariant Handshake design note | Both scalar and tensor sides must independently satisfy the closure band | `adapters.invariant_handshake`; engine integration | `test_invariant_handshake.py`, `test_legacy_adapter_integration.py` | **Implementation-derived extension.** Its named publication is absent from the checked-in PDF corpus. |

## Cross-paper semantic reconciliation

* **Activation / phase / radius.** The twelve-string membrane uses activation as
  a clamped behavioral coefficient and radius as reasoning/deformation state.
  L.D.E. uses symbol frequency activation, fixed alphabet phase, and a separate
  textual boundary radius. These are analogous abstraction levels, not shared
  units; values must not be passed between them without an explicit adapter.
* **Cost / boundedness.** Local string cost, L.D.E. reconstruction cost, runtime
  cost vectors, brim energy, and shard `cost_factor` are different quantities.
  Their common role is pressure/admission; the code does not establish unit
  equivalence or a formal machine-resource upper bound.
* **Coherence.** Channel coherence, holistic score, collective similarity, and
  mesh coherence all normalize different evidence. They are not interchangeable
  probabilities. Mesh coherence is a diagnostic aggregate, whereas admission
  and policy checks are authorization gates.
* **Energy / stability.** “Lyapunov-style” is the accurate shared term. The
  non-negative energy functional and trend heuristic are executable, but a
  global convergence theorem is neither published here as a checked proof nor
  established by tests.
* **Admissibility / projection.** Projection changes geometric evidence; it does
  not confer trust. Shard `routed` metadata is recognized so public operators
  compose, but it is excluded from capacity/privacy/latency/cost decisions.
* **Residual / promotion.** Geometric residuals make rejected/projected content
  observable. A residual record is audit evidence, not permission to ascend.
* **Kernel transition.** The corpus motivates regimes; the package implements
  threshold/trigger selection. Hysteresis must not be inferred where no stateful
  enter/exit band is encoded.

No irreconcilable equation conflict was resolved by silently choosing a paper.
The material instead uses recurring words at several abstraction levels. The
mapping above is required whenever those values cross modules.

## Confirmed discrepancies and corrections in this pass

1. **Fixed:** routing projection emitted `routed=True`, while shard admission
   rejected it as unknown. Admission now recognizes exactly that strict marker,
   excludes it from trust arithmetic, and continues to reject unknown or
   malformed metadata and evidence.
2. **Fixed:** L.D.E. verified reconstruction internally but exposed no
   reconstruction operator. `lde_reconstruct` now makes the full/compressed
   distinction executable and validates retained reconstruction evidence.
3. **Fixed:** the documentation index called 11 paths “the complete set” without
   disclosing that two are identical. It now reports filename and unique-document
   counts and identifies the absent Invariant Handshake PDF.
4. **Not silently fixed:** boundary constructor validation, complete low-level
   channel validation, global mesh evidence filtering, regime hysteresis, and
   transactional receiving layers require separate design decisions. Their
   limits are recorded rather than equations being invented.

## Claims deliberately left research-only

* Project Rainbow and the governed operator stack are inspectable simulations,
  not deployed empirical validation.
* “Bounded compute” bounds configured geometric/cost proxies, not physical
  runtime, memory, energy consumption, or worst-case complexity.
* Mesh federation is in-process orchestration, not consensus, cryptographic
  attestation, stale-evidence prevention, or Byzantine fault tolerance.
* Holistic coherence and Lyapunov-style energy are falsifiable diagnostics and
  control heuristics, not proofs of optimality, safety, or convergence.
* SAO provides deterministic simulated verdict/evidence flow, not a general
  theorem that information remains semantically equivalent across arbitrary
  receiving systems.
* L.D.E. geometry is not a lossless code by itself; only full mode plus its
  explicit reconstruction metadata supports exact round trips.
* Semantic memory, agentic planning, and multi-agent coupling are later
  architectural extensions and do not retroactively become original-paper
  claims.

## Remaining research questions

1. What units/calibration map local cost, runtime cost, brim energy and shard
   economics without treating unlike proxies as commensurate?
2. Should kernel transitions acquire explicitly researched hysteresis, or should
   threshold-only switching remain the reference behavior?
3. Which holistic metrics are intended as diagnostic-only, and which should be
   mandatory authorization inputs on every newer execution path?
4. What temporal attestation/freshness model is required before the shard mesh
   can make claims beyond an in-process deterministic simulation?
5. Can a formal discrete-time argument establish a Lyapunov condition under a
   restricted workload, without falsely asserting monotonicity for all tasks?
6. What canonical Unicode normalization/alphabet policy should full L.D.E.
   support beyond exact cases already retained by reconstruction metadata?
7. Should the duplicate publication alias remain for stable links, or be marked
   in external release metadata as an alias while preserving both published
   files?
