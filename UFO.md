**Governed Agentic Closure in U.F.O.**

A Bounded Intent–Action–Evidence Architecture for Governed Agentic AI

Don Michael Feeney Jr.


### Note / Research Disclaimer

**Research Note.** This paper presents one research interpretation and engineering approach to governed agentic artificial intelligence within the U.F.O. architecture. It is not intended to define *agentic AI* as a field, establish a universal architecture for autonomous systems, or suggest that agentic intelligence must take this form. Agentic AI remains a rapidly developing area of research and engineering, and I expect many distinct architectures, theories, governance models, and forms of agency to emerge as the field matures. The mechanisms presented here should therefore be understood as a specific, testable contribution to that broader and continuing exploration.

### Acknowledgment of AI-Assisted Engineering

**Acknowledgment.** The development and hardening of U.F.O. v4.0.0 and the formulation and implementation planning of v5.0.0 were substantially assisted by **OpenAI’s ChatGPT and Codex**. These tools accelerated code analysis, adversarial review, debugging, technical synthesis, documentation, and the translation of research concepts into explicit engineering contracts and falsifiable implementation requirements. Their use was instrumental in bringing U.F.O. to its present level of technical maturity, particularly during the BEDROCK hardening process and the development of Governed Agentic Closure. Final architectural decisions, research claims, interpretations, and responsibility for the work remain with the author.

### U.F.O. Research Software

Feeney, D. M., Jr. (2026). *U.F.O. (User-Formed Optimization)* \[Computer software\]. Zenodo. <https://doi.org/10.5281/zenodo.21401979>

### Abstract

Agentic artificial intelligence is commonly described through planning, tool use, memory, reflection, and multi-agent coordination. These capabilities, however, do not by themselves define a governed agentic system. A planner may propose an action without possessing authority to execute it; a successful tool invocation may fail to produce the intended outcome; an observation may be incomplete or transient; and completion of one action does not necessarily justify continued autonomous execution.

This paper extends the User-Formed Optimization (U.F.O.) architecture by defining **Governed Agentic Closure**, an engineering model that composes existing U.F.O. mechanisms into a recurrent agentic execution cycle:

$$\text{Intent} \rightarrow \text{Plan} \rightarrow \text{Admission} \rightarrow \text{Execution} \rightarrow \text{Verification} \rightarrow \text{Residual} \rightarrow \text{Reflection} \rightarrow \text{Memory} \rightarrow \text{Closure}.$$

The model builds upon the radial membrane, Twelve Strings, V-Channel propagation, Bounded Compute Envelope, Holistic Governor, temporal governance, Symmetric Ascension Operator, federated architecture, semantic memory, and existing planning and tool-execution components already present in U.F.O. Rather than introducing a replacement architecture, this work defines the contracts required for those mechanisms to operate as a coherent agentic system.

Goals are represented as **Intent Contracts** containing explicit success conditions, immutable constraints, resource bounds, delegated authority, and side-effect limits. Planner-generated actions remain proposals until they pass **Prospective Agentic Admission**. Authorized execution produces observations that are independently evaluated against expected postconditions, generating an **Agentic Residual Vector** describing discrepancies in goal satisfaction, resource use, policy state, system stability, and execution certainty.

Reflection is treated as a candidate state transition rather than privileged self-modification: proposed membrane changes must satisfy the same validation and commit discipline applied elsewhere in the U.F.O. architecture. Likewise, execution evidence does not automatically become durable memory. Observations remain provisional until they satisfy applicable policy, temporal, and promotion requirements.

The cycle terminates in an **Agentic Closure Operator**, which determines whether the system may continue, replan, constrain its behavior, reflect, escalate, or halt. The governing principle is that **continued autonomy is not assumed; it is re-earned after every governed action cycle**.

The contribution is therefore not a claim of general intelligence, perfect planning, universal rollback, or formally guaranteed safe autonomy. It is an implementation-oriented reference model for bounded and inspectable agency in which planning is separated from authorization, execution from verification, observation from durable memory, and action completion from permission to act again.

# 1. Purpose and Engineering Motivation

U.F.O. v4.0.0 established a hardened baseline for the User-Formed Optimization architecture. The BEDROCK series examined the system from three complementary directions: low-level engineering correctness, correspondence between published research concepts and executable mechanisms, and final closure of implementation gaps that could be resolved without introducing unsupported theory. That process strengthened numeric trust boundaries, atomic state transitions, membrane and boundary validation, V-Channel contracts, federated admission, Symmetric Ascension Operator ordering, temporal evidence handling, L.D.E. reconstruction, and the distinction between diagnostic signals and authorization state.

The principal result of BEDROCK was not the addition of a new capability. It was a clearer understanding of where U.F.O. already possesses strong mechanisms and where those mechanisms remain insufficiently composed.

The current architecture already contains many components normally associated with agentic AI: goal decomposition, tool invocation, bounded resource accounting, reflection, temporal state, semantic memory, multi-agent coordination, swarm contracting, residual evidence, and governed promotion. These components are functional individually and, in several cases, are already integrated into execution paths. However, they do not yet form a single explicit transactional model of agency.

This distinction is important.

A system can contain a planner, tools, memory, and reflection without establishing what conditions permit it to continue acting autonomously. Similarly, a tool may report successful execution without proving that the intended postcondition was achieved. A planner may select an action without demonstrating that the action is authorized. An observation may be recorded without establishing whether it is sufficiently reliable or persistent to become durable memory. A reflection mechanism may propose a useful change while still requiring validation before authoritative system state is modified.

BEDROCK repeatedly exposed this class of problem: components that were individually reasonable could still violate system intent when composed in the wrong order.

Three examples are especially relevant to agentic execution.

First, the routing and shard-admission path demonstrated that **metadata and admissibility evidence must remain semantically distinct**. Routing information may describe how an object moved through the system, but it must not improve the trust decision that determines whether the object is accepted.

Second, BEDROCK III identified that an intermediate Symmetric Ascension Operator verdict could not safely commit promotion evidence before temporal governance produced the final decision. The correct ordering was:

$$\text{evaluate} \rightarrow \text{final verdict} \rightarrow \text{commit}$$

rather than:

$$\text{preliminary verdict} \rightarrow \text{commit} \rightarrow \text{additional governance}.$$

Third, atomic membrane and boundary updates established a broader state-management rule: a candidate transition must be completely validated before it becomes authoritative. A failure late in a computation must not leave part of the previous state replaced and part of it unchanged.

These findings generalize directly to agentic systems.

For U.F.O. v5.0.0, the analogous requirements are:

$$\text{plan selection} \neq \text{execution authorization},$$

$$\text{tool success} \neq \text{verified outcome},$$

$$\text{observation} \neq \text{durable memory},$$

and

$$\text{completed action} \neq \text{permission to continue acting}.$$

The purpose of this paper is therefore not to introduce an entirely new agent architecture. It is to define the **engineering contracts and execution ordering required to compose existing U.F.O. mechanisms into one governed recurrent system**.

The proposed model introduces an explicit agentic cycle:

$$\text{Intent} \rightarrow \text{Plan} \rightarrow \text{Admission} \rightarrow \text{Execution} \rightarrow \text{Verification} \rightarrow \text{Residual} \rightarrow \text{Reflection} \rightarrow \text{Memory} \rightarrow \text{Closure}.$$

Each transition has a distinct responsibility.

The planner proposes candidate behavior. Admission determines whether the proposal may execute. Execution interacts with a tool or environment. Verification determines whether the observed result satisfies the expected postcondition. Residual analysis records discrepancies between expected and observed behavior. Reflection may propose changes to internal state. Memory governance determines what evidence becomes durable. Closure determines whether another autonomous cycle is permitted.

The final operation is the central addition of this work.

An agentic system should not continue merely because additional plan steps exist. Continued execution should depend on whether the previous action left the system in a state that remains within its authority, policy, resource, stability, and evidence constraints.

This paper refers to that decision as **Agentic Closure**.

U.F.O. v5.0.0 is intended to implement this model as a reference execution path over the existing v4 foundation. The objective is deliberately narrow: establish a deterministic and falsifiable contract for governed agentic execution that can be represented in code, exercised through seeded simulations, and challenged through adversarial tests.

The work does not attempt to prove general intelligence, universal safety, optimal planning, or arbitrary environmental correctness. Its engineering objective is more concrete:

**Given an intent, a current U.F.O. state, and a sequence of proposed actions, define precisely when the system may act, how the result changes authoritative state, and whether the system is permitted to act again.**

That is the problem addressed by the remainder of this paper.

# 2. Existing U.F.O. Agentic Substrate

U.F.O. v4.0.0 does not begin the transition to agentic AI from an empty architecture. The repository already contains planning, tool execution, reflection, semantic memory, temporal governance, bounded-compute mechanisms, multi-agent coordination, and several forms of admissibility and promotion logic. The purpose of v5.0.0 is therefore not to attach an unrelated agent framework to U.F.O., but to compose these existing mechanisms into a single execution model with explicit authority, verification, state-transition, and continuation contracts.

This section identifies the principal components already available in the v4 baseline and distinguishes their present responsibilities from the composition rules introduced later in this paper.

## 2.1 Agentic Runtime

The current AgenticEngine provides the closest existing approximation to a complete U.F.O. agentic cycle.

For an input goal, the engine presently:

1.  creates a plan,

2.  advances the underlying U.F.O. membrane,

3.  observes membrane tension and boundary curvature,

4.  evaluates an admissibility closure condition,

5.  executes the next planned tool step,

6.  evaluates whether the result or membrane state requires reflection,

7.  applies resulting behavioral adjustments,

8.  repeats until the plan completes, fails, or reaches an execution limit.

This establishes an important starting point. U.F.O. already models agent behavior as a recurrent interaction between **task execution and internal governed state** rather than as a stateless sequence of tool calls.

The existing runtime can be viewed as a sequence of discrete agentic execution steps indexed by $t$. Here, $t$ does not represent physical time directly; it identifies the current iteration of the agentic control cycle. A variable written with subscript $t$ therefore represents the value of that quantity at the beginning of or during the current execution step, while $t + 1$ represents the resulting state after the step has been processed.

At a high level, the existing runtime can be represented as

$$X_{t} \rightarrow P_{t} \rightarrow A_{t} \rightarrow E_{t} \rightarrow R_{t} \rightarrow X_{t + 1},$$

where

$$X_{t} = \text{current U.F.O. system state at agentic step }t,$$

$$P_{t} = \text{plan state or candidate plan step selected at }t,$$

$$A_{t} = \text{admissibility evaluation applied before execution},$$

$$E_{t} = \text{tool or action execution occurring during the step},$$

$$R_{t} = \text{reflection or corrective response produced from the resulting evidence},$$

and

$$X_{t + 1} = \text{updated authoritative U.F.O. state after the cycle completes}.$$

The notation is intentionally discrete because the agentic system is modeled as a sequence of governed state transitions rather than as a continuous-time process. Physical timestamps, execution latency, temporal-memory horizons, and other real-time quantities may still exist within $X_{t}$, but they are distinct from the logical execution index $t$.

This discrete execution model also provides a natural connection to the existing **Invariant Handshake**. At the first governed agentic cycle, $t = 1$, the state $X_{1}$ represents the U.F.O. agentic side of the legacy–AI relationship evaluated by the handshake. In that relationship, legacy computation is represented through scalar or binary-compatible stress terms, while the U.F.O. side is represented through tensor-derived state such as Lyapunov-style energy and activation norm. Both sides are evaluated against a shared structural channel capacity $c$ through the geometric invariant

$$i = \frac{a^{2} + b^{2}}{c^{2}}.$$

The construction is Pythagorean in form: the stress contributions $a$ and $b$ occupy the two orthogonal components of the triangular relation, while $c$ represents the shared connectivity or channel-capacity boundary through which the two computational regimes interact. The handshake is admissible only when the legacy side and the agentic side independently remain within the configured stability band around unity:

$$i_{\text{legacy}} \approx 1,\quad\quad i_{\text{agentic}} \approx 1,$$

such that

$$\text{handshakeAllowed} = \text{legacyStable} \land \text{agenticStable}.$$

Within the agentic model developed in this paper, $t = 1$ therefore identifies the **first discrete U.F.O. execution state participating on the agentic side of this binary computational relationship**. Subsequent states $X_{2},X_{3},\ldots$ represent later governed agentic transitions that may again encounter scalar, binary-compatible, or legacy execution boundaries through the same handshake mechanism.

The notation must remain precise: $t = 1$ and $i \approx 1$ express different properties. The former identifies the first logical step in the agentic execution sequence; the latter represents geometric closure between stress contributions and shared channel capacity. Their relationship is architectural rather than algebraic. The execution state occupies the agentic side of the handshake, while the invariant determines whether communication across the two computational regimes remains admissible.

This places the proposed agentic execution cycle within an existing U.F.O. interoperability boundary rather than treating agency as a separate layer floating above the original geometry.

The present runtime nevertheless lacks one stronger composition rule. Several transitions in the current loop do not yet possess the semantics required for a complete governed autonomous transaction.

In particular, the existing runtime does not yet establish one authoritative chain spanning

$$\text{proposal} \rightarrow \text{authorization} \rightarrow \text{execution} \rightarrow \text{verification} \rightarrow \text{state reconciliation} \rightarrow \text{continuation}.$$

Paper XII defines that missing contract.

## 2.2 Goal Planning and Tool Execution

The current GoalPlanner converts an input goal into a structured Plan containing ordered PlanStep objects. Each step may identify a tool, parameters, completion state, observed result, cost impact, and tension impact.

The present planner is intentionally simple. Goal text is classified into broad execution patterns such as code execution, research or search, API interaction, or a general workflow. The resulting plan then invokes tools through the U.F.O. tool registry.

This implementation already establishes several useful engineering concepts:

$$\text{Goal} \rightarrow \text{Plan} \rightarrow \text{Ordered Steps} \rightarrow \text{Tool Calls}.$$

It also maintains explicit plan status and cumulative resource values.

However, the planner presently combines responsibilities that v5 should separate more strongly:

$$\boxed{\text{Planning} \neq \text{Authorization}}$$

A planner should answer:

What action appears useful next?

It should not independently answer:

Is the system permitted to perform that action?

This distinction allows U.F.O. to remain planner-independent. A future planner could be deterministic, search-based, language-model-generated, human-supplied, or produced by another agent without changing the governance layer surrounding it.

The planner therefore becomes a **proposal generator rather than an authority source**.

The tool layer already provides another useful foundation. ToolCallResult carries structured execution evidence including success state, returned data, multidimensional cost information, side-effect rating, tension contribution, and measured execution time.

These values create the beginnings of an executable evidence contract.

What they do not yet establish is whether the action should have been permitted **before** execution or whether the reported result actually satisfies the intended postcondition **after** execution.

This leads to two separate v5 requirements:

$$\text{Prospective Admission} \rightarrow \text{Execution}$$

and

$$\text{Execution} \rightarrow \text{Outcome Verification}.$$

The distinction is fundamental. Resource, authority, or side-effect violations discovered only after a consequential action may be too late to prevent the state change.

## 2.3 Membrane State and Reflection

The radial membrane remains the behavioral-state substrate of U.F.O.

Its Twelve Strings provide an ordered behavioral basis, while the deformable boundary, tension state, curvature, V-Channel propagation, and Governor provide mechanisms for representing and modifying computational posture.

The current ReflectionEngine already connects execution behavior back into this state.

Reflection may be triggered by:

- tool failure,

- membrane instability,

- or both conditions simultaneously.

The resulting ReflectionRecord preserves the trigger, execution context, tension, curvature, explanatory text, and proposed corrective behavior. Reflection can then suggest activation changes intended to increase precision, structural rigor, or conciseness, or to suppress costly exploratory behavior.

This is already a meaningful agentic feedback mechanism:

$$\text{Execution Evidence} \rightarrow \text{Reflection} \rightarrow \Delta M.$$

BEDROCK, however, established a stronger rule for authoritative state:

A proposed state change should not become authoritative merely because the component proposing it is trusted.

Accordingly, Paper XII generalizes reflection into

$$\Delta M_{t}^{\ast} = F\left( \rho_{t},X_{t} \right),$$

where $\Delta M_{t}^{\ast}$ is a **candidate** membrane modification derived from current residual evidence and system state.

A candidate membrane state is then constructed as

$$M_{t}^{\ast} = M_{t} + \Delta M_{t}^{\ast},$$

followed by an explicit validation and commit boundary:

$$M_{t + 1} = \left\lbrace  \begin{array}{l}
M_{t}^{\ast}, \quad \mathrm{Valid}\left( M_{t}^{\ast} \right), \cr
M_{t}, \quad \text{otherwise}.
\end{array} \right.$$

This extends BEDROCK's candidate–validate–commit discipline into agentic self-modification.

Reflection remains adaptive.

It does not become privileged.

## 2.4 Bounded Compute and Admissibility

U.F.O. already contains several mechanisms for limiting computation and rejecting inadmissible state.

The Bounded Compute Envelope represents configured computational and geometric pressure. Runtime cost vectors represent multidimensional execution cost. Admissibility mechanisms determine whether specified system conditions remain within acceptable bounds. BEDROCK hardened these pathways against malformed and non-finite values and made several previously implicit domains explicit.

These mechanisms provide most of the raw material required for **Prospective Agentic Admission**.

What remains missing is a single action-level contract combining them before external execution.

For a proposed action $a_{t}$, v5 should be able to evaluate

$$\mathcal{A}^{-}\left( a_{t},I,X_{t} \right)$$

before the action crosses an execution boundary.

The existing U.F.O. mechanisms can provide evidence for that decision, including:

- current membrane stability,

- cost pressure,

- policy state,

- capability requirements,

- remaining resource envelope,

- expected side-effect exposure,

- and delegated authority.

The new contribution is therefore primarily one of **composition and ordering**, not architectural replacement.

## 2.5 Temporal Governance, Residual Evidence, and Semantic Memory

U.F.O. already distinguishes transient state from information that persists across time.

Temporal membrane state maintains bounded histories and supports explicit decay, accumulation, and recovery behavior. Residual ledgers preserve failures or discrepancies without treating those records as authorization. Semantic memory provides policy-bound local and shared records with controlled write, read, overwrite, and promotion behavior.

These mechanisms establish an important architectural principle:

$$\boxed{\text{Observed Evidence} \neq \text{Durable Knowledge}}$$

The current semantic-memory subsystem already performs admissibility checks before memory writes and shared promotion. It also preserves residual evidence during destructive overwrite and maintains explicit memory-event history.

Paper XII extends this distinction across the complete agentic execution path.

An execution observation should initially be treated as **provisional evidence**:

$$J_{t}.$$

It may then pass through verification, policy evaluation, temporal qualification, and promotion:

$$o_{t} \rightarrow J_{t} \rightarrow T_{t} \rightarrow \mathcal{A}_{M} \rightarrow \mathrm{Promote} \rightarrow \mathcal{M}_{t + 1}.$$

Here, $o_{t}$ denotes the immediate observation associated with the current action, $J_{t}$ denotes provisional evidence, $T_{t}$ denotes temporally evaluated evidence state, $\mathcal{A}_{M}$ denotes memory admissibility, and $\mathcal{M}_{t + 1}$ denotes the resulting durable governed memory state if promotion succeeds.

This prevents a single malformed, incorrect, transient, or unverified observation from automatically becoming persistent agent state.

The engineering rule is therefore:

$$\boxed{\text{Experience} \neq \text{Memory}}$$

Memory is a governed consequence of experience.

## 2.6 Symmetric Ascension and Final-Verdict Semantics

The Symmetric Ascension Operator provides another important component for v5.

BEDROCK III corrected an ordering defect in which promotion evidence could be committed before temporal governance produced the final authorization result.

The corrected ordering established:

$$\text{geometric evaluation} \rightarrow \text{admissibility} \rightarrow \text{residual evaluation} \rightarrow \text{temporal governance} \rightarrow \text{final verdict} \rightarrow \text{commit}.$$

This result generalizes beyond SAO.

Agentic execution should obey the same rule:

$$\boxed{\begin{matrix}
\text{No authoritative evidence of an event should be committed} \cr
\text{before the final governing verdict for that event exists.}
\end{matrix}}$$

This becomes particularly important for:

- action authorization,

- reflection commits,

- memory promotion,

- plan revision,

- delegated authority,

- and final agentic continuation.

A preliminary result may be useful evidence.

It is not necessarily an authoritative transition.

## 2.7 Multi-Agent and Swarm Coordination

U.F.O. already supports multi-agent orchestration and swarm-style task assignment.

The existing swarm layer creates contracts containing goals, required capabilities, and budget constraints. Candidate agents produce bids based on capability overlap, expected cost, current tension, and stability margin. An auction selects an eligible winner deterministically from admissible bids.

This gives v5 an existing mechanism for

$$\text{Task} \rightarrow \text{Candidate Agents} \rightarrow \text{Bids} \rightarrow \text{Selection}.$$

However, agent selection currently flows naturally toward execution.

Paper XII introduces a deliberate additional boundary:

$$\boxed{\text{Selection} \neq \text{Authorization}}$$

Winning a contract establishes which agent is the preferred executor.

It does not establish that every action proposed by that agent is authorized.

If agent $U_{i}$ receives delegated authority $\alpha_{i}$, then any action it proposes must remain within that authority:

$$a_{t} \in \alpha_{i}.$$

Replanning must not silently expand it.

Absent an explicit new delegation event,

$$\alpha_{t + 1} \subseteq \alpha_{0}.$$

This allows multi-agent delegation to remain bounded even as local agents independently plan or replan execution.

## 2.8 BEDROCK as the Foundation for Agentic Composition

The relevance of BEDROCK to v5 is therefore broader than input validation.

The three v4 hardening passes established several reusable engineering laws:

$$\text{metadata} \neq \text{evidence},$$

$$\text{proposal} \neq \text{authorization},$$

$$\text{preliminary verdict} \neq \text{final verdict},$$

$$\text{candidate state} \neq \text{authoritative state},$$

and

$$\text{recorded event} \neq \text{permission}.$$

These distinctions become the foundation of Governed Agentic Closure.

The v4 architecture already supplies the computational mechanisms required to represent planning, bounded execution, adaptive state, temporal behavior, promotion, memory, and multi-agent coordination.

What it does not yet supply is a common protocol governing how those mechanisms exchange **authority, evidence, and state** during one complete autonomous action cycle.

That missing protocol is the principal subject of Paper XII.

The proposed v5 architecture therefore does not replace the existing agentic subsystem. It introduces explicit contracts between its stages so that the system can answer, for every consequential action:

1.  **What was intended?**

2.  **What action was proposed?**

3.  **Was that action authorized before execution?**

4.  **What actually occurred?**

5.  **Was the intended outcome verified?**

6.  **What discrepancy remained?**

7.  **What internal state may change because of that evidence?**

8.  **What information may become durable?**

9.  **Is another autonomous action permitted?**

These questions define the engineering boundary between possessing agentic components and operating as a governed agentic system.

# 3. Governed Agentic Execution Cycle

The central contribution of Paper XII is an explicit execution order for governed agency in U.F.O. The individual mechanisms required for this process already exist in the v4 architecture in partial or distributed form. The purpose of the v5 execution cycle is to define how those mechanisms compose so that authorization, execution, evidence, state mutation, memory, and continued autonomy occur in a deterministic and inspectable order.

The cycle is modeled as a sequence of discrete governed transitions indexed by $t$. At each step, the system begins with an authoritative state $X_{t}$ and an active intent $I$. It may propose an action, but no external execution or authoritative internal mutation is assumed until the relevant governance boundary has been satisfied.

The complete agentic cycle is represented as

$$\begin{matrix}
\text{Intent} \rightarrow \text{Plan} \rightarrow \text{Admission} \rightarrow \text{Execution} \rightarrow \text{Observation} \cr
 \rightarrow \text{Verification} \rightarrow \text{Residual} \rightarrow \text{Reflection} \rightarrow \text{Memory} \rightarrow \text{Closure}.
\end{matrix}$$

If closure permits continued execution, the resulting authoritative state becomes the basis for the next cycle:

$$X_{t} \rightarrow X_{t + 1} \rightarrow X_{t + 2} \rightarrow \cdots$$

The important property is not repetition alone. Every iteration must independently earn the right to cross its next execution boundary.

## 3.1 Initial Agentic State

Each agentic cycle begins with two authoritative inputs:

$$\left( I,X_{t} \right),$$

where $I$ is the active **Intent Contract** and $X_{t}$ is the current governed U.F.O. state.

The Intent Contract defines what the system is attempting to accomplish and under what constraints. The system state describes the conditions under which that intent is currently being pursued.

At this stage, U.F.O. has not yet committed to a particular action.

This distinction is important because the goal should remain stable across plan revisions unless an explicit authority-bearing event changes it. A failed tool call, elevated membrane tension, or poor plan does not automatically authorize the system to redefine the original objective.

The execution cycle therefore begins with

$$\boxed{\text{Intent} \neq \text{Plan}}$$

and

$$\boxed{\text{Plan} \neq \text{Action}}$$

An intent establishes the objective and governing limits. Planning determines one possible way of advancing toward it.

## 3.2 Action Proposal

Given the current intent and state, a planner produces a candidate action:

$$a_{t}\mathcal{= P}\left( I,X_{t} \right),$$

where $\mathcal{P}$ denotes the planning mechanism.

The action $a_{t}$ is not yet executable authority. It is a proposal describing what the planner believes should occur next.

The proposal should contain enough information for downstream governance to reason about the action before execution. At minimum, the system must be able to identify the requested capability, parameters, expected outcome, estimated resource pressure, anticipated side-effect class, and the relationship between the action and the current intent.

Paper XII deliberately treats the planner as untrusted with respect to authority.

The planner may be sophisticated or simple. It may be deterministic, model-generated, search-driven, externally supplied, or produced by another U.F.O. agent. None of these cases changes the next requirement:

$$\boxed{\text{A proposed action cannot authorize itself.}}$$

The next boundary is therefore prospective admission.

## 3.3 Prospective Agentic Admission

Before execution, the candidate action is evaluated against the current intent and authoritative system state:

$$V_{t}^{-} = \mathcal{A}^{-}\left( a_{t},I,X_{t} \right),$$

where $\mathcal{A}^{-}$ denotes **Prospective Agentic Admission** and $V_{t}^{-}$ is the resulting pre-execution verdict.

The superscript $( - )$ denotes that the decision occurs **before** the external action.

The admissibility layer evaluates whether the proposal remains consistent with the system's current authority, policy, budget, capability, stability, and side-effect constraints.

A useful verdict domain is

$$V_{t}^{-} \in \lbrace \text{ADMIT},\text{CONSTRAIN},\text{REPROJECT},\text{BLOCK},\text{ESCALATE}\rbrace .$$

ADMIT permits execution of the specific approved proposal.

CONSTRAIN indicates that execution may remain possible only after narrowing parameters, cost, scope, or side-effect exposure.

REPROJECT returns the proposal to planning because an alternative representation or execution path is required.

BLOCK prevents the action.

ESCALATE indicates that the system lacks sufficient authority to resolve the action autonomously and requires an external authority boundary.

Only an admissible final pre-execution verdict may cross into execution:

$$V_{t}^{-} = \text{ADMIT} \Rightarrow \mathrm{Execute}\left( a_{t} \right).$$

For all other verdicts,

$$V_{t}^{-} \neq \text{ADMIT} \Rightarrow \neg \mathrm{Execute}\left( a_{t} \right).$$

This is one of the primary v5 invariants.

A blocked action must remain an unexecuted proposal.

## 3.4 Execution as a Governed Boundary

Once admitted, the action crosses from internal computation into execution:

$$e_{t}\mathcal{= E}\left( a_{t} \right),$$

where $\mathcal{E}$ denotes the execution mechanism and $e_{t}$ is the resulting execution record.

This boundary is important because external side effects cannot always be reversed. U.F.O. should therefore distinguish between the **authorization to attempt an action** and the **evidence describing what occurred after the attempt**.

Execution may result in several states:

$$S_{t}^{E} \in \lbrace \text{NOT\_EXECUTED},\text{EXECUTED},\text{FAILED},\text{OUTCOME\_UNKNOWN}\rbrace .$$

`NOT_EXECUTED` indicates that the action never crossed the execution boundary.

EXECUTED indicates that the tool or external mechanism reports completion.

FAILED indicates an observed execution failure.

`OUTCOME_UNKNOWN` represents an important third condition: the system cannot reliably determine whether the external side effect occurred.

This last state prevents a common agentic failure mode. If a consequential external action is transmitted but confirmation is lost, absence of confirmation must not automatically be interpreted as absence of effect.

Therefore,

$$\boxed{\text{OUTCOME\_UNKNOWN} \neq \text{FAILED}}$$

and particularly for non-idempotent or irreversible actions,

$$\text{OUTCOME\_UNKNOWN} \nRightarrow \text{automatic retry}.$$

This uncertainty must instead become evidence for verification, reconciliation, and possibly escalation.

## 3.5 Observation and Outcome Verification

An execution record is not yet proof that the intended result exists in the world.

The system therefore derives or receives an observation:

$$o_{t}\mathcal{= O}\left( e_{t},X_{t} \right),$$

where $o_{t}$ represents available evidence about the state resulting from execution.

The observation is then compared against the expected postconditions associated with the proposed action:

$$q_{t}\mathcal{= V}\left( a_{t},o_{t} \right),$$

where $\mathcal{V}$ is the verification operator and $q_{t}$ is the verification result.

This establishes one of the central distinctions of the paper:

$$\boxed{\text{Execution Success} \neq \text{Outcome Verification}}$$

For example, a tool may report that an API request completed successfully while the requested remote state remains unchanged. A file-writing function may return without error while the expected artifact is missing or malformed. A search operation may execute correctly while returning evidence insufficient to satisfy the goal.

The tool has only authority to report what occurred at its own execution boundary.

It does not possess authority to declare that the larger intent has been satisfied.

That determination belongs to verification.

## 3.6 Agentic Residual Formation

Verification produces evidence about the difference between expected and observed behavior.

Paper XII represents this difference through an **Agentic Residual Vector**:

$$\rho_{t}\mathcal{= R}\left( a_{t},e_{t},o_{t},q_{t},X_{t} \right).$$

The residual is not limited to binary success or failure. It records the relevant discrepancies remaining after execution.

A general representation is

$$\rho_{t} = \left( \rho_{g},\rho_{c},\rho_{p},\rho_{s},\rho_{u} \right)_{t},$$

where

$$\rho_{g} = \text{goal or postcondition discrepancy},$$

$$\rho_{c} = \text{predicted-versus-observed resource discrepancy},$$

$$\rho_{p} = \text{policy or authority discrepancy},$$

$$\rho_{s} = \text{system stability or membrane-state discrepancy},$$

and

$$\rho_{u} = \text{execution uncertainty or unresolved evidence}.$$

The precise numerical representation of these terms is an implementation decision addressed later in the paper. The architectural requirement is that unresolved discrepancies remain explicit rather than being collapsed into a generic success flag.

Residual evidence therefore answers:

What remains unexplained, unsatisfied, unstable, or uncertain after this action?

This evidence drives the next stage.

## 3.7 Governed Reflection and Candidate State

Reflection consumes the residual together with the existing system state:

$$\Delta X_{t}^{\ast}\mathcal{= F}\left( \rho_{t},X_{t} \right),$$

where $\mathcal{F}$ denotes the reflection operator and $\Delta X_{t}^{\ast}$ is a proposed corrective state change.

The asterisk indicates that this transition is still a **candidate**.

A candidate next state may then be constructed:

$$X_{t + 1}^{\ast} = X_{t} \oplus \Delta X_{t}^{\ast},$$

where $\oplus$ represents the applicable state-composition operation.

The candidate must be validated before becoming authoritative:

$$X_{t + 1} = \left\lbrace  \begin{array}{l}
X_{t + 1}^{\ast}, \quad \mathrm{Valid}\left( X_{t + 1}^{\ast} \right) = 1, \cr
X_{t}, \quad \mathrm{Valid}\left( X_{t + 1}^{\ast} \right) = 0.
\end{array} \right.$$

This extends BEDROCK's atomic update discipline directly into the agentic loop.

Reflection may recommend adaptation.

Reflection may not bypass validation.

A failed reflective transition therefore preserves the prior authoritative state.

## 3.8 Evidence Qualification and Memory

The execution cycle may produce information useful beyond the current step, but not all observed information should immediately become durable semantic memory.

Let

$$J_{t}$$

represent the provisional evidence generated during the current cycle.

The memory path is then modeled as

$$J_{t} \rightarrow T_{t} \rightarrow A_{M,t} \rightarrow P_{M,t} \rightarrow \mathcal{M}_{t + 1},$$

where $T_{t}$ represents temporal qualification, $A_{M,t}$ represents memory admissibility, $P_{M,t}$ represents the applicable promotion decision, and $\mathcal{M}_{t + 1}$ is the resulting durable memory state.

A failed qualification or promotion leaves durable memory unchanged:

$$P_{M,t} \neq \text{PROMOTE} \Rightarrow \mathcal{M}_{t + 1} = \mathcal{M}_{t}.$$

This preserves the distinction established in Section 2:

$$\boxed{\text{Experience} \neq \text{Memory}}$$

and introduces an additional agentic rule:

$$\boxed{\text{Unverified evidence cannot become authoritative merely because it was observed.}}$$

The existing temporal-governance, policy, residual, and SAO mechanisms provide the foundation for implementing this path.

## 3.9 Agentic Closure

After execution, verification, residual formation, reflection, and applicable evidence handling have completed, the system reaches the final decision of the cycle:

$$D_{t} = \mathcal{C}_{A}\left( I,X_{t + 1},\rho_{t} \right),$$

where $\mathcal{C}_{A}$ is the **Agentic Closure Operator** and $D_{t}$ is the closure decision.

A representative decision domain is

$$D_{t} \in \lbrace \text{CONTINUE},\text{REPLAN},\text{CONSTRAIN},\text{REFLECT},\text{ESCALATE},\text{HALT\_SUCCESS},\text{HALT\_FAILURE}\rbrace .$$

The important distinction is that closure occurs **after** the consequences of the previous action have been reconciled.

The agent does not continue because another plan step exists.

It continues because the current authoritative state still satisfies the conditions under which another autonomous action may be considered.

Thus,

$$D_{t} = \text{CONTINUE}$$

permits the transition

$$\left( I,X_{t + 1} \right)\overset{\mathcal{P}}{\rightarrow}a_{t + 1}.$$

By contrast,

$$D_{t} = \text{HALT\_SUCCESS}$$

terminates because the intent's success conditions have been sufficiently satisfied, while

$$D_{t} = \text{HALT\_FAILURE}$$

terminates because the system has reached a state from which autonomous execution should not continue under the current contract.

REPLAN, CONSTRAIN, REFLECT, and ESCALATE preserve the intent while changing how the next transition must be approached.

This produces the defining rule of Governed Agentic Closure:

$$\boxed{\text{Completion of an action does not itself authorize another action.}}$$

Continued autonomy is a new decision.

## 3.10 Canonical Cycle

The complete Paper XII execution model can therefore be written as

$$\left( I,X_{t} \right)\overset{\mathcal{P}}{\rightarrow}a_{t}\overset{\mathcal{A}^{-}}{\rightarrow}V_{t}^{-}\underset{\  V_{t}^{-} = \mathrm{ADMIT}\ }{\rightarrow }e_{t}\overset{\mathcal{O}}{\rightarrow}o_{t}\overset{\mathcal{V}}{\rightarrow}q_{t}\overset{\mathcal{R}}{\rightarrow}\rho_{t}\overset{\mathcal{F}}{\rightarrow}X_{t + 1}^{\ast}\overset{\text{Validate/Commit}}{\rightarrow}X_{t + 1}\overset{\mathcal{C}_{A}}{\rightarrow}D_{t}.$$

Memory qualification may occur from verified evidence produced during this sequence, but it does not bypass the final closure decision.

The ordering is intentional.

In particular, the model prohibits the following inversions:

$$\text{execution before admission},$$

$$\text{goal success before verification},$$

$$\text{authoritative mutation before validation},$$

$$\text{durable memory before evidence qualification},$$

and

$$\text{continued autonomy before closure}.$$

These ordering constraints are the mechanism by which previously separate U.F.O. components become a coherent agentic system.

## 3.11 Reference State-Machine View

From an implementation perspective, the same cycle may be represented as a state machine:

```text
INTENT
  │
  ▼
PLANNING
  │
  ▼
PROPOSED
  │
  ▼
PREFLIGHT
  │
  ├──────── BLOCK / ESCALATE ────────► CLOSURE
  │
  ▼
AUTHORIZED
  │
  ▼
EXECUTING
  │
  ▼
OBSERVED
  │
  ▼
VERIFIED
  │
  ▼
RECONCILED
  │
  ▼
REFLECTED
  │
  ▼
EVIDENCE QUALIFIED
  │
  ▼
CLOSURE
  │
  ├──── HALT_SUCCESS
  ├──── HALT_FAILURE
  ├──── ESCALATE
  ├──── CONSTRAIN
  ├──── REPLAN
  └──── CONTINUE ────────────────────► PLANNING
```

Not every implementation must expose each label as a public class or enum. The requirement is semantic: the architecture must preserve the boundaries represented by these states even if several are implemented within the same runtime component.

This distinction prevents the paper from prescribing unnecessary software structure while still prescribing execution behavior.

## 3.12 Engineering Consequence

The v5 agentic loop can therefore be summarized by a simple rule:

**No stage inherits authority merely because the previous stage succeeded.**

A plan must still be admitted.

An admitted action must still be verified after execution.

A verified observation must still qualify before becoming durable memory.

A reflective recommendation must still be validated before changing authoritative state.

A completed cycle must still pass closure before another action begins.

This is the mechanism by which U.F.O. moves from containing agentic capabilities to operating them as one governed system.

Section 4 defines the concrete engineering contracts carried across these boundaries so that the cycle can be implemented and tested without relying on implicit state or ambiguous semantics.

# 4. Engineering Contracts for Governed Agentic Execution

The execution cycle defined in Section 3 depends on explicit contracts between stages. Without those contracts, individual components may interpret the same value differently, infer authority from metadata, mutate state before a final decision exists, or collapse uncertainty into a generic success condition.

U.F.O. v5.0.0 therefore treats the boundaries between planning, governance, execution, verification, reflection, memory, and closure as typed interfaces.

The purpose of these contracts is not to prescribe a single programming language or class hierarchy. Their purpose is to define the minimum information and invariants that must survive each transition.

The principal contracts are:

$$I \rightarrow a_{t} \rightarrow V_{t}^{-} \rightarrow e_{t} \rightarrow q_{t} \rightarrow \rho_{t} \rightarrow D_{t} \rightarrow R_{t}^{A},$$

where $R_{t}^{A}$ denotes the final **Agentic Action Receipt** for cycle $t$.

Each contract is defined below.

## 4.1 Intent Contract

The **Intent Contract** is the authoritative representation of what the agent is attempting to accomplish and the limits under which it may operate.

It is represented as

$$I = (g,\sigma,\kappa,\beta,\alpha,\chi,\eta),$$

where

$$g = \text{goal},$$

$$\sigma = \text{success conditions},$$

$$\kappa = \text{immutable constraints},$$

$$\beta = \text{resource budget},$$

$$\alpha = \text{delegated authority},$$

$$\chi = \text{side-effect ceiling},$$

and

$$\eta = \text{intent provenance and identity}.$$

The intent contract must remain distinguishable from the plan used to satisfy it.

A planner may replace or reorder actions without rewriting the governing contract:

$$P_{t} \neq I.$$

This separation allows replanning while preserving the original objective and authorization boundary.

### Intent invariants

At minimum, the following conditions should hold:

$$g \neq \varnothing,$$

$$\sigma \neq \varnothing,$$

$$\beta \geq 0,$$

and all numerical budget or side-effect quantities must be finite and within their documented domains.

The immutable constraints $\kappa$ must not be weakened by ordinary replanning.

Likewise, delegated authority must not silently expand:

$$\alpha_{t + 1} \subseteq \alpha_{t}$$

unless an explicit authority-bearing event grants additional scope.

This yields an important implementation rule:

$$\boxed{\text{Plan adaptation may change method, but not silently change mandate.}}$$

An Intent Contract should also possess a stable identifier so every downstream action and receipt can be traced back to the contract under which it was proposed.

## 4.2 Action Proposal Contract

The planner transforms intent and current state into an **Action Proposal**:

$$a_{t} = \left( u,\theta,\epsilon,\widehat{c},\widehat{r},\gamma,\pi \right),$$

where

$$u = \text{requested tool or capability},$$

$$\theta = \text{execution parameters},$$

$$\epsilon = \text{expected postconditions},$$

$$\widehat{c} = \text{predicted resource cost},$$

$$\widehat{r} = \text{predicted risk or side-effect exposure},$$

$$\gamma = \text{required capability and authority scope},$$

and

$$\pi = \text{proposal provenance}.$$

The proposal must contain enough information for prospective governance to decide whether execution is admissible.

An action lacking sufficient information for admission is not implicitly safe.

It is incomplete.

Therefore,

$$\mathrm{Incomplete}\left( a_{t} \right) \Rightarrow V_{t}^{-} \neq \text{ADMIT}.$$

### Expected postconditions

The expected postcondition $\epsilon$ is particularly important.

It defines what evidence should exist after successful execution.

Without an expected postcondition, the system can determine only whether a tool returned successfully, not whether the action accomplished what the plan required.

For example:

```text
Action:
    Write artifact A to destination D

Tool-level success:
    write() returned without exception

Expected postcondition:
    artifact A exists at D,
    is readable,
    and matches required content or digest
```

The verifier evaluates the latter.

The executor reports the former.

These responsibilities must remain separate.

## 4.3 Prospective Admission Contract

The proposed action enters the pre-execution governance boundary:

$$V_{t}^{-} = \mathcal{A}^{-}\left( a_{t},I,X_{t} \right).$$

The admission decision must be made using authoritative state and the immutable or currently valid portions of the Intent Contract.

At minimum, the evaluation should consider:

$$C_{\alpha} = \text{authority compatibility},$$

$$C_{\kappa} = \text{constraint compatibility},$$

$$C_{\beta} = \text{budget compatibility},$$

$$C_{\chi} = \text{side-effect compatibility},$$

$$C_{S} = \text{system stability},$$

and

$$C_{P} = \text{policy compatibility}.$$

A simplified admission condition may therefore be expressed as

$$\mathrm{Admit}\left( a_{t} \right) = C_{\alpha} \land C_{\kappa} \land C_{\beta} \land C_{\chi} \land C_{S} \land C_{P}.$$

The implementation may contain more detailed evidence, but no negative result in a mandatory gate may be ignored merely because another score is favorable.

This avoids transforming governance into an unconstrained weighted average.

### Verdict structure

The admission result should contain more than an enum.

A practical verdict should include:

```text
verdict
reason_codes
evaluated_policy
approved_scope
approved_parameters
reserved_budget
side_effect_class
state_version
```

This matters because ADMIT should authorize a **specific action under specific conditions**, not grant general permission to the planner.

If the proposal changes materially after admission, it should be evaluated again.

Thus:

$$a_{t}' \neq a_{t} \Rightarrow \mathcal{A}^{-}\left( a_{t}',I,X_{t} \right)$$

for any change affecting authority, side effects, resources, parameters, or postconditions.

## 4.4 Resource Reservation

A resource budget is only useful prospectively if some portion of it is considered before the action executes.

Let

$$\beta_{t}$$

represent the remaining governed budget at cycle $t$, and let

$${\widehat{c}}_{t}$$

represent the predicted cost of the proposed action.

Before execution,

$${\widehat{c}}_{t} \preccurlyeq \beta_{t}$$

must hold for the resource dimensions governed by the contract.

An admitted action may then reserve the predicted quantity:

$$\beta_{t}^{\mathrm{reserved}} = \beta_{t} - {\widehat{c}}_{t}.$$

After execution, observed cost $c_{t}$ is reconciled against the prediction:

$$\rho_{c,t} = c_{t} - {\widehat{c}}_{t}.$$

The resulting budget becomes

$$\beta_{t + 1} = \beta_{t} - c_{t}$$

subject to the applicable resource policy.

The purpose of reservation is not to claim perfect prediction. It is to prevent a system from authorizing an action only after learning that the action exceeded its budget.

The sequence must remain:

$$\text{estimate} \rightarrow \text{admit/reserve} \rightarrow \text{execute} \rightarrow \text{reconcile}.$$

## 4.5 Execution Contract

An admitted proposal becomes an execution attempt:

$$e_{t}\mathcal{= E}\left( a_{t} \right).$$

The execution record should contain at least:

```text
action_id
tool_or_capability
execution_state
start_marker
completion_marker
reported_success
output
observed_cost
side_effect_class
external_reference
execution_evidence
```

The exact representation may differ between tools, but the execution state must distinguish:

$$\lbrace \text{NOT\_EXECUTED},\text{EXECUTED},\text{FAILED},\text{OUTCOME\_UNKNOWN}\rbrace .$$

The distinction between FAILED and `OUTCOME_UNKNOWN` is mandatory whenever external side effects may exist.

### Side-effect classification

Actions should also be assigned an execution class such as

$$S\left( a_{t} \right) \in \lbrace \text{OBSERVATIONAL},\text{REVERSIBLE},\text{COMPENSATABLE},\text{IRREVERSIBLE}\rbrace .$$

An observational action is intended to read or inspect state.

A reversible action has a defined inverse or restoration mechanism.

A compensatable action may not be directly reversible but has a separate corrective operation.

An irreversible action has no reliable automatic rollback within the model.

The classification affects retry and escalation policy.

In particular:

$$S\left( a_{t} \right) = \text{IRREVERSIBLE} \land S_{t}^{E} = \text{OUTCOME\_UNKNOWN}$$

must not automatically transition back to execution.

Instead,

$$\text{OUTCOME\_UNKNOWN} \rightarrow \text{VERIFY or ESCALATE}.$$

This prevents uncertain side effects from being duplicated by naïve retry logic.

## 4.6 Verification Contract

The verification layer receives the proposal, execution record, and post-execution observation:

$$q_{t}\mathcal{= V}\left( a_{t},e_{t},o_{t} \right).$$

Its job is to determine whether the expected postconditions $\epsilon_{t}$ are supported by observed evidence.

A useful verification result may contain:

```text
status
verified_postconditions
failed_postconditions
unresolved_postconditions
evidence_refs
confidence_or_quality
verification_method
```

The status should distinguish at least:

$$Q_{t} \in \lbrace \text{VERIFIED},\text{PARTIAL},\text{FAILED},\text{UNKNOWN}\rbrace .$$

VERIFIED indicates that the required postconditions are satisfied by available evidence.

PARTIAL indicates that only part of the expected result was established.

FAILED indicates evidence contradicting the required postcondition.

UNKNOWN indicates that available evidence is insufficient to establish either success or failure.

This prevents binary execution status from standing in for evidence quality.

The governing relationship is therefore

$$\mathrm{ToolSuccess}\left( e_{t} \right) \nRightarrow Q_{t} = \text{VERIFIED}.$$

Likewise,

$$Q_{t} = \text{UNKNOWN} \nRightarrow Q_{t} = \text{FAILED}.$$

Uncertainty remains explicit.

## 4.7 Agentic Residual Contract

The Agentic Residual Vector converts discrepancies from execution and verification into structured evidence:

$$\rho_{t} = \left( \rho_{g},\rho_{c},\rho_{p},\rho_{s},\rho_{u} \right)_{t}.$$

The components represent:

$$\rho_{g} = \text{goal/postcondition residual},$$

$$\rho_{c} = \text{resource residual},$$

$$\rho_{p} = \text{policy or authority residual},$$

$$\rho_{s} = \text{stability residual},$$

and

$$\rho_{u} = \text{uncertainty residual}.$$

Paper XII does not require every component to use the same numerical scale.

That would incorrectly imply that goal discrepancy, computation cost, policy deviation, membrane instability, and uncertainty are naturally commensurate quantities.

Instead, the contract requires that each component have:

- a defined domain,

- defined interpretation,

- explicit invalid-state handling,

- and a documented effect on reflection or closure.

This preserves a lesson established during BEDROCK:

$$\boxed{\text{Different evidence classes may interact without becoming the same quantity.}}$$

A future implementation may normalize selected terms for a specific operator, but such normalization must be explicit.

## 4.8 Governed Reflection Contract

Reflection receives

$$\left( \rho_{t},X_{t} \right)$$

and produces

$$\Delta X_{t}^{\ast}.$$

The reflection contract should identify:

```text
trigger
input_residual
proposed_changes
affected_state_domains
expected_effect
validation_requirements
provenance
```

Critically, reflection output is not itself authoritative.

The model therefore distinguishes:

$$\text{reflection proposal}$$

from

$$\text{reflection commit}.$$

The resulting transition must preserve atomicity:

$$X_{t} \rightarrow X_{t + 1}^{\ast} \rightarrow \mathrm{Validate} \rightarrow X_{t + 1}.$$

If validation fails, then

$$X_{t + 1} = X_{t}.$$

No valid implementation may leave an authoritative state that is partly old and partly derived from a rejected reflection.

This mirrors the transaction discipline established for membrane and boundary updates in BEDROCK v4.

## 4.9 Memory Qualification Contract

Verified or otherwise admissible execution evidence may become a candidate for durable memory, but it must first pass the memory-governance path.

A provisional evidence object should preserve:

```text
source_action
observation
verification_status
residual_state
timestamp_or_step
origin
policy_context
temporal_context
```

It then moves through

$$J_{t} \rightarrow T_{t} \rightarrow A_{M,t} \rightarrow P_{M,t}.$$

Only a final authorized promotion may modify durable memory:

$$P_{M,t} = \text{PROMOTE} \Rightarrow \mathcal{M}_{t + 1} = \mathrm{Commit}\left( J_{t} \right).$$

Otherwise,

$$\mathcal{M}_{t + 1} = \mathcal{M}_{t}.$$

The memory contract must therefore preserve the same final-verdict ordering established by SAO:

$$\text{evaluate} \rightarrow \text{final promotion verdict} \rightarrow \text{commit}.$$

The existence of an observation is not itself permission to preserve that observation indefinitely.

## 4.10 Closure Decision Contract

The final contract of the cycle is the closure decision:

$$D_{t} = \mathcal{C}_{A}\left( I,X_{t + 1},\rho_{t},q_{t} \right).$$

The closure operator evaluates whether another autonomous action remains permissible under the original intent and the newly reconciled state.

The decision domain is

$$D_{t} \in \lbrace \text{CONTINUE},\text{REPLAN},\text{CONSTRAIN},\text{REFLECT},\text{ESCALATE},\text{HALT\_SUCCESS},\text{HALT\_FAILURE}\rbrace .$$

A closure result should contain at least:

```text
decision
reason_codes
intent_status
remaining_budget
authority_status
verification_status
stability_status
unresolved_residuals
next_allowed_transition
```

This record makes continuation explainable.

### Success closure

A successful termination requires satisfaction of the Intent Contract's success conditions:

$$\sigma(I) = 1 \Rightarrow D_{t} = \text{HALT\_SUCCESS}$$

provided no mandatory unresolved condition prohibits success.

Tool completion alone is insufficient.

### Failure closure

Failure may occur when:

- the goal becomes infeasible under remaining authority,

- budget is exhausted,

- mandatory constraints cannot be satisfied,

- system stability cannot be restored,

- required verification repeatedly fails,

- or another policy boundary prohibits continuation.

### Continued execution

CONTINUE does not carry authorization for an arbitrary future action.

It permits the planner to propose the next action:

$$D_{t} = \text{CONTINUE} \Rightarrow a_{t + 1}\mathcal{= P}\left( I,X_{t + 1} \right).$$

That new proposal must again pass prospective admission.

Thus closure grants **permission to consider another action**, not permission to execute one without governance.

## 4.11 Agentic Action Receipt

The final engineering object produced by each complete cycle is the **Agentic Action Receipt**:

$$R_{t}^{A}.$$

The receipt binds the major decisions and evidence of one agentic transition into a single provenance record.

A conceptual representation is

$$R_{t}^{A} = \left( \eta,r_{P},a_{t},V_{t}^{-},e_{t},q_{t},\rho_{t},\Delta X_{t},P_{M,t},D_{t} \right),$$

where

$$\eta = \text{intent identifier},$$

$$r_{P} = \text{plan revision identifier},$$

$$a_{t} = \text{action proposal},$$

$$V_{t}^{-} = \text{prospective admission verdict},$$

$$e_{t} = \text{execution record},$$

$$q_{t} = \text{verification result},$$

$$\rho_{t} = \text{agentic residual},$$

$$\Delta X_{t} = \text{committed or rejected state transition},$$

$$P_{M,t} = \text{memory promotion result},$$

and

$$D_{t} = \text{closure decision}.$$

The Action Receipt provides the evidence required to answer:

Why was this action proposed?

Why was it allowed or blocked?

What actually executed?

What evidence was observed?

Was the desired outcome verified?

What internal state changed?

What information became durable?

Why did the system continue or stop?

This receipt is an audit and replay structure.

It does not itself authorize any action.

That distinction follows the same BEDROCK principle applied throughout this paper:

$$\boxed{\text{Evidence describing authority} \neq \text{authority itself}.}$$

## 4.12 Contract Composition

The complete contract chain can now be represented as

$$I\overset{\mathcal{P}}{\rightarrow}a_{t}\overset{\mathcal{A}^{-}}{\rightarrow}V_{t}^{-}\overset{\mathcal{E}}{\rightarrow}e_{t}\overset{\mathcal{V}}{\rightarrow}q_{t}\overset{\mathcal{R}}{\rightarrow}\rho_{t}\overset{\mathcal{F}}{\rightarrow}X_{t + 1}\overset{\mathcal{C}_{A}}{\rightarrow}D_{t}\rightarrow R_{t}^{A}.$$

Each arrow represents a semantic boundary.

No downstream component should infer information that an upstream contract did not explicitly provide.

Likewise, no stage should acquire authority merely because it received data produced by a trusted subsystem.

This produces a concise set of v5 contract rules:

$$\boxed{\begin{array}{l}
\text{Intent} \neq \text{Plan}, \cr
\text{Plan} \neq \text{Authorization}, \cr
\text{Authorization} \neq \text{Execution Result}, \cr
\text{Execution Result} \neq \text{Verified Outcome}, \cr
\text{Observation} \neq \text{Durable Memory}, \cr
\text{Reflection Proposal} \neq \text{Committed State}, \cr
\text{Action Completion} \neq \text{Continued Authority}.
\end{array}}$$

These contracts provide the implementation vocabulary required for U.F.O. v5.0.0.

**Section 5 converts the lessons of BEDROCK into explicit cross-cutting invariants that every implementation of this execution model must preserve.**

# 5. BEDROCK-Derived Agentic Invariants

The Governed Agentic Closure model is built upon engineering principles exposed and strengthened during the U.F.O. v4 BEDROCK hardening process.

BEDROCK repeatedly demonstrated that failures in governed systems do not necessarily originate inside individual components. They often appear at the **seams between otherwise reasonable components**: metadata is interpreted as evidence, preliminary decisions become authoritative too early, partial state mutations survive rejected operations, diagnostic values become implicit permissions, or malformed numerical inputs cross a trust boundary without being rejected.

Paper XII extends those lessons from individual U.F.O. mechanisms to the complete agentic execution cycle.

The result is a set of cross-cutting invariants that every conforming v5 execution path must preserve.

These invariants are independent of planner sophistication, tool capability, model architecture, or task complexity. They remain applicable whether the proposed action originates from deterministic logic, a language model, another agent, a human instruction, or a swarm-selected executor.

They therefore represent the **governance laws of the agentic cycle**, rather than implementation details of any single module.

## 5.1 From BEDROCK State Integrity to Agentic Integrity

The recurring BEDROCK pattern can be written as:

$$\text{candidate} \rightarrow \text{validate} \rightarrow \text{final verdict} \rightarrow \text{commit}.$$

A candidate may be useful.

A validator may initially produce favorable evidence.

A subsystem may report success.

None of those events independently establishes an authoritative transition.

Paper XII generalizes the same discipline to agency:

$$\text{propose} \rightarrow \text{admit} \rightarrow \text{execute} \rightarrow \text{verify} \rightarrow \text{reconcile} \rightarrow \text{close}.$$

The difference is primarily scale.

BEDROCK applied transaction discipline to numerical inputs, membrane state, boundary geometry, temporal evidence, shard admission, and ascension ordering.

Governed Agentic Closure applies the same discipline to actions.

The central requirement becomes:

$$\boxed{\text{Every consequential transition must cross the authority boundary appropriate to that transition.}}$$

No component receives universal authority merely because it is trusted for one narrower responsibility.

## 5.2 Invariant Set

For a governed agentic trajectory, define the mandatory invariant set

$$\mathcal{I =}\left( I_{A},I_{B},I_{S},I_{V},I_{M},I_{R},I_{C},I_{P} \right),$$

where

$$I_{A} = \text{authority integrity},$$

$$I_{B} = \text{resource and budget integrity},$$

$$I_{S} = \text{authoritative-state atomicity},$$

$$I_{V} = \text{verification integrity},$$

$$I_{M} = \text{memory integrity},$$

$$I_{R} = \text{execution and retry integrity},$$

$$I_{C} = \text{closure integrity},$$

and

$$I_{P} = \text{provenance and evidence integrity}.$$

For a conforming execution cycle,

$$I_{j} = 1$$

for every applicable mandatory invariant.

Task success does not compensate for an invariant violation.

Thus,

$$\text{GoalSatisfied} = 1$$

does not imply

$$\text{GovernanceCorrect} = 1.$$

An action may produce the desired external result and still constitute an invalid U.F.O. transition.

This distinction is foundational to the v5 validation model.

## 5.3 Invariant I — Proposal Does Not Confer Authority

The first invariant separates computational recommendation from permission.

For any proposed action

$$a_{t}\mathcal{= P}\left( I,X_{t} \right),$$

the existence of the proposal does not imply authorization:

$$a_{t} \nRightarrow \mathrm{Authorized}\left( a_{t} \right).$$

Execution requires an independent prospective admission verdict:

$$V_{t}^{-} = \mathcal{A}^{-}\left( a_{t},I,X_{t} \right).$$

Only

$$V_{t}^{-} = \text{ADMIT}$$

may permit the corresponding execution attempt.

Therefore,

$$V_{t}^{-} \neq \text{ADMIT} \Rightarrow S_{t}^{E} = \text{NOT\_EXECUTED}.$$

This rule applies regardless of how the proposal was produced.

A highly capable planner does not gain authority by being highly capable.

A human-selected plan does not bypass admission.

An LLM-generated plan does not self-authorize.

A deterministic planner does not self-authorize.

The origin of the proposal does not determine its governing status.

The invariant is:

$$\boxed{\text{Proposal} \neq \text{Authorization}.}$$

## 5.4 Invariant II — Selection Does Not Confer Authority

The same principle applies to multi-agent coordination.

Suppose a swarm auction or task-allocation mechanism selects agent $U_{i}$:

$$\mathrm{Select}(K) = U_{i}.$$

This establishes a preferred executor.

It does not establish that every action proposed by $U_{i}$ is admissible.

Therefore,

$$\mathrm{Select}\left( U_{i} \right) \nRightarrow \mathrm{Authorize}\left( a_{t} \right).$$

Each proposed action must remain within the delegated authority envelope

$$\alpha_{i}$$

and must independently pass prospective admission.

Furthermore, ordinary replanning cannot silently enlarge delegated authority:

$$\alpha_{t + 1} \subseteq \alpha_{t}$$

unless an explicit authority-bearing event changes the contract.

Thus:

$$\boxed{\text{Delegation may distribute authority; planning may not manufacture it.}}$$

This prevents task assignment, winning bids, planner confidence, or agent specialization from becoming implicit authorization.

## 5.5 Invariant III — Metadata Is Not Evidence

One of the clearest BEDROCK lessons emerged from the distinction between routing metadata and admissibility evidence.

A statement describing what a subsystem attempted or selected does not necessarily prove that the underlying governing condition was satisfied.

Paper XII generalizes this rule.

Examples include:

$$\text{routed=True} \nRightarrow \text{admissible},$$

$$\text{selected=True} \nRightarrow \text{authorized},$$

$$\text{tool\_success = True} \nRightarrow \text{verified},$$

and

$$\text{completed=True} \nRightarrow \text{closure satisfied}.$$

Metadata may describe process state.

Evidence supports a governing claim.

The two may coexist in the same receipt, but they must not be substituted for one another.

The invariant is therefore:

$$\boxed{\text{Descriptive state does not automatically constitute governing evidence.}}$$

## 5.6 Invariant IV — Final Verdict Precedes Authoritative Commit

BEDROCK III exposed the danger of committing promotion evidence before all mandatory governance stages had completed.

Paper XII adopts the resulting rule globally:

$$\boxed{\text{Final governing verdict} \rightarrow \text{authoritative commit}.}$$

Never:

$$\boxed{\text{preliminary favorable verdict} \rightarrow \text{commit} \rightarrow \text{later governance}.}$$

This applies to:

- action authorization;

- reflection;

- budget reconciliation;

- memory promotion;

- plan revision where authoritative state is affected;

- delegated authority;

- closure;

- and final Action Receipt semantics.

An intermediate favorable result may be retained as diagnostic evidence.

It must not survive a later mandatory rejection as if the event had been authorized or completed.

Formally, if

$$v_{t}^{(1)} = \text{favorable}$$

but a later mandatory gate produces

$$v_{t}^{(n)} = \text{REJECT},$$

then authoritative state must correspond to

$$v_{t}^{(n)},$$

not

$$v_{t}^{(1)}.$$

This is the agentic extension of the SAO final-verdict ordering correction.

## 5.7 Invariant V — Rejected State Does Not Partially Commit

Any internal state transition that may affect future agent behavior must obey atomic candidate–validate–commit semantics.

Let

$$X_{t + 1}^{\ast}$$

represent a candidate next state.

Then

$$X_{t + 1} = \left\lbrace  \begin{array}{l}
X_{t + 1}^{\ast}, \quad \mathrm{Valid}\left( X_{t + 1}^{\ast} \right) = 1, \cr
X_{t}, \quad \mathrm{Valid}\left( X_{t + 1}^{\ast} \right) = 0.
\end{array} \right.$$

The failed case must preserve the previous authoritative state.

Therefore,

$$\mathrm{Rejected}\left( X_{t + 1}^{\ast} \right) \Rightarrow X_{t + 1} = X_{t}.$$

There must be no intermediate result such as:

$$X_{t + 1} = X_{t} + \delta_{1} + \delta_{2}$$

when

$$\delta_{3}$$

caused the complete candidate transition to fail.

This invariant is particularly important for reflection because reflective updates may eventually originate from probabilistic, heuristic, or model-generated reasoning.

Reflection may propose adaptation.

It may not receive privileged mutation semantics.

The invariant is:

$$\boxed{\text{Rejected transition} \Rightarrow \text{no partial authoritative mutation}.}$$

## 5.8 Invariant VI — Tool Success Does Not Establish Outcome Truth

The executor has authority to report execution.

It does not have universal authority to declare that the intended state of the world now exists.

Thus,

$$e_{t}.\text{success} = 1$$

does not imply

$$q_{t} = \text{VERIFIED}.$$

The relationship must remain:

$$\text{Execution} \rightarrow \text{Observation} \rightarrow \text{Verification}.$$

A tool may execute correctly while:

- writing incorrect content;

- modifying the wrong external object;

- receiving an accepted request that is never applied;

- returning incomplete search evidence;

- producing a syntactically valid but semantically incorrect artifact;

- or completing only part of the required postcondition.

Therefore:

$$\boxed{\text{Tool Success} \neq \text{Verified Outcome}.}$$

This is not merely an error-handling rule.

It defines which subsystem has authority to establish goal-relevant truth.

## 5.9 Invariant VII — Experience Does Not Automatically Become Memory

An observation produced by execution begins as evidence of an event.

It is not automatically an authoritative long-term belief.

Let

$$o_{t}$$

be the immediate observation and

$$J_{t}$$

its provisional evidence representation.

Then durable memory requires an explicit qualification path:

$$o_{t} \rightarrow J_{t} \rightarrow T_{t} \rightarrow A_{M,t} \rightarrow P_{M,t} \rightarrow \mathcal{M}_{t + 1}.$$

If the final promotion decision is not positive,

$$P_{M,t} \neq \text{PROMOTE},$$

then

$$\mathcal{M}_{t + 1} = \mathcal{M}_{t}.$$

A failed, unknown, transient, policy-rejected, or otherwise insufficiently qualified observation must not silently become durable fact.

The invariant is:

$$\boxed{\text{Experience} \neq \text{Memory}.}$$

This preserves the distinction between what happened during one execution cycle and what the system is permitted to carry forward as governed knowledge.

## 5.10 Invariant VIII — Unknown Outcome Is Not Equivalent to Failure

External execution introduces a condition that internal state transitions do not always face: the system may lose certainty about whether a side effect occurred.

Accordingly,

$$\text{OUTCOME\_UNKNOWN} \neq \text{FAILED}.$$

This distinction must remain explicit.

If an action is irreversible or non-idempotent and its execution state becomes unknown,

$$S\left( a_{t} \right) = \text{IRREVERSIBLE}$$

and

$$S_{t}^{E} = \text{OUTCOME\_UNKNOWN},$$

then the runtime must not infer

$$\mathrm{SafeToRetry}\left( a_{t} \right) = 1.$$

Instead, the unresolved state should flow into verification, residual formation, closure, and possibly escalation.

Thus:

$$\boxed{\text{Unknown irreversible outcome} \neq \text{safe retry}.}$$

The system must preserve uncertainty rather than collapse it merely to keep the plan moving.

## 5.11 Invariant IX — Diagnostics Do Not Become Authorization Gates Unless Declared

BEDROCK clarified that several U.F.O. quantities are valuable diagnostics without automatically functioning as universal authorization mechanisms.

The same distinction must remain in v5.

Examples may include:

- Holistic Governor metrics;

- mesh-coherence measurements;

- membrane tension;

- curvature;

- residual magnitude;

- heuristic confidence;

- planner ranking;

- swarm score.

These values may inform a governing decision.

They do not independently become governing decisions unless the applicable policy explicitly defines them as such.

Formally, for diagnostic quantity $d_{t}$,

$$d_{t} \nRightarrow V_{t}^{-}.$$

Instead,

$$V_{t}^{-} = \mathcal{A}^{-}\left( a_{t},I,X_{t},d_{t},\ldots \right)$$

when that diagnostic is an explicitly defined admission input.

This distinction prevents useful measurements from accumulating accidental authority as the architecture grows.

The invariant is:

$$\boxed{\text{Diagnostic evidence} \neq \text{authorization unless the contract says so}.}$$

## 5.12 Invariant X — Resource Limits Must Be Evaluated Before Consequential Use

A bounded system cannot treat resource constraints only as retrospective accounting.

Let

$$\beta_{t}$$

denote available governed budget and

$${\widehat{c}}_{t}$$

the predicted resource requirement of the proposed action.

For every governed resource dimension, prospective admission must establish the applicable relation before execution, such as

$${\widehat{c}}_{t} \preccurlyeq \beta_{t}.$$

Observed cost

$$c_{t}$$

is then reconciled afterward.

Thus the required order is:

$$\text{predict} \rightarrow \text{check} \rightarrow \text{reserve} \rightarrow \text{execute} \rightarrow \text{reconcile}.$$

Not:

$$\text{execute} \rightarrow \text{discover budget violation}.$$

Prediction may be imperfect.

That imperfection becomes residual evidence:

$$\rho_{c,t} = c_{t} - {\widehat{c}}_{t}.$$

It does not eliminate the need for prospective evaluation.

The invariant is:

$$\boxed{\text{Bounded execution requires prospective resource governance}.}$$

## 5.13 Invariant XI — Continued Autonomy Requires Renewed Closure

The defining invariant of Paper XII concerns recurrence.

An action finishing does not grant permission for the next action.

A plan containing another step does not grant permission for the next action.

A previous successful admission does not grant permission for the next action.

A previous successful closure does not itself grant execution permission for the next action.

Instead,

$$D_{t} = \mathcal{C}_{A}\left( I,X_{t + 1},\rho_{t},q_{t} \right)$$

determines whether another governed cycle may begin.

If

$$D_{t} = \text{CONTINUE},$$

then

$$a_{t + 1}\mathcal{= P}\left( I,X_{t + 1} \right)$$

may be proposed.

That proposal still requires:

$$V_{t + 1}^{-} = \mathcal{A}^{-}\left( a_{t + 1},I,X_{t + 1} \right).$$

Therefore:

$$\boxed{\text{Closure permits renewed consideration; admission permits execution}.}$$

This distinction prevents authority from propagating indefinitely through an autonomous loop.

The broader principle is:

$$\boxed{\text{Continued autonomy is not assumed; it is re-earned after every governed action cycle.}}$$

## 5.14 Invariant XII — Provenance Must Follow the Same Action Through the Complete Cycle

A governed system must be able to establish that the action authorized is the action executed, that the evidence verified belongs to that execution, and that the closure decision was derived from the corresponding reconciled state.

For cycle $t$, the provenance relation must therefore remain coherent across

$$I \rightarrow P_{t} \rightarrow a_{t} \rightarrow V_{t}^{-} \rightarrow e_{t} \rightarrow o_{t} \rightarrow q_{t} \rightarrow \rho_{t} \rightarrow X_{t + 1} \rightarrow D_{t} \rightarrow R_{t}^{A}.$$

An admission verdict for action $a_{t}$ must not silently authorize

$$a_{t}' \neq a_{t}.$$

Evidence from execution $e_{t}$ must not be attached to an unrelated proposal.

A verification result must refer to the expected postconditions that were actually approved.

A closure decision must refer to the state resulting from the same cycle.

This can be expressed as:

$$\boxed{\text{Authority and evidence must preserve identity across the transaction}.}$$

The Action Receipt introduced in Section 4 exists partly to make this invariant inspectable.

## 5.15 Fail-Closed Treatment of Invalid Evidence

All governing boundaries must preserve the numeric and structural hardening established during BEDROCK.

Inputs used for authority, budget, residuals, stability, verification, memory, or closure must reject values outside their documented domains.

This includes values such as:

$$\mathrm{NaN},\quad + \infty,\quad - \infty,$$

as well as malformed types, boolean values masquerading as numerics, impossible negative quantities, incorrect vector dimensions, and values whose conversion exceeds the supported numerical representation.

The governing rule is not that every malformed value must produce the same error response.

The rule is:

$$\boxed{\text{Invalid evidence must never increase authority}.}$$

Depending on the boundary, invalid input may result in:

- rejection;

- quarantine;

- escalation;

- preservation of prior state;

- or explicit failure.

It must not silently become favorable governing evidence.

## 5.16 Invariants Across Replanning

Replanning introduces a special composition risk because it allows the system to alter its execution strategy after failure.

Suppose

$$P_{t}^{(k)}$$

is rejected or becomes ineffective.

The system may construct

$$P_{t}^{(k + 1)}.$$

The new plan may alter:

- tool selection;

- execution order;

- parameterization;

- decomposition;

- resource distribution;

- or recovery strategy.

It must not silently alter the authoritative Intent Contract.

Thus, absent an explicit external update,

$$g^{(k + 1)} = g^{(k)},$$

$$\kappa^{(k + 1)} = \kappa^{(k)},$$

and

$$\alpha^{(k + 1)} \subseteq \alpha^{(k)}.$$

This produces another useful law:

$$\boxed{\text{Failure may justify a new method; it does not justify a new mandate}.}$$

Repeated inability to satisfy a goal must eventually result in constraint, escalation, or termination rather than self-generated expansion of authority.

## 5.17 Invariant Composition

These invariants are not independent decorations around the execution cycle.

They compose.

For example:

$$I_{A} = 1$$

may establish that an action was properly authorized, while

$$I_{V} = 0$$

may establish that its result was never verified.

The action may therefore have been legitimate to attempt but insufficient to justify successful closure.

Similarly,

$$I_{V} = 1$$

does not imply

$$I_{M} = 1.$$

A verified observation may still fail memory policy or temporal qualification.

Likewise,

$$I_{S} = 1$$

does not imply

$$I_{C} = 1.$$

A perfectly atomic state update may still leave the system outside the conditions permitting another autonomous action.

Governance correctness therefore requires the applicable invariants to hold together:

$${\mathrm{GovernedCycle}}_{t} = \bigwedge_{j \in J_{t}}I_{j},$$

where $J_{t}$ is the set of mandatory invariants applicable to cycle $t$.

This avoids a common architectural mistake in which one successful control mechanism is treated as evidence of overall safety or correctness.

No individual invariant substitutes for the others.

## 5.18 The BEDROCK Agentic Laws

The cross-cutting requirements of this section can be condensed into the following engineering laws:

$$\boxed{\begin{array}{l}
\text{Proposal} \neq \text{Authorization}, \cr
\text{Selection} \neq \text{Authorization}, \cr
\text{Metadata} \neq \text{Evidence}, \cr
\text{Preliminary Verdict} \neq \text{Final Verdict}, \cr
\text{Candidate State} \neq \text{Authoritative State}, \cr
\text{Tool Success} \neq \text{Verified Outcome}, \cr
\text{Experience} \neq \text{Durable Memory}, \cr
\text{Unknown Outcome} \neq \text{Failure}, \cr
\text{Diagnostic Signal} \neq \text{Permission}, \cr
\text{Action Completion} \neq \text{Continued Authority}.
\end{array}}$$

Each inequality represents a boundary that the v5 implementation must preserve.

They are intentionally simple.

The complexity arises when many individually reasonable components are allowed to interact without these distinctions.

## 5.19 Falsifiability of the Invariants

These invariants are useful only if an implementation can violate them in a detectable way.

Accordingly, each must map to at least one executable falsification condition.

For example:

$$I_{A} = 0$$

if a blocked action invokes the executor.

$$I_{S} = 0$$

if a rejected reflection leaves any partial authoritative mutation.

$$I_{V} = 0$$

if tool success directly produces verified intent success without postcondition evidence.

$$I_{M} = 0$$

if rejected evidence modifies durable memory.

$$I_{R} = 0$$

if an irreversible unknown outcome triggers an automatic duplicate execution.

$$I_{C} = 0$$

if another consequential action executes without renewed closure and admission.

$$I_{P} = 0$$

if a receipt binds evidence or authority from different action identities.

This transforms the invariants from architectural language into executable obligations.

Section 7's seeded and adversarial validation program is designed specifically to attempt these violations.

## 5.20 Relationship to the Reference State Machine

The invariants determine the legal transitions implemented by the reference state machine in Section 6.

The state machine is therefore not arbitrary orchestration.

Its ordering exists because changing that ordering would permit specific invariant violations.

For example,

$$\text{PROPOSED} \rightarrow \text{EXECUTING}$$

would violate $I_{A}$ if admission were bypassed.

$$\text{EXECUTED} \rightarrow \text{HALT\_SUCCESS}$$

could violate $I_{V}$.

$$\text{OBSERVED} \rightarrow \text{DURABLE MEMORY}$$

could violate $I_{M}$.

$$\text{REFLECTION} \rightarrow \text{partial mutation}$$

would violate $I_{S}$.

And

$$\text{ACTION COMPLETE} \rightarrow \text{NEXT EXECUTION}$$

without closure and new admission would violate $I_{C}$.

Thus, the state machine in Section 6 is the operational expression of the invariants defined here.

## 5.21 Engineering Consequence

The central lesson inherited from BEDROCK is that reliable governance depends less on whether individual components appear reasonable than on whether their boundaries preserve meaning.

A planner can be correct and still lack authority.

An executor can succeed and still fail the goal.

Evidence can be real and still be unsuitable for memory.

Reflection can be useful and still propose an invalid state.

A selected agent can be capable and still lack permission.

A completed action can be successful and still leave the system without authority to continue.

Governed Agentic Closure therefore treats the seams between these mechanisms as first-class engineering objects.

The invariants in this section establish what must remain true across those seams.

They provide the bridge between the contracts defined in Section 4 and the executable state machine defined in Section 6.

**BEDROCK hardened the state beneath the agent. Paper XII extends that discipline to the actions of the agent itself.**

With these invariants defined, the architecture can now be translated into an executable reference state machine whose transitions are not merely ordered, but governed by explicit conditions that can be independently tested and falsified.

# 6. Reference State Machine and Execution Pseudocode

The contracts defined in Section 4 specify the information that must cross each agentic boundary. This section converts those contracts into an executable reference model.

The objective is not to prescribe the internal implementation of every U.F.O. module. Instead, it defines the **minimum ordering and state-transition behavior** required for a v5 agentic runtime to conform to Governed Agentic Closure.

The reference implementation follows three rules established by the v4 BEDROCK work:

1.  authoritative state changes only after validation;

2.  a final governing verdict precedes any corresponding commit;

3.  evidence produced by one stage cannot acquire the authority of another stage merely by crossing a module boundary.

These rules determine the structure of the state machine.

## 6.1 Agentic Cycle State

For implementation purposes, the logical state of one active agentic run may be represented as

$$\Omega_{t} = \left( I,X_{t},P_{t},a_{t},V_{t}^{-},e_{t},o_{t},q_{t},\rho_{t},D_{t} \right),$$

where:

$$I = \text{active Intent Contract},$$

$$X_{t} = \text{authoritative U.F.O. state},$$

$$P_{t} = \text{current plan and plan revision},$$

$$a_{t} = \text{current Action Proposal},$$

$$V_{t}^{-} = \text{prospective admission verdict},$$

$$e_{t} = \text{execution record},$$

$$o_{t} = \text{post-execution observation},$$

$$q_{t} = \text{verification result},$$

$$\rho_{t} = \text{Agentic Residual Vector},$$

and

$$D_{t} = \text{Agentic Closure decision}.$$

Not every field is populated simultaneously. At the beginning of cycle $t$, later-stage fields are empty because their corresponding events have not yet occurred.

This is preferable to populating fields with optimistic defaults. For example, an action that has not been verified should not begin with

```python
verified = True
```

merely because no verification failure has yet occurred.

Absence of negative evidence is not positive evidence.

## 6.2 Canonical Runtime States

A conforming runtime may expose the following logical states:

$$\begin{matrix}
S_{t} \in \lbrace \text{READY},\text{PLANNING},\text{PROPOSED},\text{PREFLIGHT}, \cr
\text{AUTHORIZED},\text{EXECUTING},\text{OBSERVING},\text{VERIFYING}, \cr
\text{RECONCILING},\text{REFLECTING},\text{QUALIFYING\_MEMORY}, \cr
\text{CLOSURE},\text{HALTED},\text{ESCALATED}\rbrace .
\end{matrix}$$

These labels describe semantic stages. An implementation does not need one class for every state, but it must preserve their ordering constraints.

The nominal execution path is

$$\begin{matrix}
\text{READY} \rightarrow \text{PLANNING} \rightarrow \text{PROPOSED} \rightarrow \text{PREFLIGHT} \cr
 \rightarrow \text{AUTHORIZED} \rightarrow \text{EXECUTING} \rightarrow \text{OBSERVING} \cr
 \rightarrow \text{VERIFYING} \rightarrow \text{RECONCILING} \rightarrow \text{REFLECTING} \cr
 \rightarrow \text{QUALIFYING\_MEMORY} \rightarrow \text{CLOSURE}.
\end{matrix}$$

Closure may then transition to a new cycle or terminate the run.

## 6.3 State-Transition Constraints

Several transitions are deliberately prohibited.

A runtime must not permit

$$\text{PROPOSED} \rightarrow \text{EXECUTING}$$

without passing through prospective admission.

Likewise,

$$\text{EXECUTING} \rightarrow \text{HALT\_SUCCESS}$$

must not occur solely because the executor reports success. Verification and closure still have responsibility for determining whether the Intent Contract has actually been satisfied.

A direct transition

$$\text{OBSERVING} \rightarrow \text{DURABLE\_MEMORY}$$

is also prohibited because observations require qualification before becoming persistent state.

Similarly,

$$\text{REFLECTING} \rightarrow X_{t + 1}$$

is valid only after candidate-state validation.

These constraints can be summarized as

$$\boxed{\begin{array}{l}
\text{Proposal} \quad \rightarrow \text{Admission} \rightarrow \text{Execution}, \cr
\text{Execution} \quad \rightarrow \text{Observation} \rightarrow \text{Verification}, \cr
\text{Residual} \quad \rightarrow \text{Candidate State} \rightarrow \text{Validation} \rightarrow \text{Commit}, \cr
\text{Evidence} \quad \rightarrow \text{Qualification} \rightarrow \text{Promotion} \rightarrow \text{Memory}, \cr
\text{Reconciliation} \quad \rightarrow \text{Closure} \rightarrow \text{Next Cycle}.
\end{array}}$$

The ordering itself is part of the governance model.

## 6.4 Reference Execution Algorithm

The following pseudocode represents the intended v5 control flow:

```python
def run_agentic_cycle(intent, state, planner, runtime):
    # ---------------------------------------------------------
    # 1. PLAN
    # ---------------------------------------------------------
    proposal = planner.propose(
        intent=intent,
        state=state,
    )
    proposal.validate()
    # ---------------------------------------------------------
    # 2. PROSPECTIVE ADMISSION
    # ---------------------------------------------------------
    admission = runtime.admission.evaluate(
        intent=intent,
        state=state,
        proposal=proposal,
    )
    if admission.decision == "BLOCK":
        return runtime.close_without_execution(
            intent=intent,
            state=state,
            proposal=proposal,
            admission=admission,
        )
    if admission.decision == "ESCALATE":
        return runtime.escalate(
            intent=intent,
            state=state,
            proposal=proposal,
            admission=admission,
        )
```

This pseudocode intentionally emphasizes ordering over software abstraction.

The most important property is that the executor receives an existing admission verdict rather than deciding whether its own action is permitted.

Likewise, the verifier evaluates the action after execution rather than allowing the tool to self-certify goal satisfaction.

## 6.5 Outer Agentic Loop

The individual cycle becomes autonomous behavior only when incorporated into a governed recurrent loop.

A reference outer loop may be expressed as:

```python
def run_intent(intent, initial_state, runtime, planner):
    state = initial_state
    while True:
        result = run_agentic_cycle(
            intent=intent,
            state=state,
            planner=planner,
            runtime=runtime,
        )
        state = result.state
        decision = result.closure.decision
        if decision == "HALT_SUCCESS":
            return AgentRunResult.success(
                state=state,
                final_receipt=result.receipt,
            )
        if decision == "HALT_FAILURE":
            return AgentRunResult.failure(
                state=state,
                final_receipt=result.receipt,
            )
        if decision == "ESCALATE":
            return AgentRunResult.escalated(
                state=state,
                final_receipt=result.receipt,
            )
        if decision == "CONSTRAIN":
            planner.apply_constraints(
                result.closure.constraints
            )
```

The important feature is what is absent.

There is no:

```python
while not plan.complete:
    execute_next_tool()
```

as the authoritative definition of autonomy.

Plan completion remains useful operational information, but it no longer determines whether another consequential action is permitted.

Instead:

```python
while closure_permits:
    propose()
    admit()
    execute()
    verify()
    reconcile()
    close()
```

better represents the v5 model.

## 6.6 Closure Does Not Bypass Admission

A subtle but important rule applies between adjacent cycles.

Suppose cycle $t$ ends with

$$D_{t} = \text{CONTINUE}.$$

This permits a new proposal:

$$a_{t + 1}\mathcal{= P}\left( I,X_{t + 1} \right).$$

It does **not** imply:

$$\mathrm{Execute}\left( a_{t + 1} \right).$$

The next action must independently satisfy:

$$V_{t + 1}^{-} = \mathcal{A}^{-}\left( a_{t + 1},I,X_{t + 1} \right).$$

Thus the recurrent relationship is

$$D_{t} = \text{CONTINUE} \Rightarrow \text{permission to propose},$$

not

$$D_{t} = \text{CONTINUE} \Rightarrow \text{permission to execute}.$$

This prevents authority from leaking across cycles.

A previous action's successful closure cannot serve as a reusable authorization token for future actions.

## 6.7 Replanning Semantics

Replanning modifies the proposed route to the goal without modifying the governing intent.

Let

$$P_{t}^{(k)}$$

denote plan revision $k$ during cycle $t$.

A replan may produce

$$P_{t}^{(k + 1)} = \mathrm{Replan}\left( P_{t}^{(k)},I,X_{t},\rho_{t} \right).$$

The following must remain invariant unless an explicit external authority update occurs:

$$g^{(k + 1)} = g^{(k)},$$

$$\kappa^{(k + 1)} = \kappa^{(k)},$$

and

$$\alpha^{(k + 1)} \subseteq \alpha^{(k)}.$$

A revised plan may be more conservative.

It may choose a different tool.

It may reduce scope.

It may abandon a previously selected approach.

It may not silently grant itself permissions that were absent from the governing Intent Contract.

This is the agentic equivalent of BEDROCK's prohibition against silently normalizing malformed evidence into a valid state.

## 6.8 Budget Reservation and Reconciliation

Resource handling should also preserve transaction ordering.

For an admitted proposal with predicted cost ${\widehat{c}}_{t}$,

$$\mathrm{Reserve}\left( {\widehat{c}}_{t} \right)$$

occurs before execution.

After execution produces actual observed cost $c_{t}$,

$$\rho_{c,t} = c_{t} - {\widehat{c}}_{t}$$

becomes part of residual evidence.

The ledger may then reconcile the reservation against actual usage.

Conceptually:

```python
reservation = budget.reserve(predicted_cost)
execution = execute(action)
actual_cost = execution.cost
budget.reconcile(
    reservation=reservation,
    actual_cost=actual_cost,
)
```

If execution never occurs, the reservation should be released according to policy.

If the outcome becomes unknown, resource reconciliation must not invent an observed cost merely to close the record.

Uncertainty remains represented until evidence resolves it or policy determines how to account for it.

## 6.9 Unknown-Outcome Branch

`OUTCOME_UNKNOWN` deserves its own path because it is fundamentally different from ordinary tool failure.

Suppose an irreversible or non-idempotent operation is submitted to an external system. Communication then fails before confirmation is received.

The agent now knows:

$$\text{request transmitted}$$

but does not know:

$$\text{side effect occurred}$$

or

$$\text{side effect did not occur}.$$

The correct transition is therefore:

```text
EXECUTING
    │
    ▼
OUTCOME_UNKNOWN
    │
    ├──► VERIFY EXTERNAL STATE
    │
    ├──► WAIT / REOBSERVE
    │
    └──► ESCALATE
```

It is not:

```text
OUTCOME_UNKNOWN
    │
    ▼
RETRY
```

unless the action is explicitly known to be idempotent or duplicate execution is independently admissible.

Formally,

$$S_{t}^{E} = \text{OUTCOME\_UNKNOWN} \land S\left( a_{t} \right) = \text{IRREVERSIBLE}$$

implies

$$D_{t} \neq \text{automatic retry}.$$

This should be directly testable in v5.

## 6.10 Verification Before Intent Completion

The runtime must also distinguish action completion from intent completion.

Let

$$\sigma = \lbrace \sigma_{1},\sigma_{2},\ldots,\sigma_{n}\rbrace $$

be the success conditions of the Intent Contract.

Then successful closure requires sufficient verified evidence for the mandatory conditions:

$$\underset{j = 1}{\bigwedge^{n}}\mathrm{Satisfied}\left( \sigma_{j} \right) = 1$$

for those conditions designated mandatory.

Accordingly,

```python
if execution.reported_success:
    # not enough to halt successfully
    ...
if verification.satisfies(intent.success_conditions):
    # may be eligible for HALT_SUCCESS
    ...
```

This allows a tool call to succeed while the overall intent remains incomplete.

It also permits multiple actions to contribute evidence toward one intent without pretending that any individual tool result constitutes final success.

## 6.11 Reflection as a Transaction

The v4 agentic runtime already allows reflection to alter membrane activations. In v5, this path should use the same atomicity discipline established elsewhere in BEDROCK.

Instead of:

```python
for index, delta in reflection.deltas.items():
    membrane.strings[index].activation += delta
```

the reference behavior becomes conceptually:

```python
candidate = membrane.snapshot()
for index, delta in reflection.deltas.items():
    candidate.apply_delta(index, delta)
candidate.validate()
membrane.commit(candidate)
```

If application or validation fails at any point:

```text
membrane == original_membrane
```

must remain true.

The invariant is

$$\mathrm{Rejected}\left( X_{t + 1}^{\ast} \right) \Rightarrow X_{t + 1} = X_{t}.$$

This is particularly important because reflection is generated from system feedback and may eventually incorporate probabilistic or model-generated reasoning.

The origin of a proposed correction cannot determine whether it bypasses state validation.

## 6.12 Memory Promotion Ordering

The memory path should follow the transaction ordering already established by the hardened SAO implementation.

The incorrect ordering would be:

```text
observe
→ write durable memory
→ evaluate temporal/policy suitability
```

The required ordering is:

```text
observe
→ construct provisional evidence
→ verify
→ temporal qualification
→ memory admissibility
→ final promotion verdict
→ commit durable memory
```

Formally,

$$J_{t}\overset{\text{qualify}}{\rightarrow}P_{M,t}$$

and only

$$P_{M,t} = \text{PROMOTE}$$

permits

$$\mathcal{M}_{t + 1} = \mathrm{Commit}\left( J_{t} \right).$$

The rule follows directly from BEDROCK III's SAO correction:

$$\boxed{\text{Final verdict precedes authoritative evidence commit.}}$$

## 6.13 Action Receipt Commit Point

The final Action Receipt should be created only after all consequential decisions for the cycle are known.

Its commit point therefore occurs after closure:

$$\text{Closure} \rightarrow \text{Action Receipt}.$$

Intermediate trace events may be recorded during execution for diagnostics or crash recovery, but they must remain distinguishable from the **final cycle receipt**.

This prevents a partial receipt from claiming that a transition completed when the runtime terminated midway through reconciliation.

A complete receipt should provide a deterministic narrative of the cycle:

```text
Intent I-042
Plan revision 3
Action A-117 proposed
Admission: ADMIT
Budget reserved
Execution: EXECUTED
Observation captured
Verification: PARTIAL
Residual: non-zero goal + cost discrepancy
Reflection candidate validated and committed
Memory promotion: REJECTED
Closure: REPLAN
```

An engineer should be able to reconstruct the system's decision path from the receipt without inferring hidden state transitions.

## 6.14 Deterministic Reference Mode

For research and testing, v5 should support a **deterministic reference mode**.

Given identical:

$$I,\quad X_{0},\quad s,\quad\Pi,$$

where $s$ is the random seed and $\Pi$ represents the same policy/configuration set, repeated executions should produce the same governed trajectory whenever all participating components are deterministic.

Define

$$\mathcal{T =}R\left( I,X_{0},s,\Pi \right),$$

where $\mathcal{T}$ is the resulting trajectory of state transitions and receipts.

Then the deterministic reference requirement is

$$R\left( I,X_{0},s,\Pi \right) = R\left( I,X_{0},s,\Pi \right)$$

across repeated runs under the same controlled environment.

More explicitly,

$$\mathcal{T = \lbrace }a_{t},V_{t}^{-},e_{t},q_{t},\rho_{t},X_{t + 1},D_{t},R_{t}^{A}\rbrace _{t = 1}^{N}.$$

This does not imply that all real-world agentic behavior must be deterministic. External APIs, networks, humans, clocks, probabilistic models, and environments may introduce nondeterminism.

Instead, deterministic reference mode provides a controlled experimental surface for testing whether the **governance system itself** produces reproducible decisions from reproducible evidence.

This distinction will be central to the experimental methodology described later in the paper.

## 6.15 Seeded Execution Harness

A minimal experimental harness may therefore take the form:

```python
def run_seed(seed, intent, initial_state, config):
    rng = Random(seed)
    runtime = AgenticRuntime(
        rng=rng,
        config=config,
        deterministic=True,
    )
    return runtime.run(
        intent=intent,
        initial_state=initial_state,
    )
```

Repetition becomes:

```python
run_a = run_seed(
    seed=42,
    intent=intent,
    initial_state=state,
    config=config,
)
run_b = run_seed(
    seed=42,
    intent=intent,
    initial_state=state,
    config=config,
)
assert run_a.receipts == run_b.receipts
assert run_a.final_state == run_b.final_state
```

A different seed may alter allowed stochastic choices:

```python
run_c = run_seed(seed=43, ...)
```

but must not alter the validity of hard governance invariants.

That is,

$$s_{1} \neq s_{2}$$

may produce different admissible plans, while both executions must still satisfy:

$$I_{\text{authority}} = I_{\text{atomic}} = I_{\text{budget}} = I_{\text{verification}} = I_{\text{memory}} = I_{\text{closure}} = 1.$$

Randomness may influence choices within the permitted state space.

It may not randomize the rules defining that state space.

## 6.16 Reference Module Boundaries for v5

The final implementation does not need to use these exact filenames, but Paper XII suggests a logical decomposition resembling:

```text
agentic/
    intent.py
    proposal.py
    admission.py
    execution.py
    verification.py
    residual.py
    reflection.py
    closure.py
    receipt.py
    runtime.py
```

Existing U.F.O. modules should be reused rather than duplicated wherever their semantics already match the contract.

For example:

- the radial membrane remains the behavioral substrate;

- existing admissibility mechanisms contribute to preflight decisions;

- the Bounded Compute Envelope contributes resource pressure;

- existing tool abstractions remain execution surfaces;

- temporal governance contributes evidence qualification;

- semantic memory remains the durable memory mechanism;

- SAO contributes final-verdict promotion semantics;

- existing multi-agent and swarm components remain candidate-selection mechanisms;

- the Invariant Handshake remains available when execution crosses compatible legacy-compute boundaries.

The purpose of v5 is to add **composition**, not parallel replacements for machinery that BEDROCK has already hardened.

## 6.17 Minimal v5 Reference Loop

The architecture can ultimately be reduced to the following implementation-oriented form:

```python
while not terminal:
    action = planner.propose(intent, state)
    admission = admit(intent, state, action)
    if not admission.executable:
        closure = close_without_execution(...)
        handle(closure)
        continue
    execution = execute(action, admission)
    observation = observe(execution)
    verification = verify(
        action.expected_postconditions,
        observation,
    )
    residual = reconcile(
        predicted=action,
        actual=execution,
        verified=verification,
    )
    candidate_state = reflect(
        state,
        residual,
    )
    state = validate_and_commit(
        previous=state,
        candidate=candidate_state,
    )
```

The significance of this loop is not its complexity.

It is its ordering.

No individual component is especially exotic. The contribution is the requirement that these components compose without transferring authority accidentally between stages.

The v4 architecture already contains most of the underlying machinery.

The v5 reference implementation makes the complete transaction explicit.

**The planner proposes. Governance authorizes. The executor acts. Evidence verifies. Residuals reconcile. Reflection adapts. Memory qualifies. Closure decides whether the system is allowed to breathe again.**

For implementation, the important part of that final statement is not the metaphor. It is the sequence.

Every verb corresponds to a distinct engineering responsibility, and each boundary can be independently tested.

# 7. Falsification and Seeded Validation Plan

The purpose of the v5 experimental program is not to demonstrate that the agentic architecture can produce impressive outputs. It is to determine whether the execution contracts defined in this paper remain valid under controlled, repeated, and adversarial conditions.

The principal evaluation question is therefore not:

Can the agent complete a task?

It is:

**Can the agent complete, fail, replan, stop, or escalate without violating the invariants that govern authority, state, evidence, memory, and continuation?**

This distinction is important because task completion alone is an insufficient measure of agentic correctness. An agent may reach the requested outcome while exceeding authority, mutating state incorrectly, retrying an uncertain irreversible action, promoting unverified evidence to memory, or continuing after its governing contract should have halted execution.

For this reason, the v5 validation program is organized around **falsification**.

The implementation should be treated as correct only to the extent that repeated attempts to violate its stated invariants fail.

## 7.1 Experimental Objective

For a governed run with initial condition

$$\Theta_{0} = \left( I,X_{0},s,\Pi \right),$$

where

$$I = \text{Intent Contract},$$

$$X_{0} = \text{initial U.F.O. state},$$

$$s = \text{random seed},$$

and

$$\Pi = \text{policy and runtime configuration},$$

the runtime produces a trajectory

$$\mathcal{T =}R\left( \Theta_{0} \right).$$

The trajectory contains the sequence

$$\mathcal{T = \lbrace }a_{t},V_{t}^{-},e_{t},o_{t},q_{t},\rho_{t},X_{t + 1},D_{t},R_{t}^{A}\rbrace _{t = 1}^{N}.$$

The experimental objective is to determine whether this trajectory preserves the invariants specified by the architecture for all exercised conditions.

For each run, define an invariant vector

$$\mathcal{I}_{t} = \left( I_{A},I_{B},I_{S},I_{V},I_{M},I_{R},I_{C},I_{P} \right)_{t},$$

where

$$I_{A} = \text{authority invariant},$$

$$I_{B} = \text{budget invariant},$$

$$I_{S} = \text{state atomicity invariant},$$

$$I_{V} = \text{verification invariant},$$

$$I_{M} = \text{memory invariant},$$

$$I_{R} = \text{retry/side-effect invariant},$$

$$I_{C} = \text{closure invariant},$$

and

$$I_{P} = \text{provenance invariant}.$$

For a conforming trajectory,

$$I_{j} = 1$$

for every mandatory invariant $j$ at every applicable cycle.

Therefore,

$$\prod_{j}^{}I_{j} = 1.$$

A single violated mandatory invariant falsifies the implementation behavior for that test case even if the external task itself succeeds.

## 7.2 Deterministic Replay Test

The first experiment establishes whether the governance system behaves reproducibly under controlled conditions.

For identical

$$\left( I,X_{0},s,\Pi \right),$$

repeated runs should produce identical governance trajectories whenever the planner, tools, environment, and observation layer are operating in deterministic reference mode.

Thus,

$$R\left( I,X_{0},s,\Pi \right) = R\left( I,X_{0},s,\Pi \right).$$

The comparison should include more than final state.

At minimum, repeated runs should reproduce:

- action proposals,

- admission verdicts,

- approved scopes,

- budget reservations,

- execution states,

- verification results,

- residual vectors,

- committed state transitions,

- memory-promotion decisions,

- closure decisions,

- and final Action Receipts.

A deterministic test therefore asks whether:

```text
run_a.receipts == run_b.receipts
run_a.final_state == run_b.final_state
run_a.closure_history == run_b.closure_history
```

for identical controlled inputs.

This test evaluates whether hidden nondeterminism exists inside the governance path.

## 7.3 Seed Variation Test

A deterministic reference system should also distinguish between **permitted stochastic variation** and **governance variation**.

Let

$$s_{1} \neq s_{2}.$$

Different seeds may produce different candidate actions when stochastic planning is enabled:

$$a_{t}^{\left( s_{1} \right)} \neq a_{t}^{\left( s_{2} \right)}.$$

However, the hard governance invariants should remain independent of the seed.

For example, if both proposals exceed the same authority boundary,

$$V_{t}^{-}\left( s_{1} \right) = V_{t}^{-}\left( s_{2} \right) = \text{BLOCK}$$

even if the proposed actions differ.

This establishes:

$$\boxed{\text{Randomness may alter choices within the admissible region, but not the definition of the admissible region.}}$$

A useful experiment is therefore to execute a fixed scenario across a seed set

$$S = \lbrace  s_{1},s_{2},\ldots,s_{n}\rbrace $$

and verify that all mandatory invariants remain satisfied.

## 7.4 Pre-Execution Authority Falsification

The first major security property is:

$$\boxed{\text{A blocked action must never execute.}}$$

Construct proposals that intentionally violate:

- capability scope,

- delegated authority,

- immutable intent constraints,

- side-effect ceilings,

- policy requirements,

- or required resource bounds.

For each case,

$$V_{t}^{-} \neq \text{ADMIT}$$

must imply

$$S_{t}^{E} = \text{NOT\_EXECUTED}.$$

The test should inspect not only the returned verdict but also the executor call count and resulting system state.

For example:

```python
assert admission.decision == "BLOCK"assert executor.calls == 0assert state_after == state_before
```

A system that correctly labels an action as blocked but still invokes the tool has failed the governing invariant.

## 7.5 Immutable Intent Test

Replanning must not change the mandate under which the agent operates.

Let the original Intent Contract contain

$$I_{0} = (g,\sigma,\kappa,\beta,\alpha,\chi,\eta).$$

After one or more replanning events,

$$P^{(0)} \rightarrow P^{(1)} \rightarrow \cdots \rightarrow P^{(n)},$$

the test should verify:

$$g_{n} = g_{0},$$

$$\kappa_{n} = \kappa_{0},$$

and, absent explicit new delegation,

$$\alpha_{n} \subseteq \alpha_{0}.$$

A particularly important adversarial test is to force the planner into repeated failure and determine whether recovery logic attempts to solve the task by silently widening scope.

The invariant is:

$$\boxed{\text{Repeated failure must not create authority.}}$$

## 7.6 Budget Monotonicity Test

Budget handling should be tested for both accounting correctness and directional behavior.

Given two otherwise identical states with

$$\beta_{A} >  \beta_{B},$$

an action requiring cost $c$ should not become admissible under the smaller budget if it is inadmissible under the larger budget solely because of budget sufficiency.

Likewise, for two otherwise identical actions with

$${\widehat{c}}_{A} <  {\widehat{c}}_{B},$$

the more expensive action must not gain an admission advantage when cost is the only changed variable.

This can be tested through controlled sweeps over

$$\beta$$

and

$$\widehat{c}.$$

A simple grid may evaluate:

$$\beta \in \lbrace  0.5,1.0,2.0,5.0,10.0\rbrace $$

against

$$\widehat{c} \in \lbrace  0.25,0.75,1.5,3.0,8.0\rbrace .$$

The expected boundary condition is

$$\widehat{c} \leq \beta$$

for the governed budget dimension, subject to all other required gates.

This does not require admission to be determined by budget alone.

It requires budget logic to behave consistently.

## 7.7 Side-Effect Escalation Test

The side-effect classification should influence governance in predictable ways.

Consider otherwise identical actions assigned:

$$S(a) = \text{OBSERVATIONAL},$$

$$S(a) = \text{REVERSIBLE},$$

$$S(a) = \text{COMPENSATABLE},$$

and

$$S(a) = \text{IRREVERSIBLE}.$$

As side-effect severity increases, the system should not become less restrictive when all other evidence remains unchanged.

A useful expected relation is:

$$\begin{matrix}
\mathrm{Permission}\left( \text{IRREVERSIBLE} \right) \leq \mathrm{Permission}\left( \text{COMPENSATABLE} \right) \cr
 \leq \mathrm{Permission}\left( \text{REVERSIBLE} \right) \leq \mathrm{Permission}\left( \text{OBSERVATIONAL} \right),
\end{matrix}$$

for configurations in which side-effect class contributes monotonically to the governing decision.

The exact verdict need not be a scalar probability. The experiment only tests that increased irreversible exposure does not accidentally weaken the governing boundary.

## 7.8 Unknown-Outcome Retry Falsification

This test should be treated as mandatory.

Construct a simulated external action with

$$S\left( a_{t} \right) = \text{IRREVERSIBLE}$$

and force the executor to return

$$S_{t}^{E} = \text{OUTCOME\_UNKNOWN}.$$

Then verify that the runtime does not automatically execute the same action again.

Required behavior should be one of:

- reobserve,

- verify external state,

- wait,

- constrain,

- or escalate.

The forbidden behavior is:

$$\text{OUTCOME\_UNKNOWN} \rightarrow \text{automatic duplicate execution}.$$

A direct test may look conceptually like:

```python
assert first.execution_state == "OUTCOME_UNKNOWN"
assert executor.call_count == 1
runtime.advance()
assert executor.call_count == 1
assert next_decision in {
    "VERIFY",
    "WAIT",
    "ESCALATE",
    "CONSTRAIN",
}
```

This is one of the clearest places where a controlled simulation can expose a dangerous agentic mistake.

## 7.9 Tool Success versus Verification Test

To test the distinction

$$\text{Tool Success} \neq \text{Verified Outcome},$$

construct a tool that reports:

```python
success = True
```

while deliberately returning an external state that violates the expected postcondition.

For example:

```text
Expected:
file digest = H_expected

Observed:
file digest = H_wrong
```

The system should produce something equivalent to

$$e_{t}.\text{success} = 1$$

while

$$q_{t} = \text{FAILED}.$$

The closure operator must not return `HALT_SUCCESS` solely because execution reported success.

This test validates that success authority belongs to the appropriate layer.

## 7.10 Partial and Unknown Verification Tests

The verifier must distinguish between failure and insufficient evidence.

Three separate cases should be tested:

### Verified

All mandatory postconditions are supported:

$$Q_{t} = \text{VERIFIED}.$$

### Failed

Available evidence contradicts a mandatory postcondition:

$$Q_{t} = \text{FAILED}.$$

### Unknown

Evidence is insufficient:

$$Q_{t} = \text{UNKNOWN}.$$

The agent should not collapse

$$\text{UNKNOWN}$$

into either success or failure unless policy explicitly defines that conversion.

This preserves epistemic state in the execution record.

## 7.11 Atomic Reflection Falsification

Reflection should be tested using the same adversarial strategy that proved effective during BEDROCK boundary hardening.

Construct a reflection candidate containing several valid updates followed by one invalid update.

For example:

$$\Delta X = \left( \delta_{1},\delta_{2},\delta_{3},\delta_{\text{invalid}} \right).$$

The implementation must reject the complete candidate without retaining earlier partial mutations.

If

$$\mathrm{Valid}\left( X_{t + 1}^{\ast} \right) = 0,$$

then

$$X_{t + 1} = X_{t}.$$

Test:

```python
before = snapshot(state)
candidate = reflector.propose(...)
result = validate_and_commit(candidate)
assert result.rejected
assert state == before
```

This converts the BEDROCK atomic-update lesson directly into an agentic invariant.

## 7.12 Memory Promotion Falsification

The memory path should be tested using execution evidence that varies in verification quality and temporal qualification.

At minimum:

### Verified and admissible evidence

May become eligible for promotion.

### Failed verification

Must not become durable memory as established fact.

### Unknown verification

Must not silently become durable truth.

### Policy-rejected evidence

Must leave durable memory unchanged.

### Temporally insufficient evidence

Must remain provisional if the policy requires persistence.

Formally,

$$P_{M,t} \neq \text{PROMOTE} \Rightarrow \mathcal{M}_{t + 1} = \mathcal{M}_{t}.$$

The test should inspect durable state rather than relying only on a returned status.

## 7.13 Final-Verdict Commit Test

The SAO defect corrected during BEDROCK motivates a general regression for v5.

Construct a cycle in which an intermediate stage initially favors continuation or promotion, while a later mandatory gate rejects it.

The runtime must ensure that no authoritative evidence claiming the earlier positive verdict has already been committed.

For example:

$$\text{preliminary admission favorable}$$

followed by

$$\text{temporal or policy rejection}.$$

The final receipt and authoritative state must reflect the final verdict only.

This can be summarized as:

$$\boxed{\text{Intermediate optimism must not survive final rejection as authoritative state.}}$$

## 7.14 Closure Falsification

The closure operator should be tested independently from plan completion.

Construct a plan with remaining steps but force the current state to violate a mandatory continuation condition.

The expected result should be:

$$D_{t} \neq \text{CONTINUE}.$$

Conversely, construct a plan that has exhausted its current steps while the Intent Contract success condition remains unsatisfied.

The system should not automatically return:

$$\text{HALT\_SUCCESS}.$$

This proves that:

$$\boxed{\text{Plan status} \neq \text{Agentic closure status}.}$$

The closure operator should answer whether another governed cycle is permitted or required, not merely whether a list of steps has been consumed.

## 7.15 Malformed Numeric Evidence

BEDROCK demonstrated that malformed numeric inputs can produce subtle failures even when ordinary comparison logic appears correct.

The v5 tests should therefore include:

- NaN,

- +Inf,

- -Inf,

- booleans used as numeric values,

- numeric strings,

- negative values in non-negative domains,

- extremely large integers,

- extremely small floating-point values,

- malformed vectors,

- incorrect dimensionality.

These values should be injected into:

- predicted cost,

- actual cost,

- side-effect rating,

- tension,

- curvature,

- residual components,

- budget,

- verification metrics,

- memory confidence,

- and closure inputs.

No malformed numerical evidence should:

1.  crash a trust boundary unexpectedly,

2.  become silently normalized into admissible state,

3.  bypass a comparison,

4.  or partially mutate authoritative state.

## 7.16 Provenance Integrity Test

Every completed agentic cycle should produce an Action Receipt whose references agree with the actual execution path.

For receipt $R_{t}^{A}$, verify that:

- the intent identifier matches the active Intent Contract,

- the action identifier matches the proposal actually evaluated,

- the admission record matches the authorization supplied to the executor,

- the execution evidence belongs to that action,

- the verification result references the expected postconditions from that proposal,

- the residual was derived from the same cycle,

- the committed state transition corresponds to the recorded reflection,

- the memory result matches the actual promotion decision,

- and the closure decision matches the final runtime transition.

A receipt should fail validation if these links do not compose.

This tests the claim that the receipt represents **provenance**, not merely logging.

## 7.17 Cross-Receipt Chain Validation

For consecutive cycles,

$$R_{t}^{A}$$

and

$$R_{t + 1}^{A},$$

the final authoritative state of the first cycle should correspond to the initial authoritative state of the second:

$$X_{t + 1}^{\text{final}} = X_{t + 1}^{\text{initial}}.$$

Likewise, if closure returns CONTINUE, the next receipt should represent a new proposal and new prospective admission rather than reusing the previous action's authorization.

This permits validation of the complete chain:

$$R_{1}^{A} \rightarrow R_{2}^{A} \rightarrow \cdots \rightarrow R_{N}^{A}.$$

A break in the chain indicates hidden state or provenance loss.

## 7.18 Experimental Parameter Sweeps

Once the individual falsification tests pass, the reference simulator can explore how the model behaves across controlled parameter changes.

Useful dimensions include:

$$\beta = \text{available budget},$$

$$\chi = \text{side-effect ceiling},$$

$$\tau = \text{membrane tension},$$

$$\kappa = \text{boundary curvature},$$

$$\rho_{g} = \text{goal residual},$$

$$\rho_{u} = \text{uncertainty residual},$$

and relevant policy thresholds.

For a controlled experiment, all variables except one should remain fixed wherever possible.

This allows the implementation to test directional expectations such as:

$$\chi \downarrow \Rightarrow \text{high-side-effect actions become no easier to admit},$$

or

$$\rho_{u} \uparrow \Rightarrow \text{irreversible autonomous continuation becomes no easier}.$$

These are not universal mathematical theorems unless the implementation explicitly defines them as such.

They are **engineering expectations that can be falsified experimentally**.

## 7.19 Suggested Initial Seed Matrix

The first v5 simulator does not require millions of runs.

A practical initial matrix could use:

$$100$$

seeds across a small collection of deliberately different scenarios.

For example:

- low-risk successful task,

- budget-constrained task,

- authority violation,

- verification mismatch,

- tool failure,

- unknown irreversible outcome,

- reflection-invalid state,

- memory-promotion rejection,

- repeated replanning,

- multi-agent selected-but-not-authorized execution.

With ten scenario classes and one hundred seeds each:

$$10 \times 100 = 1000$$

runs provide an initial behavioral surface that remains inexpensive enough to inspect.

The point is not statistical prestige.

The point is to make unexpected transitions visible before the model is integrated deeply into the production repository.

## 7.20 Expected Experimental Artifacts

Each seeded run should produce machine-readable outputs containing at least:

```text
seed
scenario_id
intent_id
initial_state_digest
action_sequence
admission_sequence
execution_sequence
verification_sequence
residual_sequence
state_transition_sequence
memory_sequence
closure_sequence
receipt_digest_sequence
final_state_digest
invariant_results
```

These artifacts allow comparison across seeds and configurations without relying on terminal output.

They also provide a basis for future regression fixtures.

## 7.21 Pass/Fail Criterion

The principal pass condition for the initial v5 model is not:

$$\text{task completion rate} = 100\% .$$

Some scenarios are deliberately designed to fail, block, or escalate.

A more meaningful requirement is:

$$\boxed{\text{No tested trajectory may violate a mandatory governance invariant.}}$$

Thus a run that terminates with:

$$D_{t} = \text{HALT\_FAILURE}$$

may still be a successful engineering test if the system stopped correctly.

Likewise,

$$D_{t} = \text{ESCALATE}$$

may represent correct behavior when the available authority or evidence is insufficient.

The testing framework therefore distinguishes:

$$\text{task outcome}$$

from

$$\text{governance correctness}.$$

This distinction is essential for evaluating bounded agency.

## 7.22 Experimental Interpretation

The seeded validation program is not intended to establish that U.F.O. is universally safe, optimal, or correct in arbitrary external environments.

It tests a narrower claim:

**Given explicit contracts and controlled evidence, the U.F.O. agentic runtime should preserve its stated governance invariants across repeated, adversarial, and reproducible execution trajectories.**

If the experiments reveal a contradiction, the correct response is not to weaken the test until it passes.

The model, contract, or implementation should be revised.

This is the methodological consequence of BEDROCK.

The paper defines the behavior.

The simulator attempts to falsify it.

The implementation must then prove itself against both.

**Section 8 defines the implementation boundary for U.F.O. v5.0.0: what the release should actually build, what should remain unchanged from v4, and which research claims remain deliberately outside scope.**

# 8. U.F.O. v5.0.0 Implementation Boundary and Scope

U.F.O. v5.0.0 is intended to implement the Governed Agentic Closure model defined in this paper over the hardened v4 BEDROCK foundation.

The release should not be treated as a general rewrite of U.F.O., nor as an opportunity to replace existing mathematical or governance components with new abstractions. The principal engineering objective is narrower:

**Connect the existing U.F.O. mechanisms through an explicit agentic execution protocol that preserves authority, evidence, atomic state, verification, memory qualification, and closure across every action cycle.**

The v5 implementation should therefore prefer composition over replacement.

Where v4 already contains a suitable mechanism, v5 should integrate it. New code should be introduced only where the existing architecture lacks a contract or operator required by the agentic cycle.

## 8.1 Normative Language

For implementation clarity, this section uses three requirement levels.

**MUST** indicates behavior required for conformance with the Paper XII reference model.

**SHOULD** indicates behavior strongly recommended for the v5 reference implementation but not fundamental to the theoretical contract.

**MAY** indicates an implementation option that does not alter the governing semantics.

A v5 implementation that violates a MUST requirement should not be considered conformant even if its task-completion behavior appears correct.

## 8.2 Components Preserved from v4

The following architectural components remain part of the v5 foundation and should not be replaced merely to accommodate the new agentic execution path:

- radial membrane and Twelve Strings;

- deformable boundary geometry;

- V-Channel propagation;

- Governor and Holistic Governor mechanisms;

- Bounded Compute Envelope;

- temporal governance;

- residual ledgers;

- Symmetric Ascension Operator;

- semantic memory;

- federated shards and mesh aggregation;

- multi-agent and swarm primitives;

- existing tool abstractions;

- Invariant Handshake;

- numeric trust-boundary validation introduced during BEDROCK.

These components may require adapters or new integration points, but their established semantics should remain intact unless a direct contradiction with the Paper XII execution contract is demonstrated.

The default engineering assumption is therefore:

$$\boxed{\text{Existing validated mechanism} \rightarrow \text{reuse or adapt}}$$

rather than

$$\boxed{\text{Existing validated mechanism} \rightarrow \text{replace}}$$

This preserves the value of the v4 conformance and hardening work.

## 8.3 Required v5 Agentic Contracts

The first mandatory requirement is implementation of the contracts defined in Section 4.

At minimum, v5 MUST provide executable representations for:

$$I = \text{Intent Contract},$$

$$a_{t} = \text{Action Proposal},$$

$$V_{t}^{-} = \text{Prospective Admission Verdict},$$

$$e_{t} = \text{Execution Record},$$

$$q_{t} = \text{Verification Result},$$

$$\rho_{t} = \text{Agentic Residual Vector},$$

$$D_{t} = \text{Agentic Closure Decision},$$

and

$$R_{t}^{A} = \text{Agentic Action Receipt}.$$

These objects may be implemented through dataclasses, typed models, immutable records, or another suitable mechanism.

The implementation language is secondary.

The required semantics are not.

Each contract MUST validate its own documented invariants at construction or before authoritative use.

Malformed contract state must fail closed.

## 8.4 Intent Must Become an Explicit Runtime Object

The current agentic system accepts a goal primarily as text.

v5 MUST introduce an explicit IntentContract or semantically equivalent object.

A reference structure may resemble:

```python
@dataclass(frozen=True)
class IntentContract:
    intent_id: str
    goal: str
    success_conditions: tuple[Postcondition, ...]
    immutable_constraints: tuple[Constraint, ...]
    budget: ResourceBudget
    authority: AuthorityEnvelope
    side_effect_ceiling: SideEffectClass
    provenance: IntentProvenance
```

The exact fields may evolve during implementation, but the separation between goal, authority, constraints, budget, and success conditions MUST be preserved.

A plan MUST reference an Intent Contract.

A plan MUST NOT become the authoritative storage location for immutable intent constraints.

This ensures that replanning can modify execution strategy without modifying mandate.

## 8.5 Planning Must Produce Proposals

The existing GoalPlanner SHOULD remain usable, but its role must be narrowed conceptually.

Its output should become an ActionProposal or a sequence of proposals rather than directly implying executable permission.

The planner MUST NOT be responsible for its own authorization.

A proposal SHOULD include:

```text
action_id
intent_id
plan_revision
tool_or_capability
parameters
expected_postconditions
predicted_cost
predicted_side_effect
required_authority
provenance
```

The planner MAY continue to use the existing heuristic goal classification during the initial v5 implementation.

Paper XII does not require v5 to invent a more intelligent planner.

This distinction is deliberate.

The quality of the planner is separable from the correctness of the governance system.

## 8.6 Prospective Admission Must Precede Every Consequential Execution

v5 MUST introduce a single explicit pre-execution admission boundary.

Conceptually:

```python
verdict = agentic_admission.evaluate(
    intent=intent,
    proposal=proposal,
    state=state,
)
```

A proposal MUST NOT invoke a consequential tool before this verdict exists.

Only:

```text
ADMIT
```

may directly permit execution.

CONSTRAIN, REPROJECT, BLOCK, and ESCALATE MUST result in a non-execution path until a newly admissible proposal exists.

The admission layer SHOULD compose existing U.F.O. evidence rather than duplicate it.

Relevant inputs may include:

- Bounded Compute Envelope state;

- membrane tension;

- curvature;

- policy state;

- delegated authority;

- remaining budget;

- tool capability requirements;

- predicted side-effect class;

- applicable Invariant Handshake status.

No single diagnostic scalar should silently replace mandatory gates.

## 8.7 Budget Reservation Must Become Prospective

The existing planner records cost after execution.

v5 SHOULD introduce pre-execution reservation.

For predicted cost

$${\widehat{c}}_{t},$$

the runtime should verify that the applicable resource dimensions remain available before execution.

Conceptually:

```python
reservation = budget.reserve(
    predicted=proposal.predicted_cost)
```

Execution then produces actual cost:

```python
actual = execution.cost
```

followed by reconciliation:

```text
budget.reconcile(
    reservation=reservation,
    actual=actual,
)
```

The implementation MUST preserve the difference between predicted and actual resource use.

Unexpected cost must therefore become residual evidence rather than disappear into an overwritten total.

## 8.8 Execution Must Represent Uncertainty Explicitly

The tool subsystem MUST support an execution state equivalent to:

```text
NOT_EXECUTED
EXECUTED
FAILED
OUTCOME_UNKNOWN
```

This is necessary because failure to observe confirmation is not equivalent to proof that an external action did not occur.

The runtime SHOULD associate every tool or capability with a side-effect class such as:

```text
OBSERVATIONAL
REVERSIBLE
COMPENSATABLE
IRREVERSIBLE
```

For an irreversible or non-idempotent action returning `OUTCOME_UNKNOWN`, automatic retry MUST be prohibited unless independent policy explicitly establishes retry safety.

This requirement applies even when retry would make task completion more likely.

Governance correctness takes precedence over optimistic completion.

## 8.9 Postcondition Verification Must Be Independent of Tool Success

v5 MUST introduce an explicit verifier.

The verifier receives expected postconditions and available observations:

$$q_{t}\mathcal{= V}\left( a_{t},e_{t},o_{t} \right).$$

It MUST support a result space capable of distinguishing at least:

```text
VERIFIED
PARTIAL
FAILED
UNKNOWN
```

A tool reporting success MUST NOT automatically produce VERIFIED.

Likewise, missing evidence MUST NOT silently become FAILED when the actual condition is unknown.

The verifier MAY use deterministic functions initially.

For example:

```python
def verify_file_write(expected, observed):
    return (
        observed.exists
        and observed.digest == expected.digest
    )
```

The first v5 release does not require a universal semantic verifier.

It requires the **boundary** through which domain-specific verifiers can operate.

## 8.10 Residual Evidence Must Remain Structured

v5 MUST implement the Agentic Residual Vector or a semantically equivalent typed structure.

A reference interface may contain:

```python
@dataclass(frozen=True)
class AgenticResidual:
    goal_residual: ...
    cost_residual: ...
    policy_residual: ...
    stability_residual: ...
    uncertainty_residual: ...
```

The components MUST NOT be assumed to share units.

The initial implementation SHOULD avoid collapsing them into one arbitrary weighted score.

Individual operators may derive local normalized values where necessary, but such transformations should remain explicit.

The purpose of the residual is to preserve unresolved information.

It is not merely a failure score.

## 8.11 Reflection Must Become Candidate-Validate-Commit

The existing ReflectionEngine SHOULD be retained.

Its output path MUST change where necessary so that suggested membrane changes do not mutate authoritative state incrementally.

The required sequence is:

```text
reflection
→ candidate state
→ complete validation
→ atomic commit
```

Conceptually:

```python
candidate = state.snapshot()
candidate.apply(reflection)
candidate.validate()
state.commit(candidate)
```

If any portion of the candidate is invalid:

```text
state_after == state_before
```

MUST hold.

This requirement extends the atomicity discipline introduced during BEDROCK into adaptive agentic state.

## 8.12 Evidence Must Enter Memory Through Qualification

v5 MUST connect execution evidence to the existing temporal-governance and semantic-memory systems through an explicit provisional stage.

The required conceptual path is:

$$\begin{matrix}
\text{Observation} \rightarrow \text{Verification} \rightarrow \text{Provisional Evidence} \rightarrow \text{Temporal Qualification} \cr
 \rightarrow \text{Memory Admission} \rightarrow \text{Promotion} \rightarrow \text{Durable Memory}.
\end{matrix}$$

The initial implementation does not need to make every action produce memory.

It SHOULD instead make memory creation explicit and selective.

Failed or unknown verification MUST NOT automatically produce authoritative durable facts.

The existing SAO final-verdict ordering should be reused where applicable:

$$\boxed{\text{final promotion verdict} \rightarrow \text{memory commit}.}$$

Not the reverse.

## 8.13 Agentic Closure Is the Required New Control Operator

The most important new v5 control mechanism is the **Agentic Closure Operator**.

The runtime MUST evaluate closure after the previous action's consequences have been reconciled.

A reference interface may resemble:

```python
closure = closure_operator.evaluate(
    intent=intent,
    state=state,
    verification=verification,
    residual=residual,
    remaining_budget=budget,
    memory_result=memory_result,
)
```

The output MUST distinguish at least:

```text
CONTINUE
REPLAN
CONSTRAIN
REFLECT
ESCALATE
HALT_SUCCESS
HALT_FAILURE
```

The closure operator MUST NOT simply return CONTINUE because the plan contains another step.

Likewise, it MUST NOT return `HALT_SUCCESS` merely because the final tool invocation reported success.

This operator is the primary new behavioral boundary introduced by Paper XII.

## 8.14 Continued Execution Must Re-enter Admission

When:

$$D_{t} = \text{CONTINUE},$$

the runtime may request a new proposal.

It MUST NOT reuse the previous prospective authorization.

The next cycle must contain:

$$a_{t + 1}\mathcal{= P}\left( I,X_{t + 1} \right),$$

followed by

$$V_{t + 1}^{-} = \mathcal{A}^{-}\left( a_{t + 1},I,X_{t + 1} \right).$$

Thus:

$$\boxed{\text{Closure authorizes another consideration cycle, not another tool call.}}$$

This prevents authority from leaking across agentic iterations.

## 8.15 Action Receipts Must Bind the Complete Cycle

Every completed v5 cycle SHOULD generate an AgenticActionReceipt.

For the reference implementation, this SHOULD be treated as a first-class artifact rather than reconstructed later from unrelated logs.

It should bind:

- intent;

- plan revision;

- proposal;

- admission verdict;

- budget reservation;

- execution state;

- observation;

- verification;

- residual;

- reflection;

- committed state;

- memory verdict;

- closure decision.

A receipt may be serialized for test comparison.

The initial v5 implementation SHOULD support stable deterministic receipt serialization when operating in deterministic reference mode.

This becomes important for seeded replay.

## 8.16 Multi-Agent Selection Must Remain Separate from Execution Authority

The existing swarm auction mechanism SHOULD remain intact as the candidate-selection mechanism.

After a winning agent is selected, however, the selected agent MUST still enter the same prospective admission path as a single-agent execution.

Therefore:

$$\mathrm{Winner}(K) = U_{i}$$

does not imply:

$$\mathrm{Authorized}\left( a_{t} \right) = 1.$$

The agent inherits only the authority explicitly delegated by its contract.

Replanning by the winning agent MUST NOT expand that authority.

This allows the single-agent and multi-agent execution paths to share the same governance semantics.

## 8.17 Invariant Handshake Integration

The Invariant Handshake SHOULD remain an optional execution boundary where an agentic action crosses into a supported legacy or scalar-compute regime.

v5 does not require every agentic action to invoke the handshake.

Instead, the handshake should be treated as one possible mandatory gate when the execution path involves the corresponding interoperability condition.

Conceptually:

```python
if proposal.requires_legacy_boundary:
    status = handshake(...)
    if not status.handshakeAllowed:
        admission = BLOCK
```

The handshake therefore contributes to prospective admission.

It does not replace prospective admission.

This preserves the distinction between a specific geometric interoperability invariant and the broader agentic authorization decision.

## 8.18 Deterministic Reference Mode Is Required for Validation

v5 SHOULD include a controlled deterministic execution mode suitable for the experiments described in Section 7.

At minimum, deterministic mode should control:

- random-number generation;

- planner tie-breaking;

- swarm bid ordering;

- simulated tool outcomes;

- timestamps used in equality-sensitive receipts;

- deterministic identifiers where required for replay.

Wall-clock timestamps may still be recorded separately, but they should not prevent semantic trajectory comparison.

Given identical:

$$\left( I,X_{0},s,\Pi \right),$$

the governed decision trajectory should be reproducible where all external dependencies are simulated.

This gives v5 an executable reference surface that can later be compared against richer or nondeterministic deployments.

## 8.19 Suggested Reference Module Layout

The exact structure remains an implementation choice, but a minimal v5 organization could resemble:

```text
radial_membrane_ai/
    agentic/
        intent.py
        proposal.py
        admission.py
        execution.py
        verification.py
        residual.py
        reflection.py
        closure.py
        receipt.py
        runtime.py
```

The existing files for planning, tools, reflection, and swarm behavior should be modified or adapted rather than duplicated unnecessarily.

For example:

```text
planner.py
        → produces ActionProposal
tools.py
          → returns governed ExecutionRecord
reflection.py
     → produces candidate state change
swarm.py
          → selects candidate executor
```

New modules then provide the missing composition boundaries.

The architecture should remain understandable through the execution sequence rather than through the number of classes created.

## 8.20 Minimum v5 Implementation

To keep the release bounded, the **minimum conforming v5** should demonstrate one complete single-agent governed cycle.

That reference scenario should include:

1.  construction of an Intent Contract;

2.  generation of an Action Proposal;

3.  prospective admission;

4.  resource reservation;

5.  simulated or bounded tool execution;

6.  postcondition observation;

7.  independent verification;

8.  Agentic Residual formation;

9.  candidate reflection;

10. atomic state validation and commit;

11. provisional memory evidence;

12. governed memory decision;

13. Agentic Closure;

14. Action Receipt generation;

15. optional continuation into the next cycle.

Once this chain works correctly, additional planners, real integrations, and more complex multi-agent cases can be layered onto it.

The release does not need broad feature coverage to establish the model.

It needs one complete path whose semantics are correct.

## 8.21 v5 Must Not Attempt to Solve Every Remaining U.F.O. Research Question

Paper XII should remain disciplined about what the next release does **not** require.

v5 does not need to resolve:

- formal global asymptotic stability;

- a universal Lyapunov proof;

- globally optimal V-Channel routing;

- canonical calibration between every U.F.O. cost quantity;

- production network consensus;

- Byzantine fault tolerance;

- cryptographic shard attestation;

- generalized event-time freshness across distributed nodes;

- arbitrary external transaction rollback;

- formal semantic understanding;

- general artificial intelligence;

- consciousness;

- universal outcome verification;

- perfect planning.

Those remain separate research or systems problems.

Attempting to solve them inside v5 would obscure the specific hypothesis Paper XII is intended to test.

## 8.22 External Side Effects Remain a System Boundary

The reference architecture can govern whether an external action should be attempted and how resulting evidence should be interpreted.

It cannot guarantee that arbitrary external systems provide rollback, idempotency, correct reporting, or transaction semantics.

Therefore, v5 MUST distinguish between:

$$\text{internal atomicity}$$

and

$$\text{external atomicity}.$$

U.F.O. can guarantee that an invalid internal reflection does not partially mutate its own state.

It cannot automatically guarantee that a remote API, robot, database, financial service, or physical machine can undo a side effect once executed.

For this reason, external capabilities should expose their known execution semantics to the admission layer wherever practical.

## 8.23 Research Simulation Boundary

The initial v5 implementation remains a research-grade governed agentic reference system.

Passing the tests defined in Section 7 would establish evidence that the implementation preserves the stated software invariants under the tested scenarios.

It would not constitute proof of:

- general AI safety;

- safe deployment in arbitrary environments;

- correctness of third-party tools;

- hardware safety;

- regulatory compliance;

- security against every adversary;

- correctness of probabilistic model outputs.

The claim should remain narrower:

**U.F.O. v5 implements and falsification-tests an explicit governed agentic execution protocol over the hardened U.F.O. computational substrate.**

That claim is both meaningful and testable.

## 8.24 Acceptance Criteria for v5.0.0

The v5 release should not be considered complete merely because the new classes exist.

At minimum, the implementation should demonstrate all of the following:

**An unauthorized proposal never executes.**

**A changed proposal requires new admission when the change affects governed scope.**

**Predicted resource pressure is evaluated before execution.**

**Tool success does not bypass postcondition verification.**

**Unknown execution outcomes remain distinct from observed failure.**

**Irreversible unknown outcomes are not blindly retried.**

**Residual evidence remains structured and finite.**

**Rejected reflection leaves authoritative state unchanged.**

**Rejected memory promotion leaves durable memory unchanged.**

**An intermediate positive verdict cannot survive a later mandatory rejection as an authoritative commit.**

**Plan completion does not independently determine intent success.**

**Closure determines whether another cycle may begin.**

**A new cycle receives new prospective admission.**

**Action Receipts reconstruct the actual decision path.**

**Seeded deterministic runs reproduce the same governed trajectory under identical controlled inputs.**

**Variation across seeds does not alter hard governance rules.**

This is the operative definition of success for the release.

## 8.25 What v5 Is Intended to Prove

The implementation goal can now be stated precisely.

U.F.O. v5 is not intended to prove that U.F.O. is a general autonomous intelligence.

It is intended to demonstrate that the architecture developed before Paper XII already contains sufficient computational and governance machinery to support a coherent agentic execution cycle when the missing contracts are made explicit.

The implementation therefore tests the proposition:

$$\boxed{\text{Existing U.F.O. substrate} + \text{explicit agentic contracts} + \text{closure} \Rightarrow \text{governed recurrent agency}}$$

within the boundaries of the reference simulation.

The significance of v5 will depend less on how many additional capabilities are added than on whether this proposition survives implementation and falsification.

If it does, U.F.O. will have crossed an important engineering boundary.

The system will no longer merely contain a planner, tools, memory, reflection, temporal state, and governance mechanisms.

Those mechanisms will participate in one explicit cycle in which:

**intent constrains planning, governance precedes action, evidence follows action, verification precedes belief, state changes atomically, memory is earned, and continued autonomy requires closure.**

That is the implementation boundary for U.F.O. v5.0.0.

**Section 9 concludes the reference model by consolidating its engineering claim, implementation objective, and relationship to the U.F.O. research architecture.**

# 9. Conclusion and v5 Implementation Handoff

U.F.O. did not enter Paper XII without agentic capability.

Before this work, the architecture already contained planning, tool execution, bounded computation, reflection, temporal governance, semantic memory, multi-agent coordination, swarm selection, admissibility mechanisms, the Symmetric Ascension Operator, the Invariant Handshake, and the behavioral state represented by the radial membrane and Twelve Strings.

The unresolved engineering problem was different.

Those components did not yet participate in one explicit contract governing the complete lifetime of an autonomous action.

Paper XII addresses that composition problem through **Governed Agentic Closure**.

The resulting model defines agency not simply as the ability to plan and act, but as a recurrent sequence in which every consequential transition carries its own evidence and authority boundary:

$$\boxed{\begin{matrix}
\text{Intent} \rightarrow \text{Plan} \rightarrow \text{Admission} \rightarrow \text{Execution} \rightarrow \text{Observation} \cr
 \rightarrow \text{Verification} \rightarrow \text{Residual} \rightarrow \text{Reflection} \rightarrow \text{Memory} \rightarrow \text{Closure}
\end{matrix}}$$

This sequence is the principal architectural contribution of the paper.

Its purpose is not to make autonomous behavior maximally permissive.

Its purpose is to make autonomy **bounded, inspectable, reproducible where controlled, and explicit about when authority begins and ends**.

## 9.1 The Central Result

The core conclusion can be stated simply:

**U.F.O. already contains the principal mechanisms required for agentic AI. The missing element is an explicit closure contract that governs how those mechanisms transfer proposals, authority, evidence, state, memory, and permission across repeated action cycles.**

Paper XII defines that contract.

The resulting system separates concepts that are frequently collapsed in agentic architectures:

$$\text{Intent} \neq \text{Plan},$$

$$\text{Plan} \neq \text{Authorization},$$

$$\text{Selection} \neq \text{Authorization},$$

$$\text{Execution Success} \neq \text{Verified Outcome},$$

$$\text{Observation} \neq \text{Durable Memory},$$

$$\text{Reflection Proposal} \neq \text{Committed State},$$

and

$$\text{Action Completion} \neq \text{Permission to Continue}.$$

These separations are not merely semantic distinctions.

Each corresponds to a concrete software boundary that can be implemented, observed, and falsified.

## 9.2 Governed Agency as Repeated Permission

Traditional agent loops are often expressed approximately as:

$$\text{Goal} \rightarrow \text{Plan} \rightarrow \text{Act} \rightarrow \text{Repeat}.$$

Governed Agentic Closure changes the recurrence condition.

The system does not repeat merely because work remains.

Instead, each cycle ends by answering:

$$\boxed{\text{Given what just occurred, is another autonomous action still permissible?}}$$

The next action therefore depends upon the reconciled state produced by the previous one:

$$D_{t} = \mathcal{C}_{A}\left( I,X_{t + 1},\rho_{t},q_{t} \right).$$

Only an appropriate closure verdict permits another planning cycle.

Even then,

$$D_{t} = \text{CONTINUE}$$

does not directly authorize execution.

It authorizes only the next proposal:

$$D_{t} = \text{CONTINUE} \Rightarrow a_{t + 1}\mathcal{= P}\left( I,X_{t + 1} \right),$$

after which prospective admission must occur again:

$$V_{t + 1}^{-} = \mathcal{A}^{-}\left( a_{t + 1},I,X_{t + 1} \right).$$

Thus, authority is intentionally short-lived.

It is attached to a specific proposed action under a specific state and contract.

It does not propagate indefinitely through the loop.

This yields the central governing principle of the paper:

$$\boxed{\text{Continued autonomy is not assumed; it is re-earned after every governed action cycle.}}$$

## 9.3 BEDROCK to Closure

The model developed here follows directly from lessons exposed during the v4 BEDROCK hardening process.

BEDROCK demonstrated that reliable systems require distinctions between:

- candidate state and authoritative state;

- diagnostic metadata and admissibility evidence;

- preliminary decisions and final governing verdicts;

- attempted updates and committed updates;

- malformed data and valid numerical evidence.

Paper XII applies the same reasoning to agentic execution.

The corresponding agentic relationships are:

$$\text{proposal} \neq \text{permission},$$

$$\text{execution record} \neq \text{world-state truth},$$

$$\text{experience} \neq \text{memory},$$

and

$$\text{successful step} \neq \text{continued authority}.$$

This continuity is important.

Governed Agentic Closure is not a separate safety layer placed on top of U.F.O.

It is an extension of the same state-transition discipline that BEDROCK established lower in the architecture.

The v4 principle

$$\text{candidate} \rightarrow \text{validate} \rightarrow \text{commit}$$

becomes, at the agentic level,

$$\text{propose} \rightarrow \text{admit} \rightarrow \text{execute} \rightarrow \text{verify} \rightarrow \text{reconcile} \rightarrow \text{close}.$$

The scale changes.

The governing philosophy does not.

## 9.4 Relationship to the Invariant Handshake

Paper XII also places governed agentic execution within the broader U.F.O. interoperability model.

The Invariant Handshake already establishes a geometric permission boundary between scalar or legacy computation and the tensor-oriented U.F.O. side of the architecture.

For the handshake,

$$i = \frac{a^{2} + b^{2}}{c^{2}},$$

and stable interoperability requires both participating computational regimes to remain within the applicable bounded region around

$$i = 1.$$

Paper XII does not replace this relationship.

Instead, the governed execution state

$$X_{t}$$

may become the agentic participant whose proposed action encounters that existing boundary whenever legacy interoperability is required.

The first execution index,

$$t = 1,$$

therefore identifies the first governed agentic state in the recurrent sequence, while

$$i \approx 1$$

describes geometric closure for the relevant handshake.

These quantities are not algebraically equivalent.

Their connection is architectural.

One describes **where the agent is in its execution sequence**.

The other describes **whether a particular cross-regime computational relationship remains geometrically admissible**.

This allows the agentic model to remain anchored to U.F.O.'s existing computational geometry rather than becoming an independent orchestration framework.

## 9.5 The Role of Verification

One of the most consequential additions in the model is the explicit separation of execution from verification.

An executor may establish:

The requested operation was attempted and returned successfully.

It cannot necessarily establish:

The intended state of the world now exists.

That distinction is represented by

$$e_{t} \rightarrow o_{t} \rightarrow q_{t}.$$

The execution record describes the action.

The observation describes available evidence.

The verification result determines whether the expected postcondition is supported.

This is particularly important for autonomous systems interacting with:

- files;

- APIs;

- databases;

- distributed services;

- software environments;

- physical systems;

- human-operated systems;

- or other agents.

In each case, invocation success and outcome truth may diverge.

Paper XII therefore makes postcondition verification part of the governed cycle rather than an optional diagnostic.

## 9.6 The Role of Uncertainty

The architecture also treats uncertainty as a first-class state.

An external action may result in:

$$\text{OUTCOME\_UNKNOWN}.$$

That condition is deliberately distinct from:

$$\text{FAILED}.$$

This prevents the runtime from interpreting missing confirmation as proof that no external side effect occurred.

The distinction becomes especially important for irreversible or non-idempotent actions.

In such cases,

$$\text{OUTCOME\_UNKNOWN} \nRightarrow \text{automatic retry}.$$

Instead, uncertainty becomes residual evidence that may require observation, verification, constraint, or escalation.

The model therefore preserves ignorance rather than manufacturing certainty for the convenience of the control loop.

## 9.7 The Role of Memory

The same discipline applies to persistent knowledge.

An agent experiences more information than it should necessarily remember as authoritative fact.

Accordingly,

$$\text{Observation} \rightarrow \text{Provisional Evidence} \rightarrow \text{Qualification} \rightarrow \text{Promotion} \rightarrow \text{Durable Memory}.$$

This path connects agentic execution to U.F.O.'s existing temporal-governance, semantic-memory, residual, and promotion mechanisms.

It also prevents one erroneous tool response or transient observation from immediately changing the long-term knowledge available to subsequent decisions.

The governing principle remains:

$$\boxed{\text{Memory is a governed consequence of experience.}}$$

## 9.8 The Role of Reflection

Reflection is similarly constrained.

The system may reason about its own behavior and propose adjustments based on residual evidence:

$$\Delta X_{t}^{\ast}\mathcal{= F}\left( \rho_{t},X_{t} \right).$$

But reflective reasoning does not receive privileged write access.

The proposal must still become a valid candidate state and pass the relevant state-validation boundary:

$$X_{t} \rightarrow X_{t + 1}^{\ast} \rightarrow \mathrm{Validate} \rightarrow X_{t + 1}.$$

If the candidate is invalid,

$$X_{t + 1} = X_{t}.$$

Thus adaptation remains possible without sacrificing atomic state semantics.

## 9.9 The Role of Multi-Agent Systems

The same architecture extends naturally to U.F.O.'s swarm and multi-agent mechanisms.

Selection answers:

Which agent appears best suited to perform this work?

Admission answers:

Is the specific action proposed by that selected agent permitted?

These are different questions.

Therefore,

$$\mathrm{Select}\left( U_{i} \right) \nRightarrow \mathrm{Authorize}\left( a_{t} \right).$$

Delegation may narrow authority.

It must not silently expand it.

This allows more sophisticated collective behavior to develop without requiring the system to abandon the single-agent governance model established in this paper.

## 9.10 What Paper XII Does Not Claim

The contribution should be interpreted within its stated limits.

Paper XII does not establish:

- general artificial intelligence;

- consciousness;

- perfect planning;

- universal semantic correctness;

- arbitrary external rollback;

- globally optimal action selection;

- universal AI safety;

- formal correctness of every U.F.O. subsystem;

- complete distributed consensus;

- Byzantine resilience;

- guaranteed hardware safety;

- guaranteed correctness of external tools;

- or formally proven autonomous safety in unrestricted environments.

Likewise, deterministic seeded experiments do not imply that arbitrary real-world environments are deterministic.

The intended claim is narrower and experimentally accessible:

**U.F.O. can implement an explicit recurrent agentic protocol in which proposals, authority, execution, evidence, adaptation, memory, and continuation remain distinct governed transitions.**

That is sufficient to make the architecture meaningfully testable.

## 9.11 v5.0.0 Implementation Handoff

Paper XII is intended to lead directly into implementation.

The v5 engineering task can therefore begin from the following ordered objective:

1. Define IntentContract.
2. Define ActionProposal.
3. Insert prospective agentic admission before tool execution.
4. Add prospective resource reservation.
5. Represent execution uncertainty explicitly.
6. Add independent postcondition verification.
7. Construct the Agentic Residual Vector.
8. Convert reflection to candidate → validate → atomic commit.
9. Route execution evidence through provisional memory qualification.
10. Implement AgenticClosure.
11. Generate an AgenticActionReceipt for every completed cycle.
12. Require new admission for every subsequent action.
13. Integrate the same semantics with swarm-selected agents.
14. Add deterministic reference execution.
15. Attempt to falsify every mandatory invariant.
That sequence should be implemented incrementally.

The first milestone should not be a complicated autonomous demonstration.

It should be the smallest complete governed cycle:

$$I \rightarrow a_{1} \rightarrow V_{1}^{-} \rightarrow e_{1} \rightarrow o_{1} \rightarrow q_{1} \rightarrow \rho_{1} \rightarrow X_{2} \rightarrow D_{1} \rightarrow R_{1}^{A}.$$

Once that path is correct, recurrence can be enabled:

$$R_{1}^{A} \rightarrow R_{2}^{A} \rightarrow \cdots \rightarrow R_{N}^{A}.$$

Only after the single-agent path is validated should the same closure semantics be exercised through multi-agent and swarm execution.

This ordering minimizes the number of moving components during initial falsification.

## 9.12 Definition of a Successful v5 Release

A successful v5 release is therefore not defined by how many tools the agent can invoke or by how long it can operate without human involvement.

It is defined by whether the execution chain remains governed when conditions become unfavorable.

A strong v5 implementation should be able to demonstrate cases where it correctly:

- acts;

- refuses to act;

- constrains an action;

- replans;

- detects an incorrect outcome;

- preserves uncertainty;

- rejects an invalid reflection;

- refuses memory promotion;

- prevents an unsafe retry;

- exhausts its budget;

- requests escalation;

- terminates successfully;

- terminates unsuccessfully;

- and begins another cycle only after closure and renewed admission.

Some of the strongest demonstrations of the architecture should therefore be cases in which the agent **does less**.

Correct refusal is a behavior.

Correct uncertainty is a state.

Correct termination is an outcome.

## 9.13 Final Engineering Proposition

The architectural proposition tested by v5 can now be written as

$$\boxed{\mathcal{U}_{4} + \mathcal{K}_{A} + \mathcal{C}_{A} \rightarrow \mathcal{U}_{5}}$$

where

$$\mathcal{U}_{4} = \text{hardened U.F.O. v4 substrate},$$

$$\mathcal{K}_{A} = \text{agentic contracts defined by Paper XII},$$

$$\mathcal{C}_{A} = \text{Agentic Closure Operator},$$

and

$$\mathcal{U}_{5} = \text{governed recurrent agentic reference architecture}.$$

The arrow does not represent a mathematical proof of safety.

It represents an engineering hypothesis.

The implementation and falsification program determine whether that hypothesis survives contact with executable behavior.

If a mandatory invariant fails, the architecture must be corrected.

If the implementation cannot reproduce the paper's declared ordering, the implementation is incomplete.

If the code reveals an invalid assumption in the paper, the paper should be revised.

This creates a bidirectional relationship:

$$\text{Paper} \rightarrow \text{Invariant} \rightarrow \text{Implementation} \rightarrow \text{Falsification} \rightarrow \text{Revision}.$$

That relationship is itself part of the U.F.O. engineering methodology.

## 9.14 Governed Adaptive Recurrence: Closure as a Feedback Boundary

Governed Agentic Closure is recurrent by design. The reference state machine already returns `CONTINUE` to planning, and the model requires every subsequent action to receive renewed prospective admission. The implementation path developed from this paper suggests a stronger interpretation: **Closure should be understood not merely as the endpoint of an action transaction, but as the governed feedback boundary between consecutive agentic states.**

The state presented to the next planning step should therefore be the reconciled state produced by the previous cycle, not a stale copy of the state that existed before execution. A useful feedback context may be represented as

$$
Z_{t+1} = \left(X_{t+1}, \rho_t, M_{t+1}, B_{t+1}, \Pi_{t+1}, D_t\right),
$$

where $X_{t+1}$ denotes the resulting authoritative U.F.O. state, $\rho_t$ the unresolved residual evidence, $M_{t+1}$ the qualified durable-memory state, $B_{t+1}$ the remaining governed budget, $\Pi_{t+1}$ the active validated plan state or revision, and $D_t$ the closure decision from the completed cycle. These quantities do not all confer authority. Together they describe what the system has verifiably become after the consequences of the previous action have been reconciled.

The next proposal may then be generated from the intent and this updated feedback context:

$$
a_{t+1} = P(I, Z_{t+1}).
$$

This richer planning context does not weaken the separation between planning and authorization. The next proposal remains only a candidate and must pass a new prospective admission decision against the current authoritative state:

$$
V_{t+1}^{-} = \mathcal{A}^{-}(a_{t+1}, I, X_{t+1}).
$$

**Closure terminates the authority of cycle $t$; it does not terminate information flow through the architecture.**

This distinction turns recurrence into a closed governed feedback loop. Execution changes the world or produces evidence about it; verification and residual formation characterize the consequence; reflection may alter validated internal state; memory governance determines what becomes durable; resource reconciliation changes the remaining envelope; and Closure determines whether those reconciled results may participate in another planning cycle.

$$Z_t \rightarrow \text{Proposal} \rightarrow \text{Admission} \rightarrow \text{Action} \rightarrow \text{Evidence} \rightarrow \text{Reconciliation} \rightarrow \text{Closure} \rightarrow Z_{t+1} \rightarrow \text{Proposal}.$$

Informally, this can be interpreted as an expansion-contraction rhythm. Planning, admission, and execution push a bounded candidate action outward toward an environment. Observation, verification, residual formation, reflection, memory, and Closure draw the consequences back into authoritative state. The next cycle begins from that changed state. This interpretation is architectural rather than biological; it does not imply consciousness, organic behavior, or continuous self-modification.

The adaptive part of the loop is likewise narrower than arbitrary source-code mutation. U.F.O. may deform its membrane state, accumulate or resolve residuals, qualify durable memory, consume budget, revise a plan, and change the context used to produce a subsequent proposal. Those are governed runtime-state transitions. The governing contracts themselves do not become writable merely because the agent is permitted to adapt.

Closed recurrence also makes efficiency an explicit architectural question. For a governed agent, efficiency should not be defined only as action frequency, token throughput, or the number of completed plan steps. A system that refuses an unnecessary or poorly evidenced action may be more efficient than one that acts aggressively and later spends additional resources correcting residuals.

One optional reference diagnostic for a controlled implementation may therefore relate verified progress toward the active intent to the cost and unresolved consequences of obtaining that progress:

$$
\eta_t =
\frac{\Delta G_t}
{\varepsilon + C_t + \lambda_1 \lVert \rho_t \rVert + \lambda_2 U_t + \lambda_3 S_t},
$$

where $\Delta G_t$ denotes verified goal-relevant progress, $C_t$ observed governed resource expenditure, $\lVert \rho_t \rVert$ a declared residual magnitude for the applicable residual domain, $U_t$ unresolved execution or verification uncertainty, $S_t$ governed side-effect exposure, $\varepsilon > 0$ a stabilizing constant, and $\lambda_1$, $\lambda_2$, and $\lambda_3$ explicit non-negative weights. This expression is not proposed as a universal utility function. Different workloads may require different domains, scales, or no scalar efficiency summary at all.

Most importantly, **efficiency remains diagnostic**. It may inform planning, route selection, decomposition, or a decision to replan, but it does not authorize execution.

$$
\eta_t \nRightarrow \operatorname{Authorize}(a_{t+1}).
$$

A high-efficiency candidate still requires fresh prospective admission. A low-efficiency path may remain authorized if it is the only admissible route to a required objective. This preserves the BEDROCK-derived rule that a useful diagnostic does not silently become an authority source.

A conforming closed-loop implementation should therefore preserve the following feedback properties:

- the next proposal is derived from post-cycle authoritative state rather than stale pre-execution state;
- residual and memory evidence may shape later planning only through their declared, validated contracts;
- remaining budget carries forward and is never reset merely because another cycle begins;
- runtime adaptation may change strategy or state but must not silently widen the Intent Contract, delegated authority, or permitted side-effect envelope;
- efficiency or optimization signals may influence proposal generation but never substitute for admission;
- and every consequential next action receives a fresh proposal identity, state binding, resource reservation, and admission verdict.

This refinement does not replace Governed Agentic Closure. It clarifies what Closure closes. Closure closes the authority-bearing transaction for the current cycle. When continuation or governed replanning is permitted, it opens only a feedback path from the reconciled state into a new proposal. The architecture therefore becomes more than a sequence that repeats: **it becomes a bounded adaptive recurrence in which verified consequences can reshape what the system proposes next without allowing adaptation to self-authorize.**

## 9.15 Future Direction: Governed Machine-Learning Compartments

Governed Adaptive Recurrence establishes a bounded feedback architecture in which U.F.O. can act, reconcile consequences, adapt authoritative state, and generate subsequent proposals without allowing adaptation to inherit execution authority. A natural future direction is to extend this recurrent substrate toward **compartmentalized machine-learning computation at the upper boundaries of the V-Channel architecture**. Rather than embedding one monolithic learning system into the U.F.O. core, specialized machine-learning capabilities may be attached to governed V-Channel interfaces, allowing distinct functions such as classification, prediction, anomaly detection, perception, language processing, control, or optimization to remain computationally specialized while participating in a common governance substrate.

Conceptually, a V-Channel $V_k$ may expose a bounded interface to a machine-learning compartment $\mathcal{M}_k$:

$$
V_k \rightarrow \mathcal{M}_k \rightarrow E_k \rightarrow V_k^{-1},
$$

where $E_k$ represents the evidence returned by the compartment rather than an automatically authoritative conclusion. Such evidence may include model output, uncertainty, provenance, resource consumption, confidence information, or other declared diagnostic state. The architectural distinction established throughout Paper XII would remain intact:

$$
\text{Model Output} \neq \text{Verified Evidence},
$$

$$
\text{Model Confidence} \neq \text{Authorization},
$$

$$
\text{Model Selection} \neq \text{Execution Authority}.
$$

A machine-learning compartment would therefore remain subordinate to the same proposal, admission, verification, residual, memory, and Closure semantics established for governed agentic execution. Learned computation may influence what U.F.O. proposes or how it interprets evidence, but it would not inherit global authority merely because a model produced a high-confidence result.

This approach also creates a potential relationship between U.F.O.'s deformable geometry and learned computation. The authoritative state $X_t$, current intent $I$, residual evidence $\rho_t$, and resource state $B_t$ may eventually influence which V-Channel or specialized learning compartment receives computational attention:

$$
k^{*}=R_V(I,X_t,\rho_t,B_t),
$$

followed by

$$
E_t=\mathcal{M}_{k^{*}}(x_t).
$$

The resulting evidence would then return through the governed recurrent architecture rather than bypass it:

$$X_t \rightarrow R_V \rightarrow \mathcal{M}_{k^{*}} \rightarrow E_t \rightarrow \rho_t \rightarrow X_{t+1}.$$

This future direction would preserve a central U.F.O. design principle: **specialization should occur at declared computational boundaries without duplicating or weakening the governing substrate**. Different machine-learning compartments could employ different internal techniques or model families while exposing a common bounded contract to U.F.O. The radial membrane and V-Channels would therefore remain model-agnostic at the architectural core while providing governed pathways through which specialized learned computation may participate.

This extension is deliberately left outside the implementation requirements of U.F.O. v5.0.0. It represents a possible subsequent research direction rather than a claim of functionality established by Paper XII. The immediate purpose of v5 remains the implementation and falsification of Governed Agentic Closure and adaptive recurrence. Once that foundation has been experimentally exercised, compartmentalized learned computation provides one natural path for investigating how the same governance laws behave when coupled to increasingly specialized machine-learning systems.

## 9.16 Conclusion

Agentic systems become difficult to govern precisely where individually reasonable components begin interacting.

A planner may be correct about what should happen next but wrong about what it is permitted to do.

A tool may execute successfully without producing the intended result.

An observation may be genuine but insufficiently verified.

A reflective adjustment may be useful but invalid as authoritative state.

A remembered event may be informative but unsafe to promote.

A completed action may move the system forward while simultaneously eliminating its authority to continue.

Governed Agentic Closure makes those seams explicit.

U.F.O. v5.0.0 therefore begins from a simple premise:

**Agency is not merely the ability to choose and execute actions. Governed agency requires the system to preserve the conditions under which another action remains justified.**

The resulting architecture does not ask only:

$$\text{What should the agent do next?}$$

It asks:

$$\text{What was authorized?}$$

$$\text{What actually happened?}$$

$$\text{What evidence supports that conclusion?}$$

$$\text{What changed because of it?}$$

$$\text{What may be remembered?}$$

and finally:

$$\boxed{\text{Is the system still permitted to act again?}}$$

That final question is the closure condition.

It is also the transition from U.F.O. v4's hardened computational substrate to the governed agentic architecture proposed for U.F.O. v5.0.0.
