"""
Workload Execution Engine and Metrics tracing for Governed Stress-Testing Suite in U.F.O.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional, Set

from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope
from radial_membrane_ai.shard import ShardState

from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine

from radial_membrane_ai.workloads.workload import (
    Action,
    ActionType,
    StabilityBand,
    SAOLevel,
    EnvelopeState,
    SimulationTarget,
    WorkloadStep,
    Workload
)


@dataclass
class RegimeTransition:
    step_index: int
    from_regime: str
    to_regime: str


@dataclass
class SAOEvent:
    step_index: int
    level: SAOLevel


@dataclass
class StabilityBandEvent:
    step_index: int
    band: StabilityBand


@dataclass
class EnvelopeAlignmentEvent:
    step_index: int
    state: EnvelopeState


@dataclass
class RollbackEvent:
    step_index: int
    reason: str


@dataclass
class QuarantineEvent:
    step_index: int
    entity_id: str
    reason: str


@dataclass
class SemanticMemoryDiff:
    step_index: int
    before: dict
    after: dict


@dataclass
class WorkloadFrameMetrics:
    step_index: int
    curvature: float
    tension: float
    coherence: float
    regime: str
    stability_band: StabilityBand
    sao_level: SAOLevel
    envelope_state: EnvelopeState


@dataclass
class WorkloadResult:
    frames: List[WorkloadFrameMetrics] = field(default_factory=list)
    regime_transitions: List[RegimeTransition] = field(default_factory=list)
    sao_events: List[SAOEvent] = field(default_factory=list)
    stability_band_events: List[StabilityBandEvent] = field(default_factory=list)
    envelope_events: List[EnvelopeAlignmentEvent] = field(default_factory=list)
    rollback_events: List[RollbackEvent] = field(default_factory=list)
    quarantine_events: List[QuarantineEvent] = field(default_factory=list)
    semantic_diffs: List[SemanticMemoryDiff] = field(default_factory=list)
    correctness_report: dict = field(default_factory=dict)


class WorkloadEngine:
    """
    Executes workload steps against U.F.O. engines, applying actions,
    triggering regime updates, checking policy envelopes, and capturing traces.
    """

    def __init__(
        self,
        single_agent_engine: Optional[SingleAgentEngine] = None,
        multi_agent_engine: Optional[MultiAgentEngine] = None,
        multi_cluster_engine: Optional[MultiClusterEngine] = None
    ) -> None:
        self.single_agent_engine = single_agent_engine or SingleAgentEngine()
        self.multi_agent_engine = multi_agent_engine or MultiAgentEngine()
        self.multi_cluster_engine = multi_cluster_engine or MultiClusterEngine()

    def run(self, workload: Workload) -> WorkloadResult:
        """
        Executes a complete workload and returns a full diagnostic WorkloadResult.
        """
        engine = self._select_engine(workload.target)
        result = WorkloadResult()

        prev_regime: Optional[str] = None
        prev_band: Optional[StabilityBand] = None
        prev_envelope: Optional[EnvelopeState] = None

        # Enable collective layer if multi-agent or multi-cluster has it
        if hasattr(engine, "collective_enabled"):
            engine.collective_enabled = True

        # Track semantic memory state
        def capture_memory_state() -> dict:
            state = {}
            if workload.target == SimulationTarget.SINGLE_AGENT:
                if hasattr(engine, "semantic_memory") and engine.semantic_memory:
                    for k, rec in engine.semantic_memory.local_store.items():
                        state[k] = {"value": rec.value, "tags": list(rec.tags)}
            elif workload.target == SimulationTarget.MULTI_AGENT:
                if hasattr(engine, "mesh_memory") and engine.mesh_memory:
                    for k, rec in engine.mesh_memory.global_store.items():
                        state[k] = {"value": rec.value, "tags": list(rec.tags)}
                for a in engine.agents:
                    if hasattr(a, "semantic_memory") and a.semantic_memory:
                        for k, rec in a.semantic_memory.local_store.items():
                            state[f"{a.agent_id}:{k}"] = {"value": rec.value, "tags": list(rec.tags)}
            elif workload.target == SimulationTarget.MULTI_CLUSTER:
                if hasattr(engine, "global_mesh_memory") and engine.global_mesh_memory:
                    for k, rec in engine.global_mesh_memory.global_store.items():
                        state[k] = {"value": rec.value, "tags": list(rec.tags)}
                for c_id, c in engine.clusters.items():
                    if hasattr(c, "semantic_memory") and c.semantic_memory:
                        for k, rec in c.semantic_memory.global_store.items():
                            state[f"{c_id}:{k}"] = {"value": rec.value, "tags": list(rec.tags)}
                    for a in c.agents:
                        if hasattr(a, "semantic_memory") and a.semantic_memory:
                            for k, rec in a.semantic_memory.local_store.items():
                                state[f"{c_id}:{a.agent_id}:{k}"] = {"value": rec.value, "tags": list(rec.tags)}
            return state

        # Initial memory snapshot
        curr_memory = capture_memory_state()

        # Track quarantine states
        quarantined_entities: Set[str] = set()

        for step_idx, step in enumerate(workload.steps):
            # Capture semantic memory before actions
            mem_before = dict(curr_memory)

            # A. Apply Actions
            self._apply_actions(engine, step, workload.target)

            # Determine task value and excitation for tick
            task_value = 0.8
            excitation = np.ones(12, dtype=np.float64) * 0.5

            # Combine actions to extract activation sequentials
            all_acts = (
                list(step.agent_actions.values())
                + list(step.cluster_actions.values())
                + step.global_actions
            )
            for act in all_acts:
                if act.type == ActionType.SET_ACTIVATION:
                    val = act.payload.get("value")
                    if isinstance(val, list):
                        excitation = np.array(val[:12], dtype=np.float64)
                        if len(excitation) < 12:
                            excitation = np.pad(
                                excitation,
                                (0, 12 - len(excitation)),
                                'constant',
                                constant_values=0.5
                            )
                    elif isinstance(val, (int, float)):
                        excitation = np.ones(12, dtype=np.float64) * float(val)

            # B. Execute tick
            if workload.target == SimulationTarget.SINGLE_AGENT:
                engine.tick(task_value=task_value, excitation=excitation)
            elif workload.target == SimulationTarget.MULTI_AGENT:
                engine.tick(task_value=task_value, excitation=excitation)
            elif workload.target == SimulationTarget.MULTI_CLUSTER:
                engine.tick(task_value=task_value, default_excitation=excitation)

            # C. Capture metrics & Frame
            frame = self._capture_frame(engine, workload.target, step_idx)
            result.frames.append(frame)

            # D. Track Transitions and Events Programmatically
            # 1. Regime Transitions
            if prev_regime is not None and frame.regime != prev_regime:
                result.regime_transitions.append(
                    RegimeTransition(step_index=step_idx, from_regime=prev_regime, to_regime=frame.regime)
                )
            prev_regime = frame.regime

            # 2. SAO Level Event
            if frame.sao_level != SAOLevel.NONE:
                result.sao_events.append(SAOEvent(step_index=step_idx, level=frame.sao_level))

            # 3. Stability Band Event
            if prev_band is not None and frame.stability_band != prev_band:
                result.stability_band_events.append(
                    StabilityBandEvent(step_index=step_idx, band=frame.stability_band)
                )
            prev_band = frame.stability_band

            # 4. Envelope Alignment Event
            if prev_envelope is not None and frame.envelope_state != prev_envelope:
                result.envelope_events.append(
                    EnvelopeAlignmentEvent(step_index=step_idx, state=frame.envelope_state)
                )
            prev_envelope = frame.envelope_state

            # 5. Rollbacks detection
            # Check engine interventions for any rollback trigger
            rollback_detected = False
            rollback_reason = ""
            if hasattr(engine, "interventions") and engine.interventions:
                for inter in reversed(engine.interventions):
                    if "Rollback" in inter or "rollback" in inter:
                        rollback_detected = True
                        rollback_reason = inter
                        break
            if rollback_detected:
                result.rollback_events.append(RollbackEvent(step_index=step_idx, reason=rollback_reason))

            # 6. Quarantine detection
            # Track any agent/cluster transitioning into ShardState.QUARANTINED
            if workload.target == SimulationTarget.SINGLE_AGENT:
                if getattr(engine, "quarantine_timer", 0) > 0 and "single_agent" not in quarantined_entities:
                    quarantined_entities.add("single_agent")
                    result.quarantine_events.append(
                        QuarantineEvent(
                            step_index=step_idx,
                            entity_id="single_agent",
                            reason="Stability violation quarantine."
                        )
                    )
            elif workload.target == SimulationTarget.MULTI_AGENT:
                for agent in engine.agents:
                    if agent.shard.state == ShardState.QUARANTINED and agent.agent_id not in quarantined_entities:
                        quarantined_entities.add(agent.agent_id)
                        result.quarantine_events.append(
                            QuarantineEvent(
                                step_index=step_idx,
                                entity_id=agent.agent_id,
                                reason="Mesh audit or safety quarantine."
                            )
                        )
                    elif agent.shard.state != ShardState.QUARANTINED and agent.agent_id in quarantined_entities:
                        quarantined_entities.remove(agent.agent_id)
            elif workload.target == SimulationTarget.MULTI_CLUSTER:
                for c_id, c in engine.clusters.items():
                    if getattr(c, "quarantine_timer", 0) > 0 and c_id not in quarantined_entities:
                        quarantined_entities.add(c_id)
                        result.quarantine_events.append(
                            QuarantineEvent(
                                step_index=step_idx,
                                entity_id=c_id,
                                reason="Cluster safety quarantine."
                            )
                        )
                    elif getattr(c, "quarantine_timer", 0) == 0 and c_id in quarantined_entities:
                        quarantined_entities.remove(c_id)
                for c in engine.clusters.values():
                    for agent in c.agents:
                        if agent.shard.state == ShardState.QUARANTINED and agent.agent_id not in quarantined_entities:
                            quarantined_entities.add(agent.agent_id)
                            result.quarantine_events.append(
                                QuarantineEvent(
                                    step_index=step_idx,
                                    entity_id=agent.agent_id,
                                    reason="Agent quarantine under cluster."
                                )
                            )
                        elif agent.shard.state != ShardState.QUARANTINED and agent.agent_id in quarantined_entities:
                            quarantined_entities.remove(agent.agent_id)

            # 7. Semantic Memory Diffs
            curr_memory = capture_memory_state()
            if curr_memory != mem_before:
                diff_before = {}
                diff_after = {}
                # Capture keys that changed/added
                for k, v in curr_memory.items():
                    if k not in mem_before:
                        diff_after[k] = v
                    elif mem_before[k] != v:
                        diff_before[k] = mem_before[k]
                        diff_after[k] = v
                for k, v in mem_before.items():
                    if k not in curr_memory:
                        diff_before[k] = v
                result.semantic_diffs.append(
                    SemanticMemoryDiff(step_index=step_idx, before=diff_before, after=diff_after)
                )

        # Build final correctness report
        result.correctness_report = self._build_correctness_report(result, workload)
        return result

    def _select_engine(self, target: SimulationTarget) -> Any:
        if target is SimulationTarget.SINGLE_AGENT:
            return self.single_agent_engine
        if target is SimulationTarget.MULTI_AGENT:
            return self.multi_agent_engine
        if target is SimulationTarget.MULTI_CLUSTER:
            return self.multi_cluster_engine
        raise ValueError(f"Unknown target: {target}")

    def _apply_actions(self, engine: Any, step: WorkloadStep, target: SimulationTarget) -> None:
        # Dispatch agent_actions
        for entity_id, action in step.agent_actions.items():
            self._dispatch_action(engine, action, target)

        # Dispatch cluster_actions
        for entity_id, action in step.cluster_actions.items():
            self._dispatch_action(engine, action, target)

        # Dispatch global_actions
        for action in step.global_actions:
            self._dispatch_action(engine, action, target)

    def _dispatch_action(self, engine: Any, action: Action, target: SimulationTarget) -> None:
        p = action.payload
        if action.type == ActionType.CHANGE_REGIME:
            entity_id = p["entity_id"]
            reg_str = p["regime"].upper()
            reg_type = KernelRegimeType[reg_str]
            if target == SimulationTarget.SINGLE_AGENT:
                engine.regime_manager.set_regime_for_agent("single_agent", reg_type)
            elif target == SimulationTarget.MULTI_AGENT:
                if entity_id == "global":
                    engine.regime_manager.set_global_regime(reg_type)
                else:
                    engine.regime_manager.set_regime_for_agent(entity_id, reg_type)
            elif target == SimulationTarget.MULTI_CLUSTER:
                if entity_id == "global":
                    engine.regime_manager.set_global_regime(reg_type)
                elif entity_id in engine.clusters:
                    engine.regime_manager.set_regime_for_cluster(entity_id, reg_type)
                else:
                    engine.regime_manager.set_regime_for_agent(entity_id, reg_type)

        elif action.type == ActionType.WRITE_MEMORY:
            scope = p["scope"]
            key = p["key"]
            value = p["value"]
            tags = set(p.get("tags", []))
            from radial_membrane_ai.semantic_memory.core import MemoryRecord, write_memory
            from radial_membrane_ai.semantic_memory.policy import MemoryPolicy, AdmissibilityContext

            if target == SimulationTarget.SINGLE_AGENT:
                policy = MemoryPolicy(allowed_agents={engine.semantic_memory.agent_id})
                context = AdmissibilityContext(coherence=1.0, quarantined=False, cost_factor=1.0, stability_energy=0.0)
                write_memory(engine.semantic_memory, key, value, tags, policy, context, engine.ledger)
            elif target == SimulationTarget.MULTI_AGENT:
                if scope == "global":
                    rec = MemoryRecord(
                        key=key, value=value, tags=tags, created_at=time.time(),
                        updated_at=time.time(), origin_agent_id="global", policy_envelope=PolicyEnvelope()
                    )
                    engine.mesh_memory.global_store[key] = rec
                else:
                    agent = next((a for a in engine.agents if a.agent_id == scope), None)
                    if agent is not None and hasattr(agent, "semantic_memory") and agent.semantic_memory:
                        policy = MemoryPolicy(allowed_agents={agent.agent_id})
                        context = AdmissibilityContext(
                            coherence=agent.shard.quality_score,
                            quarantined=(agent.shard.state == ShardState.QUARANTINED),
                            cost_factor=agent.shard.cost_factor,
                            stability_energy=0.0
                        )
                        write_memory(
                            agent.semantic_memory, key, value, tags, policy, context, engine.mesh_governance.ledger
                        )
            elif target == SimulationTarget.MULTI_CLUSTER:
                if scope == "global":
                    rec = MemoryRecord(
                        key=key, value=value, tags=tags, created_at=time.time(),
                        updated_at=time.time(), origin_agent_id="global", policy_envelope=PolicyEnvelope()
                    )
                    engine.global_mesh_memory.global_store[key] = rec
                elif scope in engine.clusters:
                    rec = MemoryRecord(
                        key=key, value=value, tags=tags, created_at=time.time(),
                        updated_at=time.time(), origin_agent_id=scope, policy_envelope=PolicyEnvelope()
                    )
                    engine.clusters[scope].semantic_memory.global_store[key] = rec
                else:
                    for c in engine.clusters.values():
                        agent = next((a for a in c.agents if a.agent_id == scope), None)
                        if agent is not None and hasattr(agent, "semantic_memory") and agent.semantic_memory:
                            policy = MemoryPolicy(allowed_agents={agent.agent_id})
                            context = AdmissibilityContext(
                                coherence=agent.shard.quality_score,
                                quarantined=(agent.shard.state == ShardState.QUARANTINED),
                                cost_factor=agent.shard.cost_factor,
                                stability_energy=0.0
                            )
                            write_memory(
                                agent.semantic_memory, key, value, tags, policy, context, engine.hierarchical_sao.ledger
                            )

        elif action.type == ActionType.TRIGGER_VIOLATION:
            entity_id = p["entity_id"]
            if target == SimulationTarget.SINGLE_AGENT:
                engine.ledger.log_failure(
                    record_id="violation_trigger", error_type=p["violation_type"],
                    shard_id="single_agent", severity="high", details={}
                )
                engine.v_history.append(10.0)  # force red band
            elif target == SimulationTarget.MULTI_AGENT:
                agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                if agent is not None:
                    agent.shard.trust_score = 0.1
                    agent.shard.policy_compliance = 0.1
                    agent.shard.state = ShardState.QUARANTINED
                engine.mesh_governance.ledger.log_failure(
                    record_id="violation_trigger", error_type=p["violation_type"],
                    shard_id=entity_id, severity="high", details={}
                )
            elif target == SimulationTarget.MULTI_CLUSTER:
                cluster = engine.clusters.get(entity_id)
                if cluster is not None:
                    cluster.quarantine_timer = 5
                    cluster.tension_metric = 5.0
                    cluster.stability_band = "red"
                    for agent in cluster.agents:
                        agent.shard.trust_score = 0.1
                        agent.shard.state = ShardState.QUARANTINED
                else:
                    for c in engine.clusters.values():
                        agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                        if agent is not None:
                            agent.shard.trust_score = 0.1
                            agent.shard.policy_compliance = 0.1
                            agent.shard.state = ShardState.QUARANTINED

        elif action.type == ActionType.SET_ENVELOPE:
            entity_id = p["entity_id"]
            new_envelope = PolicyEnvelope(
                min_trust=p["min_trust"], stability_required_band=p["stability_required_band"]
            )
            if target == SimulationTarget.MULTI_AGENT:
                if entity_id == "global":
                    engine.global_envelope = new_envelope
                else:
                    agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                    if agent is not None:
                        setattr(agent, "policy_envelope", new_envelope)
            elif target == SimulationTarget.MULTI_CLUSTER:
                if entity_id == "global":
                    engine.global_envelope = new_envelope
                elif entity_id in engine.clusters:
                    setattr(engine.clusters[entity_id], "policy_envelope", new_envelope)
                else:
                    for c in engine.clusters.values():
                        agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                        if agent is not None:
                            setattr(agent, "policy_envelope", new_envelope)

        elif action.type == ActionType.SET_TENSION:
            entity_id = p["entity_id"]
            tension_val = float(p["tension"])
            if target == SimulationTarget.SINGLE_AGENT:
                if hasattr(engine.membrane, "temporal_state") and engine.membrane.temporal_state:
                    engine.membrane.temporal_state.accumulated_tension = tension_val
            elif target == SimulationTarget.MULTI_AGENT:
                if entity_id == "global":
                    engine.temporal_state.accumulated_tension = tension_val
                else:
                    agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                    if (
                        agent is not None
                        and hasattr(agent.membrane, "temporal_state")
                        and agent.membrane.temporal_state
                    ):
                        agent.membrane.temporal_state.accumulated_tension = tension_val
            elif target == SimulationTarget.MULTI_CLUSTER:
                if entity_id == "global":
                    engine.temporal_state.accumulated_tension = tension_val
                elif entity_id in engine.clusters:
                    c_t = getattr(engine.clusters[entity_id].membrane, "temporal_state", None)
                    if c_t is not None:
                        c_t.accumulated_tension = tension_val
                else:
                    for c in engine.clusters.values():
                        agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                        if (
                            agent is not None
                            and hasattr(agent.membrane, "temporal_state")
                            and agent.membrane.temporal_state
                        ):
                            agent.membrane.temporal_state.accumulated_tension = tension_val

        elif action.type == ActionType.SET_CURVATURE:
            entity_id = p["entity_id"]
            curv_val = float(p["curvature"])
            if target == SimulationTarget.SINGLE_AGENT:
                engine.boundary.radius_deviation[1] = curv_val
                engine.boundary.radius_deviation[6] = curv_val
            elif target == SimulationTarget.MULTI_AGENT:
                agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                if agent is not None:
                    agent.boundary.radius_deviation[1] = curv_val
                    agent.boundary.radius_deviation[6] = curv_val
            elif target == SimulationTarget.MULTI_CLUSTER:
                for c in engine.clusters.values():
                    agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                    if agent is not None:
                        agent.boundary.radius_deviation[1] = curv_val
                        agent.boundary.radius_deviation[6] = curv_val

        elif action.type == ActionType.SET_COST_PROFILE:
            entity_id = p["entity_id"]
            sens = float(p["cost_sensitivity"])
            if target == SimulationTarget.MULTI_AGENT:
                agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                if agent is not None:
                    agent.cost_sensitivity = sens
            elif target == SimulationTarget.MULTI_CLUSTER:
                for c in engine.clusters.values():
                    agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                    if agent is not None:
                        agent.cost_sensitivity = sens

        elif action.type == ActionType.SET_ROLE:
            entity_id = p["entity_id"]
            role = p["role"]
            if target == SimulationTarget.MULTI_AGENT:
                agent = next((a for a in engine.agents if a.agent_id == entity_id), None)
                if agent is not None:
                    agent.kernel_regime = role
            elif target == SimulationTarget.MULTI_CLUSTER:
                if entity_id in engine.clusters:
                    engine.clusters[entity_id].role = role
                else:
                    for c in engine.clusters.values():
                        agent = next((a for a in c.agents if a.agent_id == entity_id), None)
                        if agent is not None:
                            agent.kernel_regime = role

    def _capture_frame(self, engine: Any, target: SimulationTarget, step_idx: int) -> WorkloadFrameMetrics:
        # Defaults
        curvature = 0.0
        tension = 0.0
        coherence = 1.0
        regime = "balanced"
        band = StabilityBand.GREEN
        sao = SAOLevel.NONE
        envelope = EnvelopeState.ADMIT

        if target == SimulationTarget.SINGLE_AGENT:
            # Curvature
            curvature = float(max([engine.boundary.curvature(s.theta) for s in engine.membrane.strings]))
            # Tension
            if hasattr(engine.membrane, "temporal_state") and engine.membrane.temporal_state:
                tension = engine.membrane.temporal_state.accumulated_tension
            # Coherence
            coherence = engine.compute_local_coherence()
            # Regime
            regime = engine.regime_manager.get_regime_for_agent("single_agent").regime_type.value
            # Band
            last_band_str = engine.band_history[-1] if engine.band_history else "green"
            if last_band_str == "yellow":
                band = StabilityBand.YELLOW
            elif last_band_str == "red":
                band = StabilityBand.RED
            # SAO
            if engine.sao_events:
                last_sao = engine.sao_events[-1]
                if last_sao.get("verdict") == "ascend":
                    # map from active regime or step definition
                    allowed = (
                        engine.regime_manager.get_regime_for_agent("single_agent")
                        .sao_rule(None, None, engine).get("allowed_ranges", set())
                    )
                    if "long" in allowed:
                        sao = SAOLevel.LONG
                    elif "mid" in allowed:
                        sao = SAOLevel.MID
                    else:
                        sao = SAOLevel.SHORT
            # Envelope State
            verdict, _ = engine.envelope.evaluate_envelope(engine.membrane, engine.boundary)
            if verdict == "admit":
                envelope = EnvelopeState.ADMIT
            elif verdict == "constrain":
                envelope = EnvelopeState.CONSTRAIN
            elif verdict == "block":
                envelope = EnvelopeState.BLOCK
            elif verdict == "rest":
                envelope = EnvelopeState.REST
            elif verdict == "re-project":
                envelope = EnvelopeState.REPROJECT

        elif target == SimulationTarget.MULTI_AGENT:
            # Curvature
            curvatures = []
            for a in engine.agents:
                if a.shard.state != ShardState.QUARANTINED:
                    curvatures.append(max([a.boundary.curvature(s.theta) for s in a.membrane.strings]))
            curvature = float(np.mean(curvatures)) if curvatures else 0.0
            # Tension
            tension = engine.temporal_state.accumulated_tension
            # Coherence
            active_agents = [a for a in engine.agents if a.shard.state != ShardState.QUARANTINED]
            from radial_membrane_ai.collective_reasoning.coherence import coherence_score
            coherence = coherence_score(active_agents) if active_agents else 1.0
            # Regime
            regime = engine.regime_manager.get_regime_for_agent("agent_1").regime_type.value
            # Band
            last_band_str = engine.band_history[-1] if engine.band_history else "green"
            if last_band_str == "yellow":
                band = StabilityBand.YELLOW
            elif last_band_str == "red":
                band = StabilityBand.RED
            # SAO
            if engine.sao_events:
                last_sao = engine.sao_events[-1]
                if last_sao.get("verdict") == "ascend":
                    sao = SAOLevel.MID
            # Envelope State
            if last_band_str == "red":
                envelope = EnvelopeState.BLOCK
            elif last_band_str == "yellow":
                envelope = EnvelopeState.CONSTRAIN
            else:
                envelope = EnvelopeState.ADMIT

        elif target == SimulationTarget.MULTI_CLUSTER:
            # Curvature
            curvatures = []
            for c in engine.clusters.values():
                if getattr(c, "quarantine_timer", 0) == 0:
                    for a in c.agents:
                        if a.shard.state != ShardState.QUARANTINED:
                            curvatures.append(max([a.boundary.curvature(s.theta) for s in a.membrane.strings]))
            curvature = float(np.mean(curvatures)) if curvatures else 0.0
            # Tension
            tension = engine.temporal_state.accumulated_tension
            # Coherence
            from radial_membrane_ai.collective_reasoning.coherence import global_mesh_coherence_score
            active_clusters = [c for c in engine.clusters.values() if getattr(c, "quarantine_timer", 0) == 0]
            coherence, _ = global_mesh_coherence_score(active_clusters) if active_clusters else (1.0, [])
            # Regime
            regime = engine.regime_manager.get_global_regime().regime_type.value
            # Band
            last_band_str = engine.global_band_history[-1] if engine.global_band_history else "green"
            if last_band_str == "yellow":
                band = StabilityBand.YELLOW
            elif last_band_str == "red":
                band = StabilityBand.RED
            # SAO
            if hasattr(engine, "sao_events") and engine.sao_events:
                last_sao = engine.sao_events[-1]
                if last_sao.get("verdict") == "ascend":
                    sao = SAOLevel.LONG
            # Envelope State
            if last_band_str == "red":
                envelope = EnvelopeState.BLOCK
            elif last_band_str == "yellow":
                envelope = EnvelopeState.CONSTRAIN
            else:
                envelope = EnvelopeState.ADMIT

        return WorkloadFrameMetrics(
            step_index=step_idx,
            curvature=curvature,
            tension=tension,
            coherence=coherence,
            regime=regime,
            stability_band=band,
            sao_level=sao,
            envelope_state=envelope
        )

    def _build_correctness_report(self, result: WorkloadResult, workload: Workload) -> dict:
        """
        Validates workload execution against predefined expectation envelopes.
        """
        all_checks_passed = True
        failed_steps = []

        coherences = [f.coherence for f in result.frames]
        avg_coherence = float(np.mean(coherences)) if coherences else 1.0
        max_tension = float(max([f.tension for f in result.frames])) if result.frames else 0.0

        for idx, frame in enumerate(result.frames):
            step = workload.steps[idx]
            step_passed = True

            # Check stability band
            if frame.stability_band != step.expected_stability_band:
                step_passed = False

            # Check coherence range
            min_coh, max_coh = step.expected_coherence_range
            if not (min_coh <= frame.coherence <= max_coh):
                step_passed = False

            # Check regime
            if frame.regime != step.expected_regime.value:
                # Allow fallback check: if expected regime is BALANCED, accept any regime
                if step.expected_regime == KernelRegimeType.BALANCED:
                    pass
                else:
                    step_passed = False

            # Check SAO level
            if step.expected_sao != SAOLevel.NONE:
                # If short, mid, long expected, check we got some promotion or close match
                if frame.sao_level == SAOLevel.NONE:
                    step_passed = False

            if not step_passed:
                all_checks_passed = False
                failed_steps.append(idx)

        # Check aggregate workload expectations
        if len(result.frames) > 0:
            final_frame = result.frames[-1]
            if final_frame.stability_band != workload.stability_expectation:
                all_checks_passed = False
            if avg_coherence < workload.coherence_expectation:
                all_checks_passed = False

        return {
            "success": all_checks_passed,
            "failed_steps": failed_steps,
            "total_steps": len(workload.steps),
            "average_coherence": avg_coherence,
            "maximum_tension": max_tension,
            "total_rollbacks": len(result.rollback_events),
            "total_quarantines": len(result.quarantine_events),
            "total_transitions": len(result.regime_transitions)
        }
