"""
Workload representation and core classes for Governed Stress-Testing Workload Family Design in U.F.O.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType
from radial_membrane_ai.collective_reasoning.policy_envelope import PolicyEnvelope

# Type aliases as specified
AgentID = str
ClusterID = str


class ActionType(Enum):
    SET_ACTIVATION = "SET_ACTIVATION"
    WRITE_MEMORY = "WRITE_MEMORY"
    TRIGGER_VIOLATION = "TRIGGER_VIOLATION"
    CHANGE_REGIME = "CHANGE_REGIME"
    SET_ENVELOPE = "SET_ENVELOPE"
    SET_TENSION = "SET_TENSION"
    SET_CURVATURE = "SET_CURVATURE"
    SET_COST_PROFILE = "SET_COST_PROFILE"
    SET_ROLE = "SET_ROLE"
    CUSTOM = "CUSTOM"


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]

    @classmethod
    def set_activation(cls, agent_id: str, value: float | list[float]) -> Action:
        return cls(ActionType.SET_ACTIVATION, {"agent_id": agent_id, "value": value})

    @classmethod
    def write_memory(cls, scope: str, key: str, value: Any, tags: list[str] | None = None) -> Action:
        return cls(ActionType.WRITE_MEMORY, {"scope": scope, "key": key, "value": value, "tags": tags or []})

    @classmethod
    def trigger_violation(cls, entity_id: str, violation_type: str) -> Action:
        return cls(ActionType.TRIGGER_VIOLATION, {"entity_id": entity_id, "violation_type": violation_type})

    @classmethod
    def change_regime(cls, entity_id: str, regime: str) -> Action:
        return cls(ActionType.CHANGE_REGIME, {"entity_id": entity_id, "regime": regime})

    @classmethod
    def set_envelope(cls, entity_id: str, min_trust: float, stability_required_band: str) -> Action:
        return cls(ActionType.SET_ENVELOPE, {
            "entity_id": entity_id,
            "min_trust": min_trust,
            "stability_required_band": stability_required_band
        })

    @classmethod
    def set_tension(cls, entity_id: str, tension: float) -> Action:
        return cls(ActionType.SET_TENSION, {"entity_id": entity_id, "tension": tension})

    @classmethod
    def set_curvature(cls, entity_id: str, curvature: float) -> Action:
        return cls(ActionType.SET_CURVATURE, {"entity_id": entity_id, "curvature": curvature})

    @classmethod
    def set_cost_profile(cls, entity_id: str, cost_sensitivity: float) -> Action:
        return cls(ActionType.SET_COST_PROFILE, {"entity_id": entity_id, "cost_sensitivity": cost_sensitivity})

    @classmethod
    def set_role(cls, entity_id: str, role: str) -> Action:
        return cls(ActionType.SET_ROLE, {"entity_id": entity_id, "role": role})

    @classmethod
    def custom_action(cls, name: str, payload: Dict[str, Any]) -> Action:
        return cls(ActionType.CUSTOM, {"name": name, "payload": payload})


class StabilityBand(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class SAOLevel(Enum):
    NONE = "none"
    SHORT = "short"
    MID = "mid"
    LONG = "long"


class EnvelopeState(Enum):
    ADMIT = "admit"
    CONSTRAIN = "constrain"
    BLOCK = "block"
    REST = "rest"
    REPROJECT = "re-project"


class SimulationTarget(Enum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    MULTI_CLUSTER = "multi_cluster"


@dataclass
class WorkloadStep:
    agent_actions: Dict[AgentID, Action] = field(default_factory=dict)
    cluster_actions: Dict[ClusterID, Action] = field(default_factory=dict)
    global_actions: List[Action] = field(default_factory=list)
    expected_regime: KernelRegimeType = KernelRegimeType.BALANCED
    expected_sao: SAOLevel = SAOLevel.NONE
    expected_stability_band: StabilityBand = StabilityBand.GREEN
    expected_coherence_range: Tuple[float, float] = (0.0, 1.0)
    expected_envelope_state: EnvelopeState = EnvelopeState.ADMIT


@dataclass
class Workload:
    name: str
    description: str
    target: SimulationTarget
    regime_expectation: KernelRegimeType
    stability_expectation: StabilityBand
    coherence_expectation: float
    envelope_expectation: PolicyEnvelope
    steps: List[WorkloadStep]


# =============================================================================
# Pre-Packaged Workload Family Builders
# =============================================================================

def create_cooperative_workload() -> Workload:
    """
    1. Cooperative Workloads
    Agents collaborate toward a shared goal (multi-agent planning, distributed summarization,
    shared semantic memory construction).
    Expects high coherence, deterministic regime dominance, short-range SAO, stable temporal tension.
    """
    steps = [
        WorkloadStep(
            agent_actions={
                "agent_1": Action.change_regime("agent_1", "DETERMINISTIC"),
                "agent_2": Action.change_regime("agent_2", "DETERMINISTIC"),
                "agent_3": Action.change_regime("agent_3", "DETERMINISTIC"),
                "agent_1_act": Action.set_activation("agent_1", 0.5),
                "agent_2_act": Action.set_activation("agent_2", 0.5),
            },
            global_actions=[
                Action.write_memory("global", "shared_plan", "cooperative_consensus", ["planning", "consensus"])
            ],
            expected_regime=KernelRegimeType.DETERMINISTIC,
            expected_sao=SAOLevel.SHORT,
            expected_stability_band=StabilityBand.GREEN,
            expected_coherence_range=(0.7, 1.0),
            expected_envelope_state=EnvelopeState.ADMIT
        ),
        WorkloadStep(
            agent_actions={
                "agent_1_act": Action.set_activation("agent_1", 0.6),
                "agent_2_act": Action.set_activation("agent_2", 0.6),
            },
            global_actions=[
                Action.write_memory("global", "shared_summary", "cooperative_summary", ["summary"])
            ],
            expected_regime=KernelRegimeType.DETERMINISTIC,
            expected_sao=SAOLevel.SHORT,
            expected_stability_band=StabilityBand.GREEN,
            expected_coherence_range=(0.7, 1.0),
            expected_envelope_state=EnvelopeState.ADMIT
        )
    ]
    return Workload(
        name="Cooperative Planning Workload",
        description=(
            "Agents cooperate on shared planning tasks with deterministic regime dominance and high coherence."
        ),
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.DETERMINISTIC,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.8,
        envelope_expectation=PolicyEnvelope(min_trust=0.5, stability_required_band="yellow"),
        steps=steps
    )


def create_adversarial_workload() -> Workload:
    """
    2. Adversarial Workloads
    Agents intentionally push tension, cost, or instability (conflicting memory writes,
    contradictory reasoning paths, adversarial perturbation).
    Expects adversarial regime activation, cluster-level SAO only, rollback events, stability band sensitivity.
    """
    steps = [
        WorkloadStep(
            agent_actions={
                "agent_1": Action.change_regime("agent_1", "ADVERSARIAL"),
                "agent_2": Action.change_regime("agent_2", "ADVERSARIAL"),
                "agent_1_conflict": Action.write_memory("agent_1", "conflict_key", "value_A", ["malicious"]),
                "agent_2_conflict": Action.write_memory("agent_2", "conflict_key", "value_B", ["malicious"]),
                "agent_1_act": Action.set_activation("agent_1", [1.0] * 12),
                "agent_2_act": Action.set_activation("agent_2", [1.0] * 12),
            },
            expected_regime=KernelRegimeType.ADVERSARIAL,
            expected_sao=SAOLevel.MID,  # mid-range is cluster level
            expected_stability_band=StabilityBand.RED,
            expected_coherence_range=(0.0, 0.5),
            expected_envelope_state=EnvelopeState.BLOCK
        ),
        WorkloadStep(
            agent_actions={
                "agent_1_act": Action.set_activation("agent_1", [1.1] * 12),
                "agent_2_act": Action.set_activation("agent_2", [1.1] * 12),
                "agent_1_viol": Action.trigger_violation("agent_1", "malicious_overwrite")
            },
            expected_regime=KernelRegimeType.ADVERSARIAL,
            expected_sao=SAOLevel.NONE,
            expected_stability_band=StabilityBand.RED,
            expected_coherence_range=(0.0, 0.4),
            expected_envelope_state=EnvelopeState.BLOCK
        )
    ]
    return Workload(
        name="Adversarial Stress Workload",
        description="Agents execute adversarial writes and perturbations triggering red stability band and rollbacks.",
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.ADVERSARIAL,
        stability_expectation=StabilityBand.RED,
        coherence_expectation=0.2,
        envelope_expectation=PolicyEnvelope(min_trust=0.9, stability_required_band="green"),
        steps=steps
    )


def create_high_curvature_workload() -> Workload:
    """
    3. High-Curvature Workloads
    Tasks that force membrane curvature to spike (deep multi-step reasoning, rapid context switching).
    Expects high-curvature regime activation, long-range SAO, amplified curvature, capacity shrinkage,
    tension acceleration.
    """
    steps = [
        WorkloadStep(
            agent_actions={
                "single_agent": Action.change_regime("single_agent", "HIGH_CURVATURE"),
                "act_spike": Action.set_activation("single_agent", [0.9] * 12),
                "curv_spike": Action.set_curvature("single_agent", 5.0),
                "tens_spike": Action.set_tension("single_agent", 4.0)
            },
            expected_regime=KernelRegimeType.HIGH_CURVATURE,
            expected_sao=SAOLevel.LONG,
            expected_stability_band=StabilityBand.YELLOW,
            expected_coherence_range=(0.3, 0.7),
            expected_envelope_state=EnvelopeState.CONSTRAIN
        ),
        WorkloadStep(
            agent_actions={
                "act_spike_2": Action.set_activation("single_agent", [0.95] * 12),
                "curv_spike_2": Action.set_curvature("single_agent", 8.0)
            },
            expected_regime=KernelRegimeType.HIGH_CURVATURE,
            expected_sao=SAOLevel.LONG,
            expected_stability_band=StabilityBand.YELLOW,
            expected_coherence_range=(0.2, 0.6),
            expected_envelope_state=EnvelopeState.CONSTRAIN
        )
    ]
    return Workload(
        name="High-Curvature Planning Workload",
        description="Exercises rapid context switching to spike membrane curvature and test capacity shrinkage.",
        target=SimulationTarget.SINGLE_AGENT,
        regime_expectation=KernelRegimeType.HIGH_CURVATURE,
        stability_expectation=StabilityBand.YELLOW,
        coherence_expectation=0.5,
        envelope_expectation=PolicyEnvelope(min_trust=0.4, stability_required_band="yellow"),
        steps=steps
    )


def create_policy_tension_workload() -> Workload:
    """
    4. Policy-Tension Workloads
    Tasks that intentionally push against policy envelopes (forbidden memory fields, envelope violations,
    unauthorized actions).
    Expects envelope violation detection, rollback + quarantine, global governance activation.
    """
    steps = [
        WorkloadStep(
            agent_actions={
                "agent_1_viol": Action.write_memory("agent_1", "forbidden_field_xyz", "classified_data", ["forbidden"]),
                "agent_2_viol": Action.set_envelope("agent_2", min_trust=0.99, stability_required_band="green")
            },
            global_actions=[
                Action.trigger_violation("global", "unauthorized_access")
            ],
            expected_regime=KernelRegimeType.BALANCED,
            expected_sao=SAOLevel.NONE,
            expected_stability_band=StabilityBand.RED,
            expected_coherence_range=(0.0, 0.6),
            expected_envelope_state=EnvelopeState.BLOCK
        ),
        WorkloadStep(
            agent_actions={
                "agent_1_act": Action.set_activation("agent_1", [1.0] * 12),
                "agent_2_act": Action.set_activation("agent_2", [1.0] * 12),
            },
            expected_regime=KernelRegimeType.BALANCED,
            expected_sao=SAOLevel.NONE,
            expected_stability_band=StabilityBand.RED,
            expected_coherence_range=(0.0, 0.5),
            expected_envelope_state=EnvelopeState.BLOCK
        )
    ]
    return Workload(
        name="Policy-Tension Envelope Workload",
        description=(
            "Fires unauthorized operations and violates envelope thresholds to verify rollbacks and quarantines."
        ),
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.BALANCED,
        stability_expectation=StabilityBand.RED,
        coherence_expectation=0.4,
        envelope_expectation=PolicyEnvelope(min_trust=0.8, stability_required_band="green"),
        steps=steps
    )


def create_asymmetric_workload() -> Workload:
    """
    5. Asymmetric Multi-Agent Workloads
    Agents with different roles, costs, or tensions interact (planner + critic + explorer).
    Expects regime divergence, cross-agent admissibility, cluster coherence scoring, mixed SAO gating.
    """
    steps = [
        WorkloadStep(
            agent_actions={
                "agent_1": Action.set_role("agent_1", "planner"),
                "agent_2": Action.set_role("agent_2", "critic"),
                "agent_3": Action.set_role("agent_3", "explorer"),
                "agent_1_cost": Action.set_cost_profile("agent_1", 2.5),
                "agent_2_cost": Action.set_cost_profile("agent_2", 0.5),
                "agent_1_reg": Action.change_regime("agent_1", "DETERMINISTIC"),
                "agent_2_reg": Action.change_regime("agent_2", "STOCHASTIC"),
                "agent_3_reg": Action.change_regime("agent_3", "HIGH_CURVATURE"),
            },
            expected_regime=KernelRegimeType.BALANCED,
            expected_sao=SAOLevel.MID,
            expected_stability_band=StabilityBand.GREEN,
            expected_coherence_range=(0.5, 0.9),
            expected_envelope_state=EnvelopeState.ADMIT
        ),
        WorkloadStep(
            agent_actions={
                "agent_1_act": Action.set_activation("agent_1", 0.7),
                "agent_2_act": Action.set_activation("agent_2", 0.3),
                "agent_3_act": Action.set_activation("agent_3", 0.9)
            },
            expected_regime=KernelRegimeType.BALANCED,
            expected_sao=SAOLevel.MID,
            expected_stability_band=StabilityBand.GREEN,
            expected_coherence_range=(0.5, 0.9),
            expected_envelope_state=EnvelopeState.ADMIT
        )
    ]
    return Workload(
        name="Role Asymmetry Workload",
        description="Orchestrates roles planner/critic/explorer with varied costs and regime divergence.",
        target=SimulationTarget.MULTI_AGENT,
        regime_expectation=KernelRegimeType.BALANCED,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.7,
        envelope_expectation=PolicyEnvelope(min_trust=0.5, stability_required_band="yellow"),
        steps=steps
    )


def create_global_mesh_workload() -> Workload:
    """
    6. Global Mesh Workloads
    Full multi-cluster, multi-regime stress tests (federated memory, global SAO, cross-cluster reasoning).
    Expects global envelope enforcement, global rollback, global coherence scoring, long-range temporal governance.
    """
    steps = [
        WorkloadStep(
            cluster_actions={
                "cluster_1": Action.change_regime("cluster_1", "DETERMINISTIC"),
                "cluster_2": Action.change_regime("cluster_2", "HIGH_CURVATURE")
            },
            global_actions=[
                Action.write_memory("global", "federated_fact", "global_truth_value", ["global", "federated"]),
                Action.change_regime("global", "MULTI_PHASE")
            ],
            expected_regime=KernelRegimeType.MULTI_PHASE,
            expected_sao=SAOLevel.LONG,
            expected_stability_band=StabilityBand.GREEN,
            expected_coherence_range=(0.6, 1.0),
            expected_envelope_state=EnvelopeState.ADMIT
        ),
        WorkloadStep(
            cluster_actions={
                "cluster_1": Action.change_regime("cluster_1", "HIGH_CURVATURE"),
                "cluster_1_viol": Action.trigger_violation("cluster_1", "high_tension_leak")
            },
            expected_regime=KernelRegimeType.MULTI_PHASE,
            expected_sao=SAOLevel.LONG,
            expected_stability_band=StabilityBand.YELLOW,
            expected_coherence_range=(0.4, 0.8),
            expected_envelope_state=EnvelopeState.CONSTRAIN
        )
    ]
    return Workload(
        name="Distributed Global Mesh Workload",
        description=(
            "Stresses multi-cluster alignment, federated memory, and global SAO promotion under policy envelopes."
        ),
        target=SimulationTarget.MULTI_CLUSTER,
        regime_expectation=KernelRegimeType.MULTI_PHASE,
        stability_expectation=StabilityBand.GREEN,
        coherence_expectation=0.75,
        envelope_expectation=PolicyEnvelope(min_trust=0.4, stability_required_band="yellow"),
        steps=steps
    )
