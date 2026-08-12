# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Facet-State Governance module.

This module implements the facet vectors, tension-state finite automaton,
and typed state-aware V-channel routing rules as described in Feeney (2026).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class TensionState(Enum):
    """
    Enum representing different tension levels/states of a behavioral facet.
    """
    RELAXED = auto()
    ADMISSIBLE = auto()
    STRETCHED = auto()
    CRITICAL = auto()
    BOUNDARY_COLLAPSE = auto()
    IDLE = auto()
    RESONANT = auto()
    ACTIVE = auto()
    SATURATED = auto()
    INHIBITED = auto()


@dataclass
class FacetVector:
    """
    Facet-State Vector S_i(t) representing the physical and governance state
    of a specific behavioral dimension (facet) at index i.

    S_i(t) = (f_i, t_i(t), a_i(t), c_i(t), p_i(t), pi_i(t))
    where:
    - facet_id: Unique identifier/name of the facet.
    - state: Current tension state.
    - activation: Current activation level of the facet.
    - capacity: Current admissible capacity at this facet's angle.
    - residual: Current residual deformation magnitude.
    - policy_priority: Relative policy priority weight.
    - coherence_contribution: Contribution of this facet to overall coherence.
    - persistence_trace: Trace for historical persistence tracking.
    - capacity_pressure: Pressure index for capacity boundary.
    - typed_v_channel: Channel type associated with this facet.
    """
    facet_id: str
    state: TensionState
    activation: float
    capacity: float
    residual: float
    policy_priority: float
    coherence_contribution: float = 0.0
    persistence_trace: float = 0.0
    capacity_pressure: float = 0.0
    typed_v_channel: str = "Type-W"


class TensionAutomaton:
    """
    Finite State Automaton governing transitions between tension-states
    based on local closure ratio i(t, theta) and residual deformation.
    """

    def __init__(self, critical_threshold: float = 0.9, collapse_threshold: float = 1.2) -> None:
        """
        Initializes the TensionAutomaton.

        Args:
            critical_threshold: Ratio above which state transitions towards CRITICAL.
            collapse_threshold: Ratio above which BOUNDARY_COLLAPSE is triggered.
        """
        self.critical_threshold = critical_threshold
        self.collapse_threshold = collapse_threshold

    def transition(self, current_state: TensionState, closure_ratio: float, residual: float) -> TensionState:
        """
        Evaluates the next TensionState based on current state, closure ratio, and residual deformation.

        Transition Rules:
        - If closure_ratio > collapse_threshold or residual > 0.5: BOUNDARY_COLLAPSE
        - If closure_ratio > critical_threshold and not collapse: CRITICAL
        - If 0.7 < closure_ratio <= critical_threshold: STRETCHED
        - If 0.2 < closure_ratio <= 0.7: ADMISSIBLE
        - If closure_ratio <= 0.2: RELAXED
        """
        if closure_ratio >= self.collapse_threshold or residual >= 0.5:
            return TensionState.BOUNDARY_COLLAPSE
        elif closure_ratio >= self.critical_threshold:
            return TensionState.CRITICAL
        elif closure_ratio >= 0.7:
            return TensionState.STRETCHED
        elif closure_ratio >= 0.2:
            return TensionState.ADMISSIBLE
        else:
            return TensionState.RELAXED


def route_signal(source: FacetVector, target: FacetVector, base_weight: float) -> float:
    """
    Computes a state-aware routed signal weight between source and target facets.

    Rules:
    - If either source or target is in BOUNDARY_COLLAPSE, route signal is fully suppressed (0.0).
    - If target is in CRITICAL, incoming signals are scaled down heavily to prevent further overload (0.2).
    - If target is in STRETCHED, signal weight is scaled by 0.6.
    - If source is in STRETCHED or CRITICAL and target is in ADMISSIBLE, boost routing (1.2).
    - Otherwise, returns base_weight.
    """
    if source.state == TensionState.BOUNDARY_COLLAPSE or target.state == TensionState.BOUNDARY_COLLAPSE:
        return 0.0

    if target.state == TensionState.CRITICAL:
        return base_weight * 0.2

    if target.state == TensionState.STRETCHED:
        return base_weight * 0.6

    if (source.state in (TensionState.STRETCHED, TensionState.CRITICAL)) and target.state == TensionState.ADMISSIBLE:
        return base_weight * 1.2

    return base_weight
