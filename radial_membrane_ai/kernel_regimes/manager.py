"""
Regime Manager for tracking active regimes and evaluation triggers.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Union
from radial_membrane_ai.kernel_regimes.regime import (
    KernelRegime,
    KernelRegimeType,
    REGIMES,
    DETERMINISTIC_REGIME,
    STOCHASTIC_REGIME,
    HIGH_CURVATURE_REGIME,
    ADVERSARIAL_REGIME,
    MULTI_PHASE_REGIME
)


class RegimeManager:
    """
    Manages operational regimes across agents, clusters, and the global mesh.
    Tracks active regimes, triggers transitions automatically, and enforces rules.
    """

    def __init__(self, default_regime_type: KernelRegimeType = KernelRegimeType.BALANCED) -> None:
        self.default_regime = REGIMES[default_regime_type]
        self.automatic_switching_enabled: bool = False

        # In-memory tracking
        self.agent_regimes: Dict[str, KernelRegime] = {}
        self.cluster_regimes: Dict[str, KernelRegime] = {}
        self.global_regime: KernelRegime = self.default_regime

        # Tracks phase ticks for MULTI_PHASE regime per agent/cluster/global
        # Format: (current_phase, phase_tick_count)
        # current_phase can be "deterministic" or "stochastic"
        self.multiphase_states: Dict[str, Tuple[str, int]] = {}

    def get_regime_for_agent(self, agent_id: str) -> KernelRegime:
        return self.agent_regimes.get(agent_id, self.global_regime)

    def set_regime_for_agent(self, agent_id: str, regime: Union[KernelRegime, KernelRegimeType]) -> None:
        if isinstance(regime, KernelRegimeType):
            regime = REGIMES[regime]
        self.agent_regimes[agent_id] = regime

    def get_regime_for_cluster(self, cluster_id: str) -> KernelRegime:
        return self.cluster_regimes.get(cluster_id, self.global_regime)

    def set_regime_for_cluster(self, cluster_id: str, regime: Union[KernelRegime, KernelRegimeType]) -> None:
        if isinstance(regime, KernelRegimeType):
            regime = REGIMES[regime]
        self.cluster_regimes[cluster_id] = regime

    def get_global_regime(self) -> KernelRegime:
        return self.global_regime

    def set_global_regime(self, regime: Union[KernelRegime, KernelRegimeType]) -> None:
        if isinstance(regime, KernelRegimeType):
            regime = REGIMES[regime]
        self.global_regime = regime

    def get_active_multiphase_phase(self, entity: Any) -> str:
        """
        Gets the active phase ("deterministic" or "stochastic") for an entity.
        Defaults to "deterministic".
        """
        key = self._get_entity_key(entity)
        state = self.multiphase_states.get(key)
        if state is None:
            return "deterministic"
        return state[0]

    def _get_entity_key(self, entity: Any) -> str:
        if entity is None:
            return "global"
        if hasattr(entity, "agent_id"):
            return f"agent_{entity.agent_id}"
        if hasattr(entity, "cluster_id"):
            return f"cluster_{entity.cluster_id}"
        return "global"

    def tick_multiphase_state(self, entity: Any, t_bar: float) -> None:
        """
        Ticks the multiphase tick count for an entity.
        Alternates deterministic for 10 ticks, then stochastic for 10 ticks.
        Immediate transition if t_bar > 0.8.
        """
        key = self._get_entity_key(entity)
        current_phase, ticks = self.multiphase_states.get(key, ("deterministic", 0))

        # Check transition trigger: temporal tension T_bar > 0.8
        if t_bar > 0.8:
            next_phase = "stochastic" if current_phase == "deterministic" else "deterministic"
            self.multiphase_states[key] = (next_phase, 1)
            return

        ticks += 1
        if ticks >= 10:
            next_phase = "stochastic" if current_phase == "deterministic" else "deterministic"
            self.multiphase_states[key] = (next_phase, 1)
        else:
            self.multiphase_states[key] = (current_phase, ticks)

    def evaluate_switching_triggers(
        self,
        entity: Any,
        t_bar: float,
        coherence: float,
        stability_band: str,
        curvature: float = 0.0,
        cluster_tension: float = 0.0,
        cost_band: int = 0,
        envelope_conflict: bool = False
    ) -> KernelRegime:
        """
        Evaluates the switching triggers for a given agent, cluster or global mesh,
        updates the active regime, and returns the updated KernelRegime.
        """
        current_regime = self._get_active_regime_for_entity(entity)
        if not self.automatic_switching_enabled:
            return current_regime

        # 1. Check Return to Deterministic Trigger
        # If T_bar < 0.5 AND coherence >= 0.7 AND stability band is green AND no envelope conflicts
        if (t_bar < 0.5 and coherence >= 0.7 and stability_band == "green" and not envelope_conflict):
            target_regime = DETERMINISTIC_REGIME

        # 2. Check Switch to MULTI_PHASE
        # If T_bar > 0.8 AND coherence >= 0.6 OR stability oscillates (handled via inputs or active oscillation flag)
        elif (t_bar > 0.8 and coherence >= 0.6):
            target_regime = MULTI_PHASE_REGIME

        # 3. Check Switch to HIGH_CURVATURE
        elif (t_bar > 0.8 or curvature > 0.7):
            target_regime = HIGH_CURVATURE_REGIME

        # 4. Check Switch to ADVERSARIAL
        elif (coherence < 0.4 or stability_band in ("yellow", "red") or envelope_conflict):
            target_regime = ADVERSARIAL_REGIME

        # 5. Check Switch to STOCHASTIC
        elif (coherence < 0.5 or cost_band == 1 or cluster_tension > 0.7):
            target_regime = STOCHASTIC_REGIME

        else:
            # Fall back to current or default
            target_regime = current_regime

        # Save and return the decided regime
        self._set_active_regime_for_entity(entity, target_regime)

        # If switching to MULTI_PHASE and state not initialized, initialize it
        if target_regime.regime_type == KernelRegimeType.MULTI_PHASE:
            key = self._get_entity_key(entity)
            if key not in self.multiphase_states:
                self.multiphase_states[key] = ("deterministic", 0)

        return target_regime

    def _get_active_regime_for_entity(self, entity: Any) -> KernelRegime:
        if entity is None:
            return self.global_regime
        if isinstance(entity, str):
            if entity.startswith("cluster_"):
                return self.get_regime_for_cluster(entity)
            else:
                return self.get_regime_for_agent(entity)
        if hasattr(entity, "agent_id"):
            return self.get_regime_for_agent(entity.agent_id)
        if hasattr(entity, "cluster_id"):
            return self.get_regime_for_cluster(entity.cluster_id)
        return self.global_regime

    def _set_active_regime_for_entity(self, entity: Any, regime: KernelRegime) -> None:
        if entity is None:
            self.set_global_regime(regime)
        elif isinstance(entity, str):
            if entity.startswith("cluster_"):
                self.set_regime_for_cluster(entity, regime)
            else:
                self.set_regime_for_agent(entity, regime)
        elif hasattr(entity, "agent_id"):
            self.set_regime_for_agent(entity.agent_id, regime)
        elif hasattr(entity, "cluster_id"):
            self.set_regime_for_cluster(entity.cluster_id, regime)
        else:
            self.set_global_regime(regime)
