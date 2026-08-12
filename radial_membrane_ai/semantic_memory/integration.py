# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Integration helper functions binding Policy-Bound Semantic Memory to simulation engines.
"""

from __future__ import annotations
from typing import Any, Optional

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.semantic_memory.core import (
    AgentSemanticMemory,
    MeshSemanticMemory,
    promote_memory
)
from radial_membrane_ai.semantic_memory.policy import AdmissibilityContext


def bind_to_agent(agent: UFOAgent) -> None:
    """
    Binds a brand new AgentSemanticMemory layer to a UFOAgent.
    """
    agent.semantic_memory = AgentSemanticMemory(agent_id=agent.agent_id)  # type: ignore


def bind_to_mesh(engine: Any) -> None:
    """
    Binds a brand new MeshSemanticMemory layer to the multi-agent or single-agent simulation engine.
    """
    engine.mesh_memory = MeshSemanticMemory()


def on_tick_start(engine: Any) -> None:
    """
    Pre-tick hook called at the start of a simulation tick.
    Decays curvature and tension in local and global stores to simulate dynamic temporal decay of memory stress.
    """
    # 1. Decay global shared memory state
    if hasattr(engine, "mesh_memory") and engine.mesh_memory is not None:
        engine.mesh_memory.curvature_state.decay(rate=0.05)

    # 2. Decay per-agent memory state
    agents = getattr(engine, "agents", [])
    # Also support single-agent engine where there is a single membrane/agent
    if hasattr(engine, "membrane") and hasattr(engine, "semantic_memory"):
        if engine.semantic_memory is not None:
            engine.semantic_memory.curvature_state.decay(rate=0.05)

    for agent in agents:
        if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
            agent.semantic_memory.curvature_state.decay(rate=0.05)


def on_tick_end(engine: Any) -> None:
    """
    Post-tick hook called at the end of a simulation tick.
    Updates memory state, logs events, and incorporates semantic memory tension/curvature back into engine metrics.
    """
    # Single-agent engine integration:
    # If the single agent's memory has high tension, it increases Lyapunov energy or dampens coherence.
    if hasattr(engine, "membrane") and hasattr(engine, "semantic_memory"):
        mem_store = engine.semantic_memory
        if mem_store is not None:
            # We can log a summary trace or impact single-agent stability
            # For example, append high memory tension as an extra overhead / stability factor
            tension_factor = mem_store.curvature_state.tension
            if tension_factor > 1.0 and hasattr(engine, "interventions"):
                engine.interventions.append(
                    f"Memory Tension Escalation: {tension_factor:.2f}. "
                    "Tightening security scopes and auditing local keys."
                )

    # Multi-agent engine integration:
    if hasattr(engine, "mesh_memory") and engine.mesh_memory is not None:
        mesh_mem = engine.mesh_memory
        # If global shared memory tension exceeds threshold, log mesh intervention
        tot_tension = mesh_mem.curvature_state.tension
        if tot_tension > 1.5 and hasattr(engine, "interventions"):
            engine.interventions.append(
                f"Global Shared Memory Tension Warning: {tot_tension:.2f}. "
                "Initiating mesh-wide attestation and key scrubbing."
            )


def on_sao_promotion(agent_id: str, key: str, engine: Any) -> bool:
    """
    SAO Promotion hook promoting a record from an agent's local memory store to the shared global store.
    """
    if not hasattr(engine, "mesh_memory") or engine.mesh_memory is None:
        return False

    # Find the target agent
    target_agent: Optional[UFOAgent] = None
    agents = getattr(engine, "agents", [])
    for a in agents:
        if a.agent_id == agent_id:
            target_agent = a
            break

    if target_agent is None or not hasattr(target_agent, "semantic_memory"):
        return False

    agent_mem = target_agent.semantic_memory
    if agent_mem is None:
        return False

    # Create admissibility context from the engine's current state/coherence
    coherence = 0.5
    quarantined = False
    cost_factor = 1.0
    stability_energy = 0.0

    # Extract metrics if available on engine
    if hasattr(engine, "c_mesh_history") and engine.c_mesh_history:
        coherence = engine.c_mesh_history[-1]
    elif hasattr(engine, "coherence_history") and engine.coherence_history:
        coherence = engine.coherence_history[-1]

    if hasattr(target_agent, "shard"):
        quarantined = (target_agent.shard.state.name == "QUARANTINED")
        cost_factor = target_agent.shard.cost_factor

    if hasattr(engine, "v_history") and engine.v_history:
        stability_energy = engine.v_history[-1]

    context = AdmissibilityContext(
        coherence=coherence,
        quarantined=quarantined,
        cost_factor=cost_factor,
        stability_energy=stability_energy
    )

    ledger = getattr(engine, "ledger", None)
    # Support mesh governance ledger on MultiAgentEngine
    if ledger is None and hasattr(engine, "mesh_governance"):
        ledger = getattr(engine.mesh_governance, "ledger", None)

    res = promote_memory(
        mesh_memory=engine.mesh_memory,
        agent_memory=agent_mem,
        key=key,
        context=context,
        global_ledger=ledger
    )

    return res.success
