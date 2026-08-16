# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Integration tests for the Legacy Hardware Invariant Handshake Adapter.
"""

import pytest
import numpy as np
import argparse

from radial_membrane_ai.admissibility import AdmissibilityGate
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine
from radial_membrane_ai.ufo_engine.multi_agent import MultiAgentEngine
from radial_membrane_ai.ufo_engine.multi_cluster import MultiClusterEngine
from radial_membrane_ai.workloads.engine import WorkloadEngine
from ufo_cli import run_simulation


def test_admissibility_gate_handshake() -> None:
    gate = AdmissibilityGate(tol=0.2, legacy_mode="strict")
    activations = [0.2] * 12
    status = gate.check_legacy_handshake(
        string_activations=activations,
        compute_cost=0.3,
        lyapunov_energy=0.8,
        v_channel_pressure=1.0,
        conversion_cost=0.01,
    )
    assert status.handshakeAllowed is True


def test_single_agent_engine_legacy_handshake() -> None:
    engine = SingleAgentEngine(
        seed=0,
        enable_legacy_handshake=True,
        legacy_handshake_mode="strict",
    )
    band = engine.tick(task_value=0.8, excitation=np.ones(12) * 0.1)
    assert band in ["green", "yellow", "red", "quarantined"]


def test_multi_agent_engine_legacy_handshake() -> None:
    engine = MultiAgentEngine(
        n_agents=2,
        seed=0,
        enable_legacy_handshake=True,
        legacy_handshake_mode="strict",
    )
    res = engine.tick(task_value=0.8, excitation=np.ones(12) * 0.1)
    assert res in ["green", "yellow", "red", "quarantined"]


def test_multi_cluster_engine_legacy_handshake() -> None:
    engine = MultiClusterEngine(
        seed=0,
        enable_legacy_handshake=True,
        legacy_handshake_mode="strict",
    )
    # Basic tick verify
    assert engine.enable_legacy_handshake is True


def test_workload_engine_legacy_handshake() -> None:
    wl_engine = WorkloadEngine(
        enable_legacy_handshake=True,
        legacy_handshake_mode="strict",
    )
    assert wl_engine.enable_legacy_handshake is True
    assert wl_engine.single_agent_engine.enable_legacy_handshake is True


def test_cli_legacy_handshake_execution() -> None:
    args = argparse.Namespace(
        single=True,
        multi=False,
        mode="single",
        steps=2,
        excitation=0.1,
        task_value=0.8,
        seed=0,
        legacy_handshake=True,
        legacy_mode="strict",
        visualize=False,
        quiet=True,
        verbose=False,
        demo=False,
        demo_governance=False,
    )
    run_simulation(args)
