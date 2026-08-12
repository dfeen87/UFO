# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Tests for the dynamic terminal execution block of radial_membrane_ai/simulation.py.
"""

from __future__ import annotations
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock

from radial_membrane_ai.exceptions import GovernanceError
from radial_membrane_ai.simulation import run_governed_simulation, print_banner


def test_print_banner() -> None:
    # Verify that printing a banner executes without failure
    with patch("builtins.print") as mock_print:
        print_banner("Test Banner Text")
        mock_print.assert_called()


def test_run_governed_simulation_nominal() -> None:
    # Test nominal execution of the simulation run wrapper
    # We patch time.sleep to run instantly and input to prevent hanging.
    with patch("time.sleep"), \
            patch("builtins.input", return_value=""), \
            patch("builtins.print") as mock_print:
        run_governed_simulation()

        # Verify deterministic initialization, cinematic banners, ticks, and completion printed
        printed_texts = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("[Deterministic Seed Initialized: 0]" in text for text in printed_texts)
        assert any("🛸 UFO SIMULATION MODULE EXECUTING 🛸" in text for text in printed_texts)
        assert any("🛸 MODULE EXECUTION COMPLETE — RETURNING TO SHELL 🛸" in text for text in printed_texts)


def test_run_governed_simulation_red_band() -> None:
    # Test that the RED stability band branch is taken when energy is high
    sim_instance = MagicMock()
    # Mock governor energy_history to return > 2.0
    sim_instance.governor.energy_history = [3.0]
    sim_instance.mesh_coherence_history = [0.8]
    mock_string = MagicMock()
    mock_string.theta = 0.0
    sim_instance.membrane.strings = [mock_string]
    sim_instance.boundary.curvature.return_value = 1.5
    # Configure the temporal_state mock to return a float for accumulated_tension
    sim_instance.membrane.temporal_state.accumulated_tension = 0.5

    with patch("time.sleep"), \
            patch("builtins.input", return_value=""), \
            patch("radial_membrane_ai.simulation.RainbowSimulation", return_value=sim_instance), \
            patch("builtins.print") as mock_print:
        run_governed_simulation()

        printed_texts = [call[0][0] for call in mock_print.call_args_list if call[0]]
        # Check that RED stability band was printed
        assert any("RED" in text for text in printed_texts)


def test_run_governed_simulation_halt() -> None:
    # Test execution when a GovernanceError is raised
    # We mock run_step on RainbowSimulation to raise GovernanceError
    with patch("time.sleep"), \
            patch("builtins.input", return_value=""), \
            patch("radial_membrane_ai.simulation.RainbowSimulation") as mock_sim_class, \
            patch("builtins.print") as mock_print:

        # Setup mock instance of RainbowSimulation to raise GovernanceError during run_step
        mock_sim_inst = MagicMock()
        mock_sim_inst.run_step.side_effect = GovernanceError("Simulated governance breach of temporal limits.")
        mock_sim_class.return_value = mock_sim_inst

        run_governed_simulation()

        printed_texts = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("🚨 GOVERNED HALT ENFORCED 🚨" in text for text in printed_texts)
        assert any("Violation Details:" in text for text in printed_texts)
        assert any("Simulated governance breach of temporal limits." in text for text in printed_texts)
        mock_sim_inst.render_summary.assert_called()


def test_simulation_main_direct() -> None:
    # Read simulation.py and execute it with __name__ set to "__main__"
    # to get 100% test coverage including the __main__ block
    with open("radial_membrane_ai/simulation.py") as f:
        code = f.read()

    with patch("time.sleep"), \
            patch("builtins.input", return_value=""), \
            patch("builtins.print"):
        # Create a dictionary for execution globals
        globals_dict = {"__name__": "__main__"}
        # Execute the module code
        exec(code, globals_dict)


def test_main_subprocess() -> None:
    # Run python -m radial_membrane_ai.simulation as a subprocess to hit the __main__ block
    # We pass an empty newline to standard input to simulate pressing Enter.
    # We pass UFO_TEST_RUN=1 to run a single step quickly.
    env = dict(os.environ)
    env["UFO_TEST_RUN"] = "1"

    p = subprocess.Popen(
        [sys.executable, "-m", "radial_membrane_ai.simulation"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )
    stdout, stderr = p.communicate(input="\n", timeout=15)
    assert p.returncode == 0
    assert "🛸 UFO SIMULATION MODULE EXECUTING 🛸" in stdout
    assert "🛸 MODULE EXECUTION COMPLETE — RETURNING TO SHELL 🛸" in stdout
