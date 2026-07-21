"""
Tests for the dynamic terminal execution CLI of U.F.O.
"""

from __future__ import annotations
import os
import argparse
from unittest.mock import patch

import ufo_cli


def test_append_to_halt_log() -> None:
    # Test logging of halt behavior
    if os.path.exists("logs/governance_halts.log"):
        os.remove("logs/governance_halts.log")
    ufo_cli.append_to_halt_log("Test Halt Message")
    assert os.path.exists("logs/governance_halts.log")
    with open("logs/governance_halts.log", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test Halt Message" in content


def test_cli_ambiguous_modes() -> None:
    # Test single and multi ambiguity check
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.single = True
    args.multi = True
    args.mode = None
    args.steps = None
    args.excitation = None
    args.task_value = None
    args.seed = 0
    args.quiet = False
    args.demo = False
    args.demo_governance = False

    with patch("sys.exit") as mock_exit:
        ufo_cli.run_simulation(args)
        mock_exit.assert_called_once_with(1)


def test_cli_quiet_and_verbose_single() -> None:
    # Test quiet, verbose, single agent parameters run successfully
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.single = True
    args.multi = False
    args.mode = None
    args.steps = 2
    args.excitation = 0.1
    args.task_value = 0.9
    args.seed = 0
    args.quiet = True
    args.verbose = True
    args.demo = False
    args.demo_governance = False
    args.visualize = True

    with patch("matplotlib.pyplot.savefig"):
        ufo_cli.run_simulation(args)


def test_cli_multi_nominal() -> None:
    # Test multi-agent run
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.single = False
    args.multi = True
    args.mode = None
    args.steps = 2
    args.excitation = 0.1
    args.task_value = 0.9
    args.seed = 0
    args.quiet = False
    args.verbose = True
    args.demo = False
    args.demo_governance = False
    args.visualize = False

    ufo_cli.run_simulation(args)


def test_cli_demo_mode() -> None:
    # Test demo mode
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.single = True
    args.multi = False
    args.mode = None
    args.steps = None
    args.excitation = None
    args.task_value = None
    args.seed = 0
    args.quiet = False
    args.verbose = False
    args.demo = True
    args.demo_governance = False
    args.visualize = False

    ufo_cli.run_simulation(args)


def test_cli_demo_governance_halt() -> None:
    # Test that governance error is cleanly caught and logged
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.single = True
    args.multi = False
    args.mode = None
    args.steps = None
    args.excitation = None
    args.task_value = None
    args.seed = 0
    args.quiet = False
    args.verbose = False
    args.demo = False
    args.demo_governance = True
    args.visualize = False

    with patch("sys.exit") as mock_exit:
        ufo_cli.run_simulation(args)
        mock_exit.assert_called_once_with(1)


def test_main_entry() -> None:
    # Test main function parsing
    with patch("argparse.ArgumentParser.parse_args") as mock_args, \
            patch("ufo_cli.run_simulation") as mock_run:
        mock_args.return_value = argparse.Namespace(
            single=True, multi=False, mode=None, steps=2,
            excitation=0.1, task_value=0.9, seed=0, quiet=False,
            verbose=False, demo=False, demo_governance=False, visualize=False
        )
        ufo_cli.main()
        mock_run.assert_called_once()


def test_main_entry_keyboard_interrupt() -> None:
    # Test keyboard interrupt handling
    with patch("argparse.ArgumentParser.parse_args") as mock_args, \
            patch("ufo_cli.run_simulation", side_effect=KeyboardInterrupt), \
            patch("sys.exit") as mock_exit:
        mock_args.return_value = argparse.Namespace()
        ufo_cli.main()
        mock_exit.assert_called_once_with(0)
