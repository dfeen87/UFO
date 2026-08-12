# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Centralized Utilities for the UFO Governed Deformable Radial Membrane framework.
"""

from __future__ import annotations
import random
import numpy as np


def set_deterministic_env(seed: int = 0) -> None:
    """
    Sets environment seeds and properties across Python's `random` module
    and NumPy's generator system to ensure absolute reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
