# The Invariant Handshake: Legacy Hardware Adapter

This document details the Python implementation of the geometric stability framework defined in *The Invariant Handshake: Stabilizing AI–Legacy Interoperability Through Geometric Normalization* (Feeney, 2026).

## Overview

The Invariant Handshake provides a geometric permission boundary between legacy binary systems (scalar execution) and modern UFO agentic AI (tensor flows). It evaluates shared channel capacity and ensures cross-era compute interactions remain stable within a bounded geometric surface.

## Core Formula

The geometric invariant ratio $i$ is defined as:

$$i = \frac{a^2 + b^2}{c^2}$$

where:
- **$a$**: Primary stress contribution (scalar activation or Lyapunov energy).
- **$b$**: Secondary stress contribution (compute cost or tensor norm).
- **$c$**: Shared channel capacity (the structural bound / V-Channel pressure).

## Stability Condition

The handshake is admissible if both systems independently satisfy the stability condition:

$$1.0 - \text{tol} < i < 1.0 + \text{tol} \quad (\text{default } \text{tol} = 0.2)$$

$$\text{handshakeAllowed} = \text{legacyStable} \land \text{aiStable}$$

## Mapping UFO Internal Dynamics

| Parameter | Legacy Input (Scalar) | AI Input (Tensor) |
|---|---|---|
| **a** | String activation L2 norm | Lyapunov energy |
| **b** | Estimated compute cost | Tensor activation norm |
| **c** | V-Channel pressure ($c > 0$) | V-Channel pressure ($c > 0$) |
| **conversion** | N/A | Tensor $\to$ Binary representation cost |

## Governance Modes

1. **Strict Mode** (`strict`): Raises `GovernanceError` on handshake failure, halting step execution.
2. **Soft Mode** (`soft`): Applies a secondary $\Delta AG \to \Delta v$ contraction pass before quarantining agent state if still unstable.
3. **Simulation Mode** (`simulation`): Logs failure and flags the step as `legacy-unsafe`.

## Usage Example

```python
from radial_membrane_ai.adapters import LegacyInput, AIInput, handshake

legacy = LegacyInput(a=0.4, b=0.9, c=1.0)
ai = AIInput(a=0.4, b=0.9, c=1.0, conversion=0.05)

status = handshake(legacy, ai, tol=0.2, mode="strict")
print(f"Handshake Allowed: {status.handshakeAllowed}")
```

For engine integration:
```python
from radial_membrane_ai.ufo_engine.single_agent import SingleAgentEngine

engine = SingleAgentEngine(
    seed=0,
    enable_legacy_handshake=True,
    legacy_handshake_mode="strict"
)
```
