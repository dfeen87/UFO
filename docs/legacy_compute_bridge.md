# Legacy Compute Bridge Architecture

The Legacy Compute Bridge bridges legacy binary hardware systems with the UFO Deformable Radial Membrane architecture using geometric normalization.

## Architectural Flow

```
+---------------------+       +----------------------+
| Legacy Binary System|       | UFO AI Tensor Field  |
|  (Scalar Execution) |       |  (Radial Membrane)   |
+----------+----------+       +----------+-----------+
           |                             |
           v                             v
      LegacyInput                     AIInput
      (a, b, c)                   (a, b, c, conv)
           |                             |
           +--------------+--------------+
                          |
                          v
                AdmissibilityGate
             (Invariant Handshake)
                          |
             i = (a^2 + b^2) / c^2 ≈ 1
                          |
          +---------------+---------------+
          |                               |
          v                               v
    Handshake Allowed             Handshake Rejected
  (Execute Step / Route)        (GovernanceError / Soft Quarantine)
```

## CLI Interface

Run simulations with Legacy Hardware Handshake enabled:

```bash
poetry run ufo --single --legacy-handshake --legacy-mode strict
```

Run interactive demo:

```bash
python examples/run_legacy_invariant_handshake.py
```
