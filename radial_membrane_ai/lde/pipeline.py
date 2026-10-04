# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Pipeline for Letter‑Depth Encoding (L.D.E.) converting raw text to a governed symbolic state.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from radial_membrane_ai.lde.models import (
    LDEConfig,
    LDEString,
    LDEVChannel,
    LDEBoundaryGeometry,
    LDEState
)
from radial_membrane_ai.exceptions import ReconstructionError


def lde_encode(text: str, config: Optional[LDEConfig] = None) -> LDEState:
    """
    Executes the full Letter-Depth Encoding pipeline on the input raw text.
    """
    if config is None:
        config = LDEConfig()

    sigma = config.sigma
    N_sigma = len(sigma)

    # 1. Extract reconstruction metadata and build normalized stream
    inserts: Dict[int, str] = {}
    casing: List[bool] = []
    norm_chars: List[str] = []
    accum: List[str] = []
    raw_positions: Dict[str, List[int]] = {l: [] for l in sigma}

    for idx, ch in enumerate(text):
        ch_lower = ch.lower()
        if ch_lower in raw_positions:  # is a valid symbol in Sigma
            if accum:
                inserts[len(norm_chars)] = "".join(accum)
                accum = []
            norm_chars.append(ch_lower)
            casing.append(ch.isupper())
            raw_positions[ch_lower].append(idx)
        else:
            accum.append(ch)

    if accum:
        inserts[len(norm_chars)] = "".join(accum)

    T_n = "".join(norm_chars)
    N = len(T_n)

    # 2. Build letter position maps (1-based for L.D.E.)
    positions: Dict[str, List[int]] = {l: [] for l in sigma}
    counts: Dict[str, int] = {l: 0 for l in sigma}
    for idx, char in enumerate(norm_chars):
        positions[char].append(idx + 1)
        counts[char] += 1

    # 3. Compute local metrics for each letter string
    activation: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    boundary_participation: Dict[str, float] = {}
    repetition_pressure: Dict[str, float] = {}
    local_tension: Dict[str, float] = {}
    stiffness: Dict[str, float] = {}
    reconstruction_cost: Dict[str, float] = {}

    max_count = max(counts.values()) if counts and any(c > 0 for c in counts.values()) else 0

    for l in sigma:
        n_l = counts[l]
        P_l = positions[l]

        # Activation
        activation[l] = n_l / N if N > 0 else 0.0

        # Spread
        if n_l <= 1:
            spread[l] = 0.0
        else:
            spread[l] = (max(P_l) - min(P_l)) / max(1, N - 1)

        # Boundary Participation
        boundary_occurrences = 0
        for raw_idx in raw_positions[l]:
            is_first = (raw_idx == 0 or not text[raw_idx - 1].isalpha())
            is_last = (raw_idx == len(text) - 1 or not text[raw_idx + 1].isalpha())
            if is_first or is_last:
                boundary_occurrences += 1
        boundary_participation[l] = boundary_occurrences / max(1, n_l)

        # Repetition Pressure
        gaps = []
        if n_l >= 2:
            for k in range(len(P_l) - 1):
                gaps.append(P_l[k+1] - P_l[k])

        if len(gaps) < 1:
            repetition_pressure[l] = 0.0
        else:
            repetition_pressure[l] = float(1.0 / np.mean(gaps))

        # Local Tension
        if len(gaps) < 2:
            local_tension[l] = repetition_pressure[l]
        else:
            local_tension[l] = float(np.std(gaps) / (np.mean(gaps) + config.epsilon))

        # Stiffness
        stiffness[l] = 1.0 / (1.0 + spread[l])

        # Reconstruction Cost
        if max_count > 0:
            reconstruction_cost[l] = 1.0 - (n_l / max_count)
        else:
            reconstruction_cost[l] = 0.0

    # 4. Compute adjacency pressure and coherence matrix
    adjacency_pressure: Dict[Tuple[str, str], float] = {}
    coherence_matrix: Dict[Tuple[str, str], float] = {}

    # Initialize all pairs to 0.0
    for i in sigma:
        for j in sigma:
            adjacency_pressure[(i, j)] = 0.0
            coherence_matrix[(i, j)] = 0.0

    # Count consecutive transitions
    for idx in range(N - 1):
        char_i = norm_chars[idx]
        char_j = norm_chars[idx + 1]
        if char_i in sigma and char_j in sigma:
            adjacency_pressure[(char_i, char_j)] += 1.0

    for i in sigma:
        for j in sigma:
            n_i = counts[i]
            A_ij = adjacency_pressure[(i, j)]
            coherence_matrix[(i, j)] = A_ij / max(1, n_i)

    # 5. Compute Channel Contribution
    channel_contribution: Dict[str, float] = {}
    active_other_letters = [j for j in sigma if counts[j] > 0]

    for l in sigma:
        others = [j for j in active_other_letters if j != l]
        if not others:
            channel_contribution[l] = 0.0
        else:
            coh_sum = sum(coherence_matrix[(l, j)] for j in others)
            channel_contribution[l] = coh_sum / len(others)

    # 6. Compute letter-depth for each symbol l
    depth: Dict[str, float] = {}
    for l in sigma:
        # F_l is normalized frequency: n_l / max_count
        F_l = counts[l] / max_count if max_count > 0 else 0.0
        S_l = spread[l]
        B_l = boundary_participation[l]
        R_l = repetition_pressure[l]
        C_l = channel_contribution[l]

        depth[l] = (
            config.w_f * F_l
            + config.w_s * S_l
            + config.w_b * B_l
            + config.w_r * R_l
            + config.w_c * C_l
        )

    # 7. Compute channel pressures and assemble LDEVChannels
    channels: List[LDEVChannel] = []
    links: Dict[str, Dict[str, float]] = {l: {} for l in sigma}

    for i in sigma:
        for j in sigma:
            A_ij = adjacency_pressure[(i, j)]
            Q_ij = coherence_matrix[(i, j)]
            if A_ij > 0 or Q_ij > 0:
                idx_i = sigma.index(i)
                idx_j = sigma.index(j)
                theta_i = (2.0 * math.pi * idx_i) / N_sigma
                theta_j = (2.0 * math.pi * idx_j) / N_sigma
                alpha_ij = (1.0 + math.cos(theta_i - theta_j)) / 2.0
                gamma_j = 1.0 / (1.0 + reconstruction_cost[j])

                # Sigmoid of target activation
                a_j = activation[j]
                sigmoid_a_j = 1.0 / (1.0 + math.exp(-a_j))

                P_i_to_j = sigmoid_a_j * alpha_ij * Q_ij * gamma_j

                channels.append(
                    LDEVChannel(
                        source=i,
                        target=j,
                        adjacency=A_ij,
                        coherence=Q_ij,
                        pressure=P_i_to_j
                    )
                )
                links[i][j] = P_i_to_j

    # 8. Compute boundary geometry (configurable sample angles)
    # Basis function: phi_l(theta) = max(0, cos(angular_distance(theta, theta_l)))
    radius_map: List[float] = []
    samples = config.geometry_samples
    dtheta = (2.0 * math.pi) / samples

    for k in range(samples):
        theta = k * dtheta
        r_val = config.r_0
        for l in sigma:
            d_l = depth[l]
            if d_l > 0:
                idx_l = sigma.index(l)
                theta_l = (2.0 * math.pi * idx_l) / N_sigma

                # angular distance
                diff = abs(theta - theta_l)
                ang_dist = min(diff, 2.0 * math.pi - diff)
                phi_l = max(0.0, math.cos(ang_dist))
                r_val += d_l * phi_l
        radius_map.append(r_val)

    # Tangent & Curvature via numerical differentiation with wrap-around
    tangent_map: List[float] = []
    curvature_map: List[float] = []

    for k in range(samples):
        r_k = radius_map[k]
        r_prev = radius_map[(k - 1) % samples]
        r_next = radius_map[(k + 1) % samples]

        # tangent = (r_{k+1} - r_{k-1}) / (2 * dtheta)
        tangent_val = (r_next - r_prev) / (2.0 * dtheta)
        tangent_map.append(tangent_val)

        # curvature = (r_{k+1} - 2*r_k + r_{k-1}) / (dtheta^2)
        curvature_val = (r_next - 2.0 * r_k + r_prev) / (dtheta ** 2)
        curvature_map.append(curvature_val)

    # Asymmetry
    r_min = min(radius_map)
    r_max = max(radius_map)
    asymmetry = 1.0 - (r_min / r_max) if r_max > 0.0 else 0.0

    boundary = LDEBoundaryGeometry(
        radius_map=radius_map,
        curvature_map=curvature_map,
        tangent_map=tangent_map,
        asymmetry=asymmetry
    )

    # 9. Assemble LDEStrings
    strings: Dict[str, LDEString] = {}
    for l in sigma:
        idx_l = sigma.index(l)
        theta_l = (2.0 * math.pi * idx_l) / N_sigma
        strings[l] = LDEString(
            activation=activation[l],
            phase=theta_l,
            positions=positions[l],
            spread=spread[l],
            depth=depth[l],
            tension=local_tension[l],
            stiffness=stiffness[l],
            cost=reconstruction_cost[l],
            links=links[l]
        )

    # 10. Reconstruction metadata map
    reconstruction_map = {
        "mode": config.rho,
        "inserts": inserts,
        "casing": casing,
        "normalized_stream": T_n
    }

    # 11. Verify reconstruction if required
    if config.rho == "full-reconstruction":
        reconstructed = []
        for idx in range(N + 1):
            if idx in inserts:
                reconstructed.append(inserts[idx])
            if idx < N:
                char = norm_chars[idx]
                if casing[idx]:
                    char = char.upper()
                reconstructed.append(char)
        reconstructed_text = "".join(reconstructed)
        if reconstructed_text != text:
            raise ReconstructionError("Reconstruction verification failed!")

    return LDEState(
        strings=strings,
        coherence_matrix=coherence_matrix,
        channels=channels,
        boundary=boundary,
        reconstruction_map=reconstruction_map
    )


def lde_reconstruct(state: LDEState) -> str:
    """Reconstruct text from a full-reconstruction state.

    Letter-depth geometry is not sufficient for lossless reconstruction.  This
    operator therefore requires the ordering, casing, and non-alphabet-symbol
    evidence retained by :func:`lde_encode`, and refuses compressed states.
    """
    metadata = state.reconstruction_map
    if metadata.get("mode") != "full-reconstruction":
        raise ReconstructionError("Compressed L.D.E. states are not lossless.")
    stream = metadata.get("normalized_stream")
    casing = metadata.get("casing")
    inserts = metadata.get("inserts")
    if not isinstance(stream, str) or not isinstance(casing, list) or not isinstance(inserts, dict):
        raise ReconstructionError("Malformed full-reconstruction metadata.")
    if len(casing) != len(stream) or any(type(flag) is not bool for flag in casing):
        raise ReconstructionError("Malformed casing metadata.")
    if any(type(index) is not int or not isinstance(value, str) for index, value in inserts.items()):
        raise ReconstructionError("Malformed insertion metadata.")
    if any(index < 0 or index > len(stream) for index in inserts):
        raise ReconstructionError("Insertion index is outside the normalized stream.")

    reconstructed: List[str] = []
    for index in range(len(stream) + 1):
        reconstructed.append(inserts.get(index, ""))
        if index < len(stream):
            symbol = stream[index]
            reconstructed.append(symbol.upper() if casing[index] else symbol)
    return "".join(reconstructed)
