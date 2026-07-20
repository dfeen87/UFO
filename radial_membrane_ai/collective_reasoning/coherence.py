"""
Emergent Coherence Detection for U.F.O. collective reasoning.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Sequence, List, Set, Any, Dict, Tuple

from radial_membrane_ai.multi_agent.agent import UFOAgent
from radial_membrane_ai.multi_agent.cluster import UFOCluster
from radial_membrane_ai.shard import ShardState


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D arrays. Returns 1.0 if either is all zeros.
    """
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x < 1e-9 or norm_y < 1e-9:
        return 1.0
    return float(np.dot(x, y) / (norm_x * norm_y))


def jaccard_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    """
    Computes Jaccard similarity between two sets. Returns 1.0 if both are empty.
    """
    if not s1 and not s2:
        return 1.0
    u = s1.union(s2)
    if not u:
        return 1.0
    return len(s1.intersection(s2)) / len(u)


def value_similarity(v1: Any, v2: Any) -> float:
    """
    Measures the similarity of two semantic memory values.
    Handles numpy arrays, strings, numbers, and fallback types.
    """
    if type(v1) != type(v2):
        return 0.0

    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        # Cosine similarity for arrays
        if v1.shape == v2.shape:
            return cosine_similarity(v1.flatten(), v2.flatten())
        return 0.0

    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        # Try converting to float arrays
        try:
            arr1 = np.array(v1, dtype=np.float64)
            arr2 = np.array(v2, dtype=np.float64)
            if arr1.shape == arr2.shape:
                return cosine_similarity(arr1, arr2)
        except Exception:
            pass
        # Fallback to Jaccard similarity of elements
        return jaccard_similarity(set(v1), set(v2))

    if isinstance(v1, str) and isinstance(v2, str):
        # Character overlap / exact match
        if v1 == v2:
            return 1.0
        words1 = set(v1.split())
        words2 = set(v2.split())
        return jaccard_similarity(words1, words2)

    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        # Absolute difference relative similarity
        denom = max(abs(v1), abs(v2), 1e-9)
        return float(1.0 - abs(v1 - v2) / denom)

    # Exact match fallback
    return 1.0 if v1 == v2 else 0.0


def compute_semantic_memory_coherence(entity_memories: List[Any]) -> float:
    """
    Computes Cm (semantic memory coherence) across a list of Agent or Cluster semantic memories.
    Combines tag overlap (Jaccard) and value similarity.
    """
    if len(entity_memories) < 2:
        return 1.0

    scores = []
    # Pairwise comparison
    for i in range(len(entity_memories)):
        for j in range(i + 1, len(entity_memories)):
            m1 = entity_memories[i]
            m2 = entity_memories[j]

            # Get stores depending on agent (local_store) or cluster (global_store)
            store1 = getattr(m1, "local_store", getattr(m1, "global_store", {}))
            store2 = getattr(m2, "local_store", getattr(m2, "global_store", {}))

            common_keys = set(store1.keys()).intersection(store2.keys())
            if not common_keys:
                # If there are no common keys, but stores are empty, they are coherent
                if not store1 and not store2:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
                continue

            pair_scores = []
            for key in common_keys:
                rec1 = store1[key]
                rec2 = store2[key]
                t_sim = jaccard_similarity(rec1.tags, rec2.tags)
                v_sim = value_similarity(rec1.value, rec2.value)
                pair_scores.append(0.5 * t_sim + 0.5 * v_sim)

            scores.append(float(np.mean(pair_scores)) if pair_scores else 1.0)

    return float(np.mean(scores)) if scores else 1.0


def compute_tension_curvature_coherence(entities: List[Any]) -> float:
    """
    Computes Ct (tension/curvature coherence) across a list of agents/clusters.
    Compares curvature trajectories and tension history vectors.
    """
    if len(entities) < 2:
        return 1.0

    scores = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i]
            e2 = entities[j]

            # Extract queues from temporal membrane states
            t1 = getattr(e1.membrane, "temporal_state", None) if hasattr(e1, "membrane") else None
            t2 = getattr(e2.membrane, "temporal_state", None) if hasattr(e2, "membrane") else None

            # Get tension histories
            tens1 = list(t1.tension_history) if t1 else []
            tens2 = list(t2.tension_history) if t2 else []

            # Curvatures
            curv1 = list(t1.curvature_history) if t1 else []
            curv2 = list(t2.curvature_history) if t2 else []

            # Vector pad/align
            max_len = max(len(tens1), len(tens2))
            if max_len > 0:
                tens1_pad = np.pad(tens1, (0, max_len - len(tens1)), mode='constant') if tens1 else np.zeros(max_len)
                tens2_pad = np.pad(tens2, (0, max_len - len(tens2)), mode='constant') if tens2 else np.zeros(max_len)
                t_sim = cosine_similarity(tens1_pad, tens2_pad)
            else:
                t_sim = 1.0

            max_len_c = max(len(curv1), len(curv2))
            if max_len_c > 0:
                curv1_pad = np.pad(curv1, (0, max_len_c - len(curv1)), mode='constant') if curv1 else np.zeros(max_len_c)
                curv2_pad = np.pad(curv2, (0, max_len_c - len(curv2)), mode='constant') if curv2 else np.zeros(max_len_c)
                c_sim = cosine_similarity(curv1_pad, curv2_pad)
            else:
                c_sim = 1.0

            scores.append(0.5 * t_sim + 0.5 * c_sim)

    return float(np.mean(scores)) if scores else 1.0


def compute_policy_sao_coherence(entities: List[Any]) -> float:
    """
    Computes Cp (policy/SAO pattern coherence) across a list of agents/clusters.
    Measures similarity of active policy envelopes and SAO promotion patterns.
    """
    if len(entities) < 2:
        return 1.0

    scores = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i]
            e2 = entities[j]

            # 1. Compare Policy Envelopes
            env1 = getattr(e1, "policy_envelope", None)
            env2 = getattr(e2, "policy_envelope", None)

            p_sim = 1.0
            if env1 and env2:
                # Numerical attributes comparison
                diff_trust = abs(env1.min_trust - env2.min_trust)
                diff_cost = abs(env1.max_cost_band - env2.max_cost_band) / 2.0
                p_sim = max(0.0, 1.0 - (0.5 * diff_trust + 0.5 * diff_cost))
            elif env1 or env2:
                p_sim = 0.5

            # 2. Compare SAO patterns (e.g. comparing quality or promotion frequency/capacity)
            cap1 = getattr(e1.shard, "capacity", 1.0) if hasattr(e1, "shard") else 1.0
            cap2 = getattr(e2.shard, "capacity", 1.0) if hasattr(e2, "shard") else 1.0
            cap_sim = 1.0 - abs(cap1 - cap2)

            scores.append(0.6 * p_sim + 0.4 * cap_sim)

    return float(np.mean(scores)) if scores else 1.0


def coherence_score(
    entities: Sequence[Any],
    w_m: float = 0.4,
    w_t: float = 0.3,
    w_p: float = 0.3
) -> float:
    """
    Computes the composite collective coherence score C_collective(t).

    C_collective = w_m * C_m + w_t * C_t + w_p * C_p
    """
    if not entities:
        return 1.0

    # Extract memory stores
    memories = [getattr(e, "semantic_memory") for e in entities if getattr(e, "semantic_memory", None) is not None]

    c_m = compute_semantic_memory_coherence(memories)
    c_t = compute_tension_curvature_coherence(list(entities))
    c_p = compute_policy_sao_coherence(list(entities))

    score = w_m * c_m + w_t * c_t + w_p * c_p
    return float(np.clip(score, 0.0, 1.0))


def cluster_coherence_score(cluster: UFOCluster) -> float:
    """
    Computes coherence across all active agents within a cluster.
    """
    active_agents = [a for a in cluster.agents if a.shard.state != ShardState.QUARANTINED]
    if not active_agents:
        return 1.0
    return coherence_score(active_agents)


def global_mesh_coherence_score(clusters: Sequence[UFOCluster]) -> Tuple[float, float]:
    """
    Aggregates cluster coherence scores.
    Returns (mean_coherence, max_tension_or_warning_value).
    """
    if not clusters:
        return 1.0, 0.0

    cluster_scores = [cluster_coherence_score(c) for c in clusters]
    mean_coherence = float(np.mean(cluster_scores))

    # Get max tension across clusters as a warning signal
    max_tension = float(np.max([c.tension_metric for c in clusters])) if clusters else 0.0

    return mean_coherence, max_tension
