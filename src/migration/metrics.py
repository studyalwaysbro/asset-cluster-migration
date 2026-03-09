"""Novel migration and topology deformation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wasserstein_distance
from sklearn.metrics import normalized_mutual_info_score


@dataclass
class MigrationSnapshot:
    """Container for migration metrics at a single time step."""
    date: pd.Timestamp
    assignments: dict[str, int]
    cmi: float = 0.0
    amf: dict[str, float] = field(default_factory=dict)
    cps: dict[int, float] = field(default_factory=dict)
    tds: float = 0.0
    bridge_scores: dict[str, float] = field(default_factory=dict)


def cluster_migration_index(
    current: dict[str, int], previous: dict[str, int]
) -> float:
    """Fraction of assets that changed cluster assignment."""
    common = set(current.keys()) & set(previous.keys())
    if not common:
        return 0.0
    changed = sum(1 for a in common if current[a] != previous[a])
    return changed / len(common)


def asset_migration_frequency(
    history: list[dict[str, int]],
) -> dict[str, float]:
    """Cumulative migration frequency per asset over history."""
    if len(history) < 2:
        return {}

    all_assets = set()
    for h in history:
        all_assets.update(h.keys())

    counts: dict[str, int] = {a: 0 for a in all_assets}
    total_steps = len(history) - 1

    for t in range(1, len(history)):
        for asset in all_assets:
            if asset in history[t] and asset in history[t - 1]:
                if history[t][asset] != history[t - 1][asset]:
                    counts[asset] += 1

    return {a: counts[a] / total_steps for a in all_assets}


def cluster_persistence_score(
    current: dict[str, int], previous: dict[str, int]
) -> dict[int, float]:
    """Jaccard similarity of each cluster membership over consecutive steps."""
    # Get cluster sets for current
    curr_clusters: dict[int, set[str]] = {}
    for asset, cid in current.items():
        curr_clusters.setdefault(cid, set()).add(asset)

    # Get cluster sets for previous
    prev_clusters: dict[int, set[str]] = {}
    for asset, cid in previous.items():
        prev_clusters.setdefault(cid, set()).add(asset)

    # For each current cluster, find best-matching previous cluster
    scores: dict[int, float] = {}
    for cid, members in curr_clusters.items():
        best_jaccard = 0.0
        for prev_members in prev_clusters.values():
            intersection = len(members & prev_members)
            union = len(members | prev_members)
            if union > 0:
                jaccard = intersection / union
                best_jaccard = max(best_jaccard, jaccard)
        scores[cid] = best_jaccard

    return scores


def topology_deformation_score(
    current_graph: nx.Graph,
    baseline_graph: nx.Graph,
    current_communities: dict[str, int],
    baseline_communities: dict[str, int],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    """Composite topology deformation: Wasserstein degree + NMI community + spectral."""
    alpha, beta, gamma = weights

    # 1. Wasserstein distance on degree distributions
    curr_degrees = sorted([d for _, d in current_graph.degree()])
    base_degrees = sorted([d for _, d in baseline_graph.degree()])
    if curr_degrees and base_degrees:
        w_degree = wasserstein_distance(curr_degrees, base_degrees)
        # Normalize by max degree
        max_deg = max(max(curr_degrees, default=1), max(base_degrees, default=1))
        w_degree = w_degree / max_deg if max_deg > 0 else 0
    else:
        w_degree = 0.0

    # 2. 1 - NMI on community assignments
    common = sorted(set(current_communities.keys()) & set(baseline_communities.keys()))
    if len(common) > 1:
        curr_labels = [current_communities[a] for a in common]
        base_labels = [baseline_communities[a] for a in common]
        nmi = normalized_mutual_info_score(base_labels, curr_labels)
        j_community = 1.0 - nmi
    else:
        j_community = 1.0

    # 3. Spectral distance on graph Laplacians
    from src.graphs.topology import graph_laplacian_eigenvalues

    try:
        curr_eig = graph_laplacian_eigenvalues(current_graph)
        base_eig = graph_laplacian_eigenvalues(baseline_graph)
        # Pad to same length
        max_len = max(len(curr_eig), len(base_eig))
        curr_eig = np.pad(curr_eig, (0, max_len - len(curr_eig)))
        base_eig = np.pad(base_eig, (0, max_len - len(base_eig)))
        s_spectral = np.linalg.norm(curr_eig - base_eig) / max(max_len, 1)
    except Exception:
        s_spectral = 0.0

    return alpha * w_degree + beta * j_community + gamma * s_spectral
