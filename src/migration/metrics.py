"""Novel migration and topology deformation metrics.

v0.4.0 — Fixed CMI permutation invariance (Hungarian matching),
TDS component scaling (z-score normalization), CPS bidirectional matching,
spectral distance via Wasserstein instead of zero-padded L2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
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


def _match_cluster_labels(
    current: dict[str, int], previous: dict[str, int]
) -> dict[int, int]:
    """Find optimal cluster label mapping using Hungarian algorithm.

    Solves the permutation invariance problem: Leiden can relabel clusters
    arbitrarily between windows. This finds the bijection between current
    and previous cluster IDs that minimizes total migration.

    Returns mapping: current_id -> matched_previous_id.
    """
    common = set(current.keys()) & set(previous.keys())
    if not common:
        return {}

    curr_ids = sorted(set(current[a] for a in common))
    prev_ids = sorted(set(previous[a] for a in common))

    # Build cost matrix: cost[i,j] = number of assets NOT shared between
    # current cluster i and previous cluster j
    n_curr, n_prev = len(curr_ids), len(prev_ids)
    max_dim = max(n_curr, n_prev)
    cost = np.zeros((max_dim, max_dim))

    # Build membership sets
    curr_sets = {cid: set() for cid in curr_ids}
    prev_sets = {pid: set() for pid in prev_ids}
    for a in common:
        curr_sets[current[a]].add(a)
        prev_sets[previous[a]].add(a)

    for i, cid in enumerate(curr_ids):
        for j, pid in enumerate(prev_ids):
            overlap = len(curr_sets[cid] & prev_sets[pid])
            cost[i, j] = len(common) - overlap  # minimize non-overlap

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_curr and c < n_prev:
            mapping[curr_ids[r]] = prev_ids[c]

    return mapping


def cluster_migration_index(
    current: dict[str, int], previous: dict[str, int]
) -> float:
    """Fraction of assets that changed cluster assignment.

    Uses Hungarian matching to solve cluster relabeling invariance:
    if Leiden assigns cluster IDs {0,1,2} in one window and {2,0,1}
    in the next (same structure, different labels), CMI correctly = 0.
    """
    common = sorted(set(current.keys()) & set(previous.keys()))
    if not common:
        return 0.0

    # Find optimal label matching
    label_map = _match_cluster_labels(current, previous)

    changed = 0
    for a in common:
        curr_label = current[a]
        prev_label = previous[a]
        # Map current label to matched previous label
        matched_prev = label_map.get(curr_label)
        if matched_prev is None or matched_prev != prev_label:
            changed += 1

    return changed / len(common)


def asset_migration_frequency(
    history: list[dict[str, int]],
) -> dict[str, float]:
    """Cumulative migration frequency per asset over history.

    Uses Hungarian matching at each step to handle cluster relabeling.
    """
    if len(history) < 2:
        return {}

    all_assets = set()
    for h in history:
        all_assets.update(h.keys())

    counts: dict[str, int] = {a: 0 for a in all_assets}
    appearances: dict[str, int] = {a: 0 for a in all_assets}

    for t in range(1, len(history)):
        label_map = _match_cluster_labels(history[t], history[t - 1])
        for asset in all_assets:
            if asset in history[t] and asset in history[t - 1]:
                appearances[asset] += 1
                curr_label = history[t][asset]
                prev_label = history[t - 1][asset]
                matched_prev = label_map.get(curr_label)
                if matched_prev is None or matched_prev != prev_label:
                    counts[asset] += 1

    return {
        a: counts[a] / max(appearances[a], 1)
        for a in all_assets
    }


def cluster_persistence_score(
    current: dict[str, int], previous: dict[str, int]
) -> dict[int, float]:
    """Jaccard similarity of cluster membership using Hungarian matching.

    Bidirectional: matches current clusters to previous clusters optimally,
    avoiding the many-to-one matching bug in naive best-Jaccard approach.
    """
    label_map = _match_cluster_labels(current, previous)

    # Build membership sets
    curr_clusters: dict[int, set[str]] = {}
    for asset, cid in current.items():
        curr_clusters.setdefault(cid, set()).add(asset)

    prev_clusters: dict[int, set[str]] = {}
    for asset, cid in previous.items():
        prev_clusters.setdefault(cid, set()).add(asset)

    scores: dict[int, float] = {}
    for cid, members in curr_clusters.items():
        matched_pid = label_map.get(cid)
        if matched_pid is not None and matched_pid in prev_clusters:
            prev_members = prev_clusters[matched_pid]
            intersection = len(members & prev_members)
            union = len(members | prev_members)
            scores[cid] = intersection / union if union > 0 else 0.0
        else:
            scores[cid] = 0.0  # new cluster, no persistence

    return scores


class TDSNormalizer:
    """Running z-score normalizer for TDS components.

    Solves the incommensurability problem: Wasserstein degree, NMI distance,
    and spectral distance have different natural scales. This tracks running
    mean/std of each component and converts to z-scores before combining.
    """

    def __init__(self):
        self._history: dict[str, list[float]] = {
            "w_degree": [],
            "j_community": [],
            "s_spectral": [],
        }

    def normalize(
        self, w_degree: float, j_community: float, s_spectral: float
    ) -> tuple[float, float, float]:
        """Add raw values and return z-scored values."""
        self._history["w_degree"].append(w_degree)
        self._history["j_community"].append(j_community)
        self._history["s_spectral"].append(s_spectral)

        result = []
        for key, val in [("w_degree", w_degree),
                         ("j_community", j_community),
                         ("s_spectral", s_spectral)]:
            arr = np.array(self._history[key])
            if len(arr) < 3 or arr.std() < 1e-10:
                result.append(val)
            else:
                result.append((val - arr.mean()) / arr.std())

        return tuple(result)


def topology_deformation_score(
    current_graph: nx.Graph,
    baseline_graph: nx.Graph,
    current_communities: dict[str, int],
    baseline_communities: dict[str, int],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    normalizer: TDSNormalizer | None = None,
) -> float:
    """Composite topology deformation score.

    Components:
    1. Wasserstein distance on degree distributions (normalized by n-1)
    2. 1 - NMI on community assignments (permutation-invariant by definition)
    3. Wasserstein distance on Laplacian spectra (replaces zero-padded L2)

    If a TDSNormalizer is provided, components are z-scored to ensure
    commensurability before weighted combination.
    """
    alpha, beta, gamma = weights

    # 1. Wasserstein distance on degree distributions
    n_nodes = max(
        current_graph.number_of_nodes(),
        baseline_graph.number_of_nodes(),
        2,
    )
    curr_degrees = sorted([d for _, d in current_graph.degree()])
    base_degrees = sorted([d for _, d in baseline_graph.degree()])
    if curr_degrees and base_degrees:
        w_degree = wasserstein_distance(curr_degrees, base_degrees)
        w_degree = w_degree / (n_nodes - 1)  # normalize by max possible degree
    else:
        w_degree = 0.0

    # 2. 1 - NMI on community assignments (NMI is already permutation-invariant)
    common = sorted(
        set(current_communities.keys()) & set(baseline_communities.keys())
    )
    if len(common) > 1:
        curr_labels = [current_communities[a] for a in common]
        base_labels = [baseline_communities[a] for a in common]
        nmi = normalized_mutual_info_score(base_labels, curr_labels)
        j_community = 1.0 - nmi
    else:
        j_community = 1.0

    # 3. Wasserstein distance on Laplacian spectra
    from src.graphs.topology import graph_laplacian_eigenvalues

    try:
        curr_eig = graph_laplacian_eigenvalues(current_graph)
        base_eig = graph_laplacian_eigenvalues(baseline_graph)
        # Use Wasserstein distance on spectra instead of zero-padded L2
        # This handles different-sized graphs naturally
        s_spectral = wasserstein_distance(curr_eig, base_eig)
        # Normalize: eigenvalues of normalized Laplacian are in [0, 2]
        s_spectral = s_spectral / 2.0
    except Exception:
        s_spectral = 0.0

    # Optional z-score normalization for commensurability
    if normalizer is not None:
        w_degree, j_community, s_spectral = normalizer.normalize(
            w_degree, j_community, s_spectral
        )

    return alpha * w_degree + beta * j_community + gamma * s_spectral
