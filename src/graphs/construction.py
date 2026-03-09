"""Graph construction from similarity matrices."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree


def similarity_to_distance(S: np.ndarray, method: str = "angular") -> np.ndarray:
    """Convert similarity matrix to distance matrix."""
    if method == "angular":
        # Angular distance: d = sqrt(2(1-rho))
        return np.sqrt(2.0 * np.clip(1.0 - S, 0.0, 2.0))
    elif method == "abs":
        return 1.0 - np.abs(S)
    else:
        raise ValueError(f"Unknown method: {method}")


def build_threshold_graph(
    S: np.ndarray,
    labels: list[str],
    threshold: float | None = None,
    top_k: int | None = None,
) -> nx.Graph:
    """Build sparse graph via threshold or top-K edges per node."""
    n = len(labels)
    G = nx.Graph()
    G.add_nodes_from(labels)

    if top_k is not None:
        for i in range(n):
            # Get top-K strongest connections (excluding self)
            row = S[i].copy()
            row[i] = -np.inf
            top_indices = np.argsort(row)[-top_k:]
            for j in top_indices:
                if S[i, j] > 0:
                    G.add_edge(labels[i], labels[j], weight=float(S[i, j]))
    elif threshold is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(S[i, j]) >= threshold:
                    G.add_edge(labels[i], labels[j], weight=float(S[i, j]))
    else:
        raise ValueError("Must specify threshold or top_k")

    return G


def build_mst(D: np.ndarray, labels: list[str]) -> nx.Graph:
    """Build minimum spanning tree from distance matrix."""
    from scipy.sparse import csr_matrix

    sparse_D = csr_matrix(D)
    mst = minimum_spanning_tree(sparse_D)
    mst_dense = mst.toarray()

    G = nx.Graph()
    G.add_nodes_from(labels)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = mst_dense[i, j] + mst_dense[j, i]
            if w > 0:
                G.add_edge(labels[i], labels[j], weight=float(w))

    return G


def build_multilayer_graph(
    layers: dict[str, np.ndarray],
    labels: list[str],
    top_k: int = 5,
) -> dict[str, nx.Graph]:
    """Build graph for each similarity layer."""
    result = {}
    for layer_name, S in layers.items():
        result[layer_name] = build_threshold_graph(S, labels, top_k=top_k)
    return result
