"""Graph filtering methods (MST, PMFG)."""
from __future__ import annotations

import numpy as np
import networkx as nx


def build_pmfg(S: np.ndarray, labels: list[str]) -> nx.Graph:
    """Build Planar Maximally Filtered Graph from similarity matrix.

    The PMFG keeps 3(n-2) edges (vs n-1 for MST) while maintaining planarity.
    Algorithm: greedily add edges by decreasing similarity, checking planarity.
    """
    n = len(labels)
    max_edges = 3 * (n - 2)

    # Build sorted edge list (descending similarity)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, S[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, j, w in edges:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(i, j, weight=float(w))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(i, j)

    # Relabel nodes
    mapping = {i: labels[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    return G
