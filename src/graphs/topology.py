"""Graph topology diagnostics."""
from __future__ import annotations

import networkx as nx
import numpy as np


def compute_centrality_metrics(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute multiple centrality measures for all nodes."""
    return {
        "degree": dict(nx.degree_centrality(G)),
        "betweenness": dict(nx.betweenness_centrality(G, weight="weight")),
        "eigenvector": dict(
            nx.eigenvector_centrality_numpy(G, weight="weight")
            if len(G.edges) > 0
            else {n: 0.0 for n in G.nodes}
        ),
        "closeness": dict(nx.closeness_centrality(G)),
    }


def compute_modularity(G: nx.Graph, communities: dict[str, int]) -> float:
    """Compute Newman modularity for given community assignment."""
    # Convert to list of sets
    comm_sets: dict[int, set] = {}
    for node, comm_id in communities.items():
        comm_sets.setdefault(comm_id, set()).add(node)
    partition = list(comm_sets.values())
    return nx.algorithms.community.modularity(G, partition)


def graph_density(G: nx.Graph) -> float:
    return nx.density(G)


def mean_clustering_coefficient(G: nx.Graph) -> float:
    return nx.average_clustering(G, weight="weight")


def graph_laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """Compute eigenvalues of the normalized graph Laplacian."""
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.sort(np.real(np.linalg.eigvals(L)))
    return eigenvalues
