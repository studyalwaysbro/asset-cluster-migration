"""Graph topology diagnostics."""
from __future__ import annotations

import networkx as nx
import numpy as np


def _safe_eigenvector_centrality(G: nx.Graph) -> dict[str, float]:
    """Eigenvector centrality with fallback for disconnected graphs."""
    if len(G.edges) == 0:
        return {n: 0.0 for n in G.nodes}
    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except (nx.NetworkXException, nx.AmbiguousSolution):
        # For disconnected graphs, compute per-component and normalize
        result = {n: 0.0 for n in G.nodes}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph.edges) > 0:
                try:
                    sub_cent = nx.eigenvector_centrality_numpy(subgraph, weight="weight")
                    scale = len(component) / len(G.nodes)
                    for node, val in sub_cent.items():
                        result[node] = val * scale
                except Exception:
                    pass
        return result


def compute_centrality_metrics(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute multiple centrality measures for all nodes."""
    return {
        "degree": dict(nx.degree_centrality(G)),
        "betweenness": dict(nx.betweenness_centrality(G, weight="weight")),
        "eigenvector": dict(_safe_eigenvector_centrality(G)),
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
