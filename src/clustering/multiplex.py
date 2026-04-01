"""Multi-layer community detection for multiplex networks."""
from __future__ import annotations

import numpy as np
import networkx as nx


def multiplex_consensus(
    layer_graphs: dict[str, nx.Graph],
    labels: list[str],
    n_runs: int = 50,
    resolution: float = 1.0,
    omega: float = 0.5,
    seed: int = 42,
) -> dict[str, int]:
    """Multiplex community detection via supra-adjacency consensus.

    Combines multiple network layers into a single consensus partition
    by building a fused adjacency matrix and running Leiden on it.

    Parameters
    ----------
    layer_graphs : dict mapping layer name to nx.Graph
    labels : asset ticker list
    n_runs : number of Leiden runs for consensus
    resolution : Leiden resolution parameter
    omega : inter-layer coupling strength
    seed : random seed
    """
    import igraph as ig
    import leidenalg

    n = len(labels)
    node_idx = {label: i for i, label in enumerate(labels)}

    # Build fused adjacency: average edge weights across layers
    fused = np.zeros((n, n))
    n_layers = len(layer_graphs)

    for layer_name, G in layer_graphs.items():
        for u, v, data in G.edges(data=True):
            i, j = node_idx.get(u), node_idx.get(v)
            if i is not None and j is not None:
                w = data.get("weight", 1.0)
                fused[i, j] += w / n_layers
                fused[j, i] += w / n_layers

    # Build igraph from fused adjacency
    ig_graph = ig.Graph.Weighted_Adjacency(
        fused.tolist(), mode="undirected", loops=False
    )

    # Consensus over multiple runs
    consensus_matrix = np.zeros((n, n))
    rng = np.random.RandomState(seed)

    for run in range(n_runs):
        res = resolution * rng.uniform(0.7, 1.3)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=ig_graph.es["weight"],
            resolution_parameter=res,
            seed=seed + run,
        )
        membership = partition.membership
        for a in range(n):
            for b in range(a + 1, n):
                if membership[a] == membership[b]:
                    consensus_matrix[a, b] += 1
                    consensus_matrix[b, a] += 1

    consensus_matrix /= n_runs
    np.fill_diagonal(consensus_matrix, 1.0)

    # Final clustering on consensus matrix via spectral
    from sklearn.cluster import SpectralClustering
    from scipy.linalg import eigvalsh

    # Eigengap heuristic for k
    D_diag = np.diag(consensus_matrix.sum(axis=1))
    L = D_diag - consensus_matrix
    eigenvalues = np.sort(eigvalsh(L, D_diag + 1e-10 * np.eye(n)))
    gaps = np.diff(eigenvalues[:min(15, n)])
    n_clusters = int(np.argmax(gaps[1:]) + 2)
    n_clusters = max(2, min(n_clusters, 10))

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
    )
    cluster_labels = sc.fit_predict(consensus_matrix)

    return {labels[i]: int(cluster_labels[i]) for i in range(n)}


def layer_agreement(
    layer_partitions: dict[str, dict[str, int]],
    labels: list[str],
) -> np.ndarray:
    """Compute pairwise NMI between layer partitions.

    Returns a (n_layers x n_layers) agreement matrix.
    """
    from sklearn.metrics import normalized_mutual_info_score

    layer_names = list(layer_partitions.keys())
    n_layers = len(layer_names)
    agreement = np.ones((n_layers, n_layers))

    for i in range(n_layers):
        p_i = [layer_partitions[layer_names[i]].get(l, 0) for l in labels]
        for j in range(i + 1, n_layers):
            p_j = [layer_partitions[layer_names[j]].get(l, 0) for l in labels]
            nmi = normalized_mutual_info_score(p_i, p_j)
            agreement[i, j] = nmi
            agreement[j, i] = nmi

    return agreement
