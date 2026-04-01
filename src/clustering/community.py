"""Community detection methods."""
from __future__ import annotations

import numpy as np
import networkx as nx


def leiden_communities(
    G: nx.Graph, resolution: float = 1.0, seed: int = 42
) -> dict[str, int]:
    """Leiden algorithm community detection."""
    import igraph as ig
    import leidenalg

    # Convert networkx to igraph
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(mapping))
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_graph.add_edges(edges)

    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    ig_graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )

    return {reverse_mapping[i]: partition.membership[i] for i in range(len(mapping))}


def spectral_communities(
    S: np.ndarray,
    labels: list[str],
    n_clusters: int | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Spectral clustering with eigengap heuristic for k selection."""
    from sklearn.cluster import SpectralClustering

    if n_clusters is None:
        # Eigengap heuristic
        from scipy.linalg import eigvalsh

        # Use similarity as affinity
        affinity = np.abs(S)
        np.fill_diagonal(affinity, 0)
        D = np.diag(affinity.sum(axis=1))
        L = D - affinity
        eigenvalues = np.sort(eigvalsh(L, D + 1e-10 * np.eye(len(S))))
        gaps = np.diff(eigenvalues[:min(15, len(eigenvalues))])
        n_clusters = int(np.argmax(gaps[1:]) + 2)  # +2 because of indexing
        n_clusters = max(2, min(n_clusters, 10))

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
    )
    # Use absolute similarity as affinity
    affinity = np.abs(S)
    np.fill_diagonal(affinity, 0)
    cluster_labels = sc.fit_predict(affinity)

    return {labels[i]: int(cluster_labels[i]) for i in range(len(labels))}


def consensus_communities(
    G: nx.Graph,
    n_runs: int = 100,
    resolution_range: tuple[float, float] = (0.5, 2.0),
    seed: int = 42,
) -> dict[str, int]:
    """Consensus clustering over multiple Leiden runs."""
    import igraph as ig
    import leidenalg

    nodes = list(G.nodes())
    n = len(nodes)
    consensus_matrix = np.zeros((n, n))

    mapping = {node: i for i, node in enumerate(nodes)}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(n)
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_graph.add_edges(edges)
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    ig_graph.es["weight"] = weights

    rng = np.random.RandomState(seed)
    resolutions = rng.uniform(resolution_range[0], resolution_range[1], n_runs)

    for i, res in enumerate(resolutions):
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=res,
            seed=seed + i,
        )
        membership = partition.membership
        for a in range(n):
            for b in range(a + 1, n):
                if membership[a] == membership[b]:
                    consensus_matrix[a, b] += 1
                    consensus_matrix[b, a] += 1

    consensus_matrix /= n_runs
    np.fill_diagonal(consensus_matrix, 1.0)

    # Final clustering on consensus matrix
    from sklearn.cluster import SpectralClustering

    n_clusters = max(2, min(8, n // 4))
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=seed,
    )
    labels_arr = sc.fit_predict(consensus_matrix)

    return {nodes[i]: int(labels_arr[i]) for i in range(n)}
