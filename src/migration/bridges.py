"""Bridge-asset detection during stress episodes."""
from __future__ import annotations

import networkx as nx
import numpy as np


def bridge_score(
    G: nx.Graph,
    amf: dict[str, float],
) -> dict[str, float]:
    """Compute bridge score: betweenness * AMF * (1 - clustering_coeff)."""
    betweenness = nx.betweenness_centrality(G, weight="weight")
    clustering = nx.clustering(G, weight="weight")

    scores = {}
    for node in G.nodes():
        b = betweenness.get(node, 0.0)
        a = amf.get(node, 0.0)
        c = clustering.get(node, 0.0)
        scores[node] = b * a * (1.0 - c)

    return scores


def top_bridges(
    scores: dict[str, float], n: int = 10
) -> list[tuple[str, float]]:
    """Return top-N bridge assets by score."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
