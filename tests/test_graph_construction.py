"""Tests for graph construction."""
import numpy as np
import networkx as nx
from src.graphs.construction import similarity_to_distance, build_threshold_graph, build_mst


def test_angular_distance():
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    D = similarity_to_distance(S, method="angular")
    assert D[0, 0] == 0.0
    assert D[0, 1] > 0.0


def test_threshold_graph_top_k():
    S = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
    G = build_threshold_graph(S, ["A", "B", "C"], top_k=1)
    assert len(G.nodes) == 3


def test_mst_connected():
    D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    G = build_mst(D, ["A", "B", "C"])
    assert nx.is_connected(G)
    assert len(G.edges) == 2
