"""Network graph visualization."""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.colors
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "equity_us": "#1f77b4",
    "sector": "#2ca02c",
    "equity_intl": "#ff7f0e",
    "equity_em": "#d62728",
    "bond_govt": "#9467bd",
    "bond_credit": "#8c564b",
    "bond_em": "#e377c2",
    "commodity": "#bcbd22",
    "real_asset": "#17becf",
    "fx": "#7f7f7f",
    "volatility": "#aec7e8",
    "thematic": "#ff9896",
    "crypto": "#ffbb78",
    "inflation": "#98df8a",
    "alternatives": "#c5b0d5",
}


def plot_cluster_network(
    G: nx.Graph,
    assignments: dict[str, int],
    categories: dict[str, str],
    centrality: dict[str, float] | None = None,
    title: str = "Asset Cluster Network",
    save_path: Path | None = None,
) -> go.Figure:
    """Plot asset network colored by cluster assignment.

    Parameters
    ----------
    G : nx.Graph
        Asset correlation graph.
    assignments : dict[str, int]
        Mapping of ticker to cluster id.
    categories : dict[str, str]
        Mapping of ticker to asset category.
    centrality : dict[str, float] | None
        Optional centrality values for node sizing.
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    pos = nx.spring_layout(G, seed=42)
    cluster_palette = plotly.colors.qualitative.Set2

    # Determine top-k edges by weight to avoid clutter
    edges_with_weight = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        edges_with_weight.append((u, v, w))
    edges_with_weight.sort(key=lambda x: abs(x[2]), reverse=True)
    top_k = min(len(edges_with_weight), max(100, len(G.nodes) * 3))
    kept_edges = edges_with_weight[:top_k]

    # Build edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v, _w in kept_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#cccccc"),
        hoverinfo="skip",
        showlegend=False,
    )

    # Build node traces grouped by cluster for legend
    unique_clusters = sorted(set(assignments.values()))
    node_traces: list[go.Scatter] = []

    for cid in unique_clusters:
        color = cluster_palette[cid % len(cluster_palette)]
        tickers = [t for t, c in assignments.items() if c == cid and t in pos]
        xs = [pos[t][0] for t in tickers]
        ys = [pos[t][1] for t in tickers]

        if centrality is not None:
            cent_vals = np.array([centrality.get(t, 0.0) for t in tickers])
            if cent_vals.max() > 0:
                sizes = 8 + 30 * (cent_vals / cent_vals.max())
            else:
                sizes = np.full(len(tickers), 10.0)
        else:
            sizes = np.full(len(tickers), 10.0)

        hover = [
            (
                f"<b>{t}</b><br>"
                f"Cluster: {cid}<br>"
                f"Category: {categories.get(t, 'unknown')}<br>"
                f"Centrality: {centrality.get(t, 0.0):.4f}"
                if centrality
                else f"<b>{t}</b><br>Cluster: {cid}<br>"
                f"Category: {categories.get(t, 'unknown')}"
            )
            for t in tickers
        ]

        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=sizes, color=color, line=dict(width=0.5, color="white")),
                text=tickers,
                textposition="top center",
                textfont=dict(size=7),
                hovertext=hover,
                hoverinfo="text",
                name=f"Cluster {cid}",
                legendgroup=f"cluster_{cid}",
            )
        )

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved network plot to %s", save_path)

    return fig
