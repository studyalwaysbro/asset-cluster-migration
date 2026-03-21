"""Correlation and similarity heatmaps."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

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


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: list[str],
    categories: dict[str, str] | None = None,
    title: str = "Correlation Matrix",
    save_path: Path | None = None,
) -> go.Figure:
    """Plot correlation matrix as an interactive heatmap.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Square correlation matrix.
    labels : list[str]
        Ticker labels matching matrix dimensions.
    categories : dict[str, str] | None
        Mapping of ticker to asset category for grouping.
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    n = len(labels)
    labels = list(labels)

    # Hierarchical clustering ordering for block structure
    try:
        dist = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, None)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    except Exception:
        logger.warning("Hierarchical ordering failed, falling back to original order")
        order = np.arange(n)

    # If categories provided, do a secondary sort by category within HC order
    if categories is not None:
        # Sort by category first, then by HC order within each category
        cat_order = sorted(set(categories.get(l, "zzz") for l in labels))
        cat_rank = {c: i for i, c in enumerate(cat_order)}
        decorated = [(cat_rank.get(categories.get(labels[i], "zzz"), 999), int(pos), i) for pos, i in enumerate(order)]
        decorated.sort(key=lambda x: (x[0], x[1]))
        order = np.array([d[2] for d in decorated])

    ordered_labels = [labels[i] for i in order]
    ordered_corr = corr_matrix[np.ix_(order, order)]

    # Build hover text matrix
    hover_text = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            hover_text[i, j] = (
                f"{ordered_labels[i]} vs {ordered_labels[j]}<br>"
                f"Correlation: {ordered_corr[i, j]:.3f}"
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=ordered_corr,
            x=ordered_labels,
            y=ordered_labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(title="Corr", thickness=15),
        )
    )

    # Add category color bar on the side if categories provided
    if categories is not None:
        cat_colors = [
            CATEGORY_COLORS.get(categories.get(lbl, ""), "#333333")
            for lbl in ordered_labels
        ]
        # Add a narrow heatmap column as category indicator
        cat_indices = [list(CATEGORY_COLORS.values()).index(c) if c in CATEGORY_COLORS.values() else 0 for c in cat_colors]
        fig.add_trace(
            go.Heatmap(
                z=[[ci] for ci in cat_indices],
                y=ordered_labels,
                x=["Category"],
                colorscale=[[i / max(len(CATEGORY_COLORS) - 1, 1), c] for i, c in enumerate(CATEGORY_COLORS.values())],
                showscale=False,
                hovertext=[[categories.get(lbl, "unknown")] for lbl in ordered_labels],
                hoverinfo="text",
                xaxis="x2",
            )
        )
        fig.update_layout(
            xaxis2=dict(
                domain=[0.96, 1.0],
                showticklabels=False,
                anchor="y",
            ),
            xaxis=dict(domain=[0.0, 0.95]),
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=900,
        height=850,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=40, t=50, b=80),
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved correlation heatmap to %s", save_path)

    return fig
