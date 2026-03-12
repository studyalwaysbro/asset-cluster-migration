"""Migration and Sankey chart visualization (Phase 2)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
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


def plot_cmi_comparison(
    comparison_df: pd.DataFrame,
    event_windows: list[dict] | None = None,
    title: str = "CMI: K-Means vs Leiden",
    save_path: Path | None = None,
) -> go.Figure:
    """Compare CMI time series for K-Means and Leiden clustering.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Date-indexed with columns ``kmeans_cmi`` and ``leiden_cmi``.
    event_windows : list[dict] | None
        Dicts with keys ``event_start``, ``event_end``, ``name``.
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comparison_df.index,
            y=comparison_df["kmeans_cmi"],
            mode="lines",
            name="K-Means CMI",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="Date: %{x}<br>K-Means CMI: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison_df.index,
            y=comparison_df["leiden_cmi"],
            mode="lines",
            name="Leiden CMI",
            line=dict(color="#d62728", width=1.5),
            hovertemplate="Date: %{x}<br>Leiden CMI: %{y:.4f}<extra></extra>",
        )
    )

    if event_windows:
        for ew in event_windows:
            fig.add_vrect(
                x0=ew["event_start"],
                x1=ew["event_end"],
                fillcolor="rgba(150,150,150,0.15)",
                line_width=0,
                annotation_text=ew.get("name", ""),
                annotation_position="top left",
                annotation_font_size=9,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="CMI",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved CMI comparison plot to %s", save_path)

    return fig


def plot_migration_sankey(
    flow_matrix: np.ndarray | pd.DataFrame,
    labels: list[str],
    title: str = "Cluster Migration Flow",
    save_path: Path | None = None,
) -> go.Figure:
    """Sankey diagram of cluster migration flows.

    Parameters
    ----------
    flow_matrix : np.ndarray | pd.DataFrame
        Square matrix where entry (i, j) is the flow from source cluster i
        to target cluster j.
    labels : list[str]
        Cluster labels (length matches matrix dimension).
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    if isinstance(flow_matrix, pd.DataFrame):
        flow_matrix = flow_matrix.values

    n = len(labels)
    palette = plotly.colors.qualitative.Set2

    # Build source labels (left) and target labels (right)
    source_labels = [f"{lbl} (t)" for lbl in labels]
    target_labels = [f"{lbl} (t+1)" for lbl in labels]
    all_labels = source_labels + target_labels

    # Node colors
    node_colors = [palette[i % len(palette)] for i in range(n)] * 2

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []

    for i in range(n):
        for j in range(n):
            val = float(flow_matrix[i, j])
            if val > 0:
                sources.append(i)
                targets.append(n + j)
                values.append(val)
                # Semi-transparent source color
                base = palette[i % len(palette)]
                link_colors.append(base.replace("rgb", "rgba").replace(")", ",0.4)") if "rgb" in base else base)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved migration Sankey to %s", save_path)

    return fig
