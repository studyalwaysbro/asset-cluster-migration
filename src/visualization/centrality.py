"""Centrality evolution plots."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
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

_FALLBACK_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
    "#aec7e8", "#ff9896", "#ffbb78", "#98df8a", "#c5b0d5",
]


def plot_centrality_evolution(
    centrality_history: pd.DataFrame,
    categories: dict[str, str] | None = None,
    top_n: int = 15,
    metric_name: str = "Betweenness Centrality",
    title: str = "Centrality Evolution",
    save_path: Path | None = None,
) -> go.Figure:
    """Plot centrality evolution with top-N assets highlighted.

    Parameters
    ----------
    centrality_history : pd.DataFrame
        Date-indexed DataFrame, columns = tickers, values = centrality.
    categories : dict[str, str] | None
        Mapping of ticker to asset category for coloring.
    top_n : int
        Number of assets to highlight.
    metric_name : str
        Name of the centrality metric (for axis label).
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Identify top-N by mean centrality
    mean_cent = centrality_history.mean().sort_values(ascending=False)
    top_tickers = list(mean_cent.head(top_n).index)
    all_tickers = list(centrality_history.columns)

    # Background: all assets as thin gray lines
    for ticker in all_tickers:
        fig.add_trace(
            go.Scatter(
                x=centrality_history.index,
                y=centrality_history[ticker],
                mode="lines",
                line=dict(color="rgba(180,180,180,0.15)", width=0.8),
                hovertemplate=f"<b>{ticker}</b><br>Date: %{{x}}<br>{metric_name}: %{{y:.4f}}<extra></extra>",
                showlegend=False,
                name=ticker,
            )
        )

    # Overlay: top-N with colored lines
    for i, ticker in enumerate(top_tickers):
        if categories is not None:
            cat = categories.get(ticker, "")
            color = CATEGORY_COLORS.get(cat, _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)])
        else:
            color = _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=centrality_history.index,
                y=centrality_history[ticker],
                mode="lines",
                name=ticker,
                line=dict(color=color, width=2.0),
                hovertemplate=(
                    f"<b>{ticker}</b><br>"
                    f"Date: %{{x}}<br>"
                    f"{metric_name}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=metric_name,
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=9),
        ),
        margin=dict(l=60, r=140, t=50, b=40),
    )

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved centrality evolution plot to %s", save_path)

    return fig
