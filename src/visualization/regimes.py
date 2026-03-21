"""Regime overlay visualizations."""
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

_REGIME_COLORS = {
    "calm": "#2ecc71",
    "transition": "#f39c12",
    "stress": "#e74c3c",
}

_TRACE_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
]


def plot_regime_timeline(
    regime_labels: pd.Series,
    metrics_overlay: pd.DataFrame | None = None,
    title: str = "Market Regime Timeline",
    save_path: Path | None = None,
) -> go.Figure:
    """Plot market regime timeline with optional metric overlays.

    Parameters
    ----------
    regime_labels : pd.Series
        Date-indexed series with values ``calm``, ``transition``, ``stress``.
    metrics_overlay : pd.DataFrame | None
        Date-indexed DataFrame of metrics to plot as line traces on top.
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Build regime spans
    spans = _extract_regime_spans(regime_labels)

    # Add background rectangles for each regime span
    for regime, start, end in spans:
        color = _REGIME_COLORS.get(str(regime), "#cccccc")
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.2,
            line_width=0,
        )

    # Add invisible traces for regime legend entries
    for regime, color in _REGIME_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=f"Regime: {regime}",
                showlegend=True,
            )
        )

    # Overlay metric line traces
    if metrics_overlay is not None:
        for i, col in enumerate(metrics_overlay.columns):
            color = _TRACE_PALETTE[i % len(_TRACE_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=metrics_overlay.index,
                    y=metrics_overlay[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{col}: %{{y:.4f}}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value" if metrics_overlay is not None else "",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    # If no overlay metrics, hide y-axis (regime-only view)
    if metrics_overlay is None:
        fig.update_yaxes(visible=False)

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved regime timeline to %s", save_path)

    return fig


def _extract_regime_spans(
    regime_labels: pd.Series,
) -> list[tuple[str, object, object]]:
    """Extract contiguous regime spans from a label series."""
    if regime_labels.empty:
        return []

    dates = regime_labels.index.tolist()
    regimes = regime_labels.values.tolist()

    spans: list[tuple[str, object, object]] = []
    current_regime = str(regimes[0])
    span_start = dates[0]

    for i in range(1, len(dates)):
        r = str(regimes[i])
        if r != current_regime:
            spans.append((current_regime, span_start, dates[i - 1]))
            current_regime = r
            span_start = dates[i]
    spans.append((current_regime, span_start, dates[-1]))

    return spans
