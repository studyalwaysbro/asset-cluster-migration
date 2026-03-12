"""Rolling metric time series plots."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Primary metrics shown in top subplot
_PRIMARY_METRICS = {"CMI", "cmi", "TDS", "tds"}

# Regime background colors
_REGIME_COLORS = {
    "calm": "rgba(46,204,113,0.12)",
    "transition": "rgba(243,156,18,0.12)",
    "stress": "rgba(231,76,60,0.12)",
}

_TRACE_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
]


def plot_metric_timeseries(
    metrics_df: pd.DataFrame,
    event_windows: list[dict] | None = None,
    regime_labels: pd.Series | None = None,
    title: str = "Topology Metrics Over Time",
    save_path: Path | None = None,
) -> go.Figure:
    """Plot topology metrics over time with optional regime shading.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Date-indexed DataFrame whose columns are metric names.
    event_windows : list[dict] | None
        Dicts with ``event_start``, ``event_end``, ``name`` for vertical markers.
    regime_labels : pd.Series | None
        Date-indexed series with values ``calm``, ``transition``, ``stress``.
    title : str
        Figure title.
    save_path : Path | None
        If provided, save interactive HTML.

    Returns
    -------
    go.Figure
    """
    cols = list(metrics_df.columns)
    primary = [c for c in cols if c in _PRIMARY_METRICS or c.upper() in _PRIMARY_METRICS]
    secondary = [c for c in cols if c not in primary]

    if not primary:
        # If no recognized primary metrics, put first half in top
        mid = max(1, len(cols) // 2)
        primary = cols[:mid]
        secondary = cols[mid:]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=["Primary Metrics (CMI / TDS)", "Secondary Metrics"],
    )

    # Top subplot: primary metrics
    for i, col in enumerate(primary):
        color = _TRACE_PALETTE[i % len(_TRACE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.5),
                hovertemplate=f"{col}: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Bottom subplot: secondary metrics
    for i, col in enumerate(secondary):
        color = _TRACE_PALETTE[(i + len(primary)) % len(_TRACE_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=metrics_df.index,
                y=metrics_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.2),
                hovertemplate=f"{col}: %{{y:.4f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Regime background shading
    if regime_labels is not None:
        _add_regime_shading(fig, regime_labels, rows=[1, 2])

    # Event windows as vertical dashed lines with annotations
    if event_windows:
        for ew in event_windows:
            for row in [1, 2]:
                fig.add_vline(
                    x=pd.Timestamp(ew["event_start"]).timestamp() * 1000
                    if not isinstance(ew["event_start"], str)
                    else ew["event_start"],
                    line=dict(color="black", width=1, dash="dash"),
                    row=row,
                    col=1,
                )
            fig.add_annotation(
                x=ew["event_start"],
                y=1.0,
                yref="paper",
                text=ew.get("name", ""),
                showarrow=False,
                font=dict(size=9),
                textangle=-45,
            )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    if save_path is not None:
        fig.write_html(str(save_path))
        logger.info("Saved metric timeseries to %s", save_path)

    return fig


def _add_regime_shading(
    fig: go.Figure,
    regime_labels: pd.Series,
    rows: list[int],
) -> None:
    """Add colored background rectangles for regime spans."""
    if regime_labels.empty:
        return

    dates = regime_labels.index.tolist()
    regimes = regime_labels.values.tolist()

    spans: list[tuple[str, object, object]] = []
    current_regime = regimes[0]
    span_start = dates[0]

    for i in range(1, len(dates)):
        if regimes[i] != current_regime:
            spans.append((current_regime, span_start, dates[i - 1]))
            current_regime = regimes[i]
            span_start = dates[i]
    spans.append((current_regime, span_start, dates[-1]))

    for regime, start, end in spans:
        color = _REGIME_COLORS.get(str(regime), "rgba(200,200,200,0.08)")
        for row in rows:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                line_width=0,
                row=row,
                col=1,
            )
