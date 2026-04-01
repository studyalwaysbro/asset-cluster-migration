"""Chart generators for the ACM daily report.

Produces self-contained HTML snippets (Plotly) that embed directly into the report:
  1. CMI/TDS timeseries with regime shading
  2. Interactive cluster network graph
  3. Cross-asset correlation heatmap (latest window)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PROJECT_ROOT, get_universe_config

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Dark theme matching report CSS
DARK_THEME = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="system-ui, -apple-system, sans-serif"),
    colorway=["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#f0883e", "#39d353", "#db61a2"],
)

GRID_STYLE = dict(gridcolor="#30363d", gridwidth=0.5, zerolinecolor="#30363d")


def build_migration_timeseries_chart() -> str:
    """CMI + TDS timeseries with regime shading. Returns HTML div string."""
    migration = pd.read_parquet(PROCESSED_DIR / "migration_timeseries.parquet")
    migration["date"] = pd.to_datetime(migration["date"])

    # Compute TDS z-score
    tds_mean = migration["tds"].expanding().mean()
    tds_std = migration["tds"].expanding().std().fillna(1).replace(0, 1)
    migration["tds_zscore"] = (migration["tds"] - tds_mean) / tds_std

    # Load regimes
    regime_path = PROCESSED_DIR / "regime_labels.csv"
    regimes = None
    if regime_path.exists():
        regimes = pd.read_csv(regime_path)
        regimes["date"] = pd.to_datetime(regimes["date"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=("Cluster Migration Index (CMI)", "Topology Deformation Score (TDS Z-Score)", "Active Clusters"),
    )

    # Regime shading
    if regimes is not None:
        regime_colors = {"stress": "rgba(248,81,73,0.12)", "transition": "rgba(210,153,34,0.08)"}
        current_regime = None
        start_date = None

        for _, row in regimes.iterrows():
            regime = row["regime"]
            if regime in regime_colors:
                if regime != current_regime:
                    if current_regime in regime_colors and start_date is not None:
                        for r in range(1, 4):
                            fig.add_vrect(x0=start_date, x1=row["date"], fillcolor=regime_colors[current_regime],
                                          line_width=0, layer="below", row=r, col=1)
                    start_date = row["date"]
                    current_regime = regime
            else:
                if current_regime in regime_colors and start_date is not None:
                    for r in range(1, 4):
                        fig.add_vrect(x0=start_date, x1=row["date"], fillcolor=regime_colors[current_regime],
                                      line_width=0, layer="below", row=r, col=1)
                current_regime = regime
                start_date = None

    # CMI trace
    fig.add_trace(go.Scatter(
        x=migration["date"], y=migration["cmi"],
        mode="lines", name="CMI",
        line=dict(color="#58a6ff", width=1.2),
        hovertemplate="Date: %{x}<br>CMI: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # CMI 30-window moving average
    fig.add_trace(go.Scatter(
        x=migration["date"], y=migration["cmi"].rolling(30).mean(),
        mode="lines", name="CMI MA(30)",
        line=dict(color="#3fb950", width=1.5, dash="dash"),
    ), row=1, col=1)

    # TDS z-score
    fig.add_trace(go.Scatter(
        x=migration["date"], y=migration["tds_zscore"],
        mode="lines", name="TDS Z-Score",
        line=dict(color="#bc8cff", width=1.2),
        hovertemplate="Date: %{x}<br>TDS Z: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # TDS alert thresholds
    fig.add_hline(y=2.0, line_dash="dot", line_color="#f85149", annotation_text="Alert (2.0)", row=2, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="#f85149", row=2, col=1)

    # Cluster count
    fig.add_trace(go.Scatter(
        x=migration["date"], y=migration["n_clusters"],
        mode="lines", name="Clusters",
        line=dict(color="#d29922", width=1.5),
        fill="tozeroy", fillcolor="rgba(210,153,34,0.1)",
        hovertemplate="Date: %{x}<br>Clusters: %{y}<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        height=700, margin=dict(l=60, r=30, t=40, b=30),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **DARK_THEME,
    )
    for i in range(1, 4):
        fig.update_xaxes(**GRID_STYLE, row=i, col=1)
        fig.update_yaxes(**GRID_STYLE, row=i, col=1)

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="migration-timeseries")


def build_cluster_network_chart() -> str:
    """Interactive cluster network graph for the latest window. Returns HTML div string."""
    assignments = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
    latest_date = assignments["date"].max()
    latest = assignments[assignments["date"] == latest_date]

    centrality_path = PROCESSED_DIR / "centrality_metrics.parquet"
    centrality = None
    if centrality_path.exists():
        cent_df = pd.read_parquet(centrality_path)
        centrality = cent_df[cent_df["date"] == latest_date].set_index("ticker")

    # Load cleaned prices for correlation
    prices = pd.read_parquet(PROCESSED_DIR / "cleaned_prices.parquet")
    tickers = latest["ticker"].tolist()
    available = [t for t in tickers if t in prices.columns]
    recent_prices = prices[available].tail(120)
    corr_matrix = recent_prices.pct_change().dropna().corr()

    # Build graph layout using correlation as edge weights
    # Use spring layout approximation
    cluster_map = dict(zip(latest["ticker"], latest["cluster"]))
    clusters = sorted(set(cluster_map.values()))

    # Category labels from universe config
    ticker_category = {}
    try:
        config = get_universe_config()
        for group_name, assets in config.get("assets", {}).items():
            for asset in assets:
                ticker_category[asset["ticker"]] = group_name
    except Exception:
        pass

    # Assign colors per cluster
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#f0883e", "#39d353", "#db61a2",
              "#79c0ff", "#56d364", "#e3b341", "#ff7b72", "#d2a8ff", "#ffa657", "#7ee787", "#ff9bce"]

    # Spring layout — position nodes by cluster
    import networkx as nx
    G = nx.Graph()
    for t in available:
        G.add_node(t)
    for i, t1 in enumerate(available):
        for t2 in available[i + 1:]:
            if t1 in corr_matrix.index and t2 in corr_matrix.index:
                w = corr_matrix.loc[t1, t2]
                if abs(w) > 0.3:
                    G.add_edge(t1, t2, weight=max(0.01, abs(w)))

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42, weight="weight")

    # Build traces
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.3, color="#30363d"),
        hoverinfo="none",
    )

    # Node traces per cluster
    node_traces = []
    for cluster_id in clusters:
        cluster_tickers = [t for t in available if cluster_map.get(t) == cluster_id]
        if not cluster_tickers:
            continue

        node_x = [pos[t][0] for t in cluster_tickers if t in pos]
        node_y = [pos[t][1] for t in cluster_tickers if t in pos]

        # Size by eigenvector centrality
        sizes = []
        hover_texts = []
        for t in cluster_tickers:
            if t not in pos:
                continue
            size = 12
            eig = 0
            if centrality is not None and t in centrality.index:
                eig = centrality.loc[t, "eigenvector"]
                size = 8 + eig * 50
            cat = ticker_category.get(t, "other")
            hover_texts.append(f"<b>{t}</b><br>Cluster {cluster_id}<br>Category: {cat}<br>Eigenvector: {eig:.3f}")
            sizes.append(size)

        color = colors[cluster_id % len(colors)]
        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=sizes, color=color, line=dict(width=1, color="#0d1117")),
            text=[t for t in cluster_tickers if t in pos],
            textposition="top center",
            textfont=dict(size=8, color=color),
            name=f"Cluster {cluster_id} ({len(cluster_tickers)})",
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        height=650, margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text=f"Cluster Network — {latest_date} ({len(available)} assets)", font=dict(size=14)),
        **DARK_THEME,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="cluster-network")


def build_correlation_heatmap() -> str:
    """Cross-asset correlation heatmap for the latest 120-day window. Returns HTML div string."""
    prices = pd.read_parquet(PROCESSED_DIR / "cleaned_prices.parquet")
    returns = prices.pct_change().dropna().tail(120)
    corr = returns.corr()

    # Sort by hierarchical clustering for visual grouping
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, 2)
    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    ordered_tickers = [corr.columns[i] for i in order]
    corr_ordered = corr.loc[ordered_tickers, ordered_tickers]

    fig = go.Figure(data=go.Heatmap(
        z=corr_ordered.values,
        x=ordered_tickers,
        y=ordered_tickers,
        colorscale=[
            [0.0, "#f85149"],
            [0.25, "#f0883e"],
            [0.5, "#0d1117"],
            [0.75, "#3fb950"],
            [1.0, "#58a6ff"],
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Corr", side="right"), tickfont=dict(color="#8b949e")),
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=700, width=900,
        margin=dict(l=80, r=30, t=50, b=80),
        title=dict(text=f"Cross-Asset Correlation (120-day window, {len(ordered_tickers)} assets)", font=dict(size=14)),
        xaxis=dict(tickfont=dict(size=7), tickangle=45),
        yaxis=dict(tickfont=dict(size=7), autorange="reversed"),
        **DARK_THEME,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="correlation-heatmap")


def generate_all_chart_html() -> dict[str, str]:
    """Generate all charts and return dict of {name: html_div}."""
    charts = {}

    try:
        logger.info("Generating CMI/TDS timeseries chart...")
        charts["migration_timeseries"] = build_migration_timeseries_chart()
    except Exception as e:
        logger.error(f"Migration timeseries chart failed: {e}")
        charts["migration_timeseries"] = f'<div class="alert-box">Chart generation failed: {e}</div>'

    try:
        logger.info("Generating cluster network chart...")
        charts["cluster_network"] = build_cluster_network_chart()
    except Exception as e:
        logger.error(f"Cluster network chart failed: {e}")
        charts["cluster_network"] = f'<div class="alert-box">Chart generation failed: {e}</div>'

    try:
        logger.info("Generating correlation heatmap...")
        charts["correlation_heatmap"] = build_correlation_heatmap()
    except Exception as e:
        logger.error(f"Correlation heatmap chart failed: {e}")
        charts["correlation_heatmap"] = f'<div class="alert-box">Chart generation failed: {e}</div>'

    return charts
