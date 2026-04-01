"""Presentation-grade charts for ACM slides.

These are NOT the same as the daily report charts. These are designed for
a projector/presentation with:
  - Event annotations with callouts
  - Proper axis labels with dates
  - Larger fonts
  - Key findings highlighted
  - Speaker-ready context
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PROJECT_ROOT, get_event_windows_config, get_universe_config

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Presentation dark theme — bigger fonts, cleaner gridlines
PRES_THEME = dict(
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    font=dict(color="#e6edf3", family="system-ui, sans-serif", size=14),
)
GRID = dict(gridcolor="#2a2a2a", gridwidth=0.5, zerolinecolor="#2a2a2a")

# Event colors
EVENT_COLORS = {
    "COVID": "#f85149",
    "Fed": "#d29922",
    "SVB": "#f0883e",
    "Japan": "#bc8cff",
    "Iran": "#ff7b72",
    "DeepSeek": "#58a6ff",
    "EU Debt": "#8b949e",
}


def _load_events() -> list[dict]:
    """Load event windows and return simplified list."""
    try:
        cfg = get_event_windows_config()
    except Exception:
        return []

    events = []
    for key, ev in cfg.get("events", {}).items():
        short_name = key.replace("_", " ").title()
        # Shorter display names
        if "covid" in key:
            short_name = "COVID"
        elif "fed" in key:
            short_name = "Fed Tightening"
        elif "svb" in key:
            short_name = "SVB Crisis"
        elif "japan" in key:
            short_name = "Japan Carry"
        elif "iran" in key and "oct" in key:
            short_name = "Iran II"
        elif "iran" in key:
            short_name = "Iran I"
        elif "deepseek" in key:
            short_name = "DeepSeek"
        elif "eu_debt" in key:
            short_name = "EU Debt"

        events.append({
            "name": short_name,
            "start": pd.Timestamp(ev["event_start"]),
            "end": pd.Timestamp(ev["event_end"]),
            "peak": pd.Timestamp(ev.get("peak_stress_date", ev["event_start"])),
            "color": EVENT_COLORS.get(short_name.split()[0], "#8b949e"),
        })
    return events


def build_pres_migration_chart() -> str:
    """Presentation-grade CMI/TDS timeseries with event annotations."""
    mig = pd.read_parquet(PROCESSED_DIR / "migration_timeseries.parquet")
    mig["date"] = pd.to_datetime(mig["date"])

    # TDS z-score
    tds_mean = mig["tds"].expanding().mean()
    tds_std = mig["tds"].expanding().std().fillna(1).replace(0, 1)
    mig["tds_zscore"] = (mig["tds"] - tds_mean) / tds_std

    # Regimes for shading
    regime_path = PROCESSED_DIR / "regime_labels.csv"
    regimes = None
    if regime_path.exists():
        regimes = pd.read_csv(regime_path)
        regimes["date"] = pd.to_datetime(regimes["date"])

    events = _load_events()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("", ""),
    )

    # ── Regime shading (both panels) ──
    if regimes is not None:
        regime_colors = {"stress": "rgba(248,81,73,0.08)", "transition": "rgba(210,153,34,0.05)"}
        current_regime = None
        start_date = None
        for _, row in regimes.iterrows():
            regime = row["regime"]
            if regime in regime_colors:
                if regime != current_regime:
                    if current_regime in regime_colors and start_date is not None:
                        for r in [1, 2]:
                            fig.add_vrect(x0=start_date, x1=row["date"],
                                          fillcolor=regime_colors[current_regime],
                                          line_width=0, layer="below", row=r, col=1)
                    start_date = row["date"]
                    current_regime = regime
            else:
                if current_regime in regime_colors and start_date is not None:
                    for r in [1, 2]:
                        fig.add_vrect(x0=start_date, x1=row["date"],
                                      fillcolor=regime_colors[current_regime],
                                      line_width=0, layer="below", row=r, col=1)
                current_regime = regime
                start_date = None

    # ── Event markers ──
    def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for ev in events:
        for r in [1, 2]:
            fig.add_vrect(x0=ev["start"], x1=ev["end"],
                          fillcolor=_hex_to_rgba(ev["color"], 0.12),
                          line=dict(color=ev["color"], width=1, dash="dot"),
                          layer="below", row=r, col=1)

        # Annotation on top panel at peak
        peak_idx = mig[mig["date"] >= ev["peak"]].head(1)
        if len(peak_idx) > 0:
            y_val = peak_idx.iloc[0]["cmi"]
            fig.add_annotation(
                x=ev["peak"], y=y_val + 0.03,
                text=f"<b>{ev['name']}</b>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=ev["color"],
                font=dict(size=11, color=ev["color"]),
                ax=0, ay=-35, row=1, col=1,
            )

    # ── CMI trace ──
    fig.add_trace(go.Scatter(
        x=mig["date"], y=mig["cmi"], mode="lines", name="CMI",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>CMI: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # CMI 30-window MA
    fig.add_trace(go.Scatter(
        x=mig["date"], y=mig["cmi"].rolling(30).mean(),
        mode="lines", name="CMI MA(30)",
        line=dict(color="#3fb950", width=2, dash="dash"),
    ), row=1, col=1)

    # ── TDS z-score ──
    # Color by magnitude
    fig.add_trace(go.Scatter(
        x=mig["date"], y=mig["tds_zscore"], mode="lines", name="TDS Z-Score",
        line=dict(color="#bc8cff", width=1.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>TDS Z: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # Alert thresholds
    fig.add_hline(y=2.0, line_dash="dot", line_color="#f85149", line_width=1,
                  annotation_text="Alert Threshold (z=2)", annotation_font_size=10,
                  annotation_font_color="#f85149", row=2, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="#f85149", line_width=1, row=2, col=1)

    # Layout
    fig.update_layout(
        height=500, margin=dict(l=60, r=30, t=30, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=11),
        **PRES_THEME,
    )
    fig.update_yaxes(title_text="Cluster Migration Index", title_font_size=12, row=1, col=1, **GRID)
    fig.update_yaxes(title_text="TDS Z-Score", title_font_size=12, row=2, col=1, **GRID)
    fig.update_xaxes(
        dtick="M12", tickformat="%Y", tickfont_size=11,
        **GRID, row=1, col=1,
    )
    fig.update_xaxes(
        dtick="M12", tickformat="%Y", tickfont_size=11,
        title_text="Date", title_font_size=12,
        **GRID, row=2, col=1,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pres-migration")


def build_pres_network_chart() -> str:
    """Presentation-grade cluster network with larger nodes and labels."""
    import networkx as nx

    assignments = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
    latest_date = assignments["date"].max()
    latest = assignments[assignments["date"] == latest_date]

    centrality = None
    cent_path = PROCESSED_DIR / "centrality_metrics.parquet"
    if cent_path.exists():
        cent_df = pd.read_parquet(cent_path)
        centrality = cent_df[cent_df["date"] == latest_date].set_index("ticker")

    prices = pd.read_parquet(PROCESSED_DIR / "cleaned_prices.parquet")
    tickers = latest["ticker"].tolist()
    available = [t for t in tickers if t in prices.columns]
    corr = prices[available].tail(120).pct_change().dropna().corr()

    cluster_map = dict(zip(latest["ticker"], latest["cluster"]))

    # Category labels
    ticker_cat = {}
    try:
        config = get_universe_config()
        for gname, assets in config.get("assets", {}).items():
            for a in assets:
                ticker_cat[a["ticker"]] = gname.replace("_", " ").title()
    except Exception:
        pass

    # Cluster interpretation labels
    cluster_labels = {}
    for cid in sorted(set(cluster_map.values())):
        members = [t for t, c in cluster_map.items() if c == cid]
        cats = {}
        for t in members:
            c = ticker_cat.get(t, "Other")
            cats.setdefault(c, []).append(t)
        dominant = max(cats.items(), key=lambda x: len(x[1]))
        cluster_labels[cid] = f"C{cid}: {dominant[0]} ({len(members)})"

    # Build graph
    G = nx.Graph()
    for t in available:
        G.add_node(t)
    for i, t1 in enumerate(available):
        for t2 in available[i + 1:]:
            if t1 in corr.index and t2 in corr.index:
                w = corr.loc[t1, t2]
                if abs(w) > 0.35:
                    G.add_edge(t1, t2, weight=max(0.01, abs(w)))

    pos = nx.spring_layout(G, k=2.8, iterations=100, seed=42, weight="weight")

    # Edges
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.3, color="#333333"), hoverinfo="none",
    )

    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#f0883e", "#39d353", "#db61a2",
              "#79c0ff", "#56d364", "#e3b341", "#ff7b72"]

    node_traces = []
    for cid in sorted(set(cluster_map.values())):
        ctickers = [t for t in available if cluster_map.get(t) == cid and t in pos]
        if not ctickers:
            continue

        node_x = [pos[t][0] for t in ctickers]
        node_y = [pos[t][1] for t in ctickers]
        sizes = []
        hovers = []
        for t in ctickers:
            eig = centrality.loc[t, "eigenvector"] if centrality is not None and t in centrality.index else 0
            sizes.append(max(12, 10 + eig * 60))
            cat = ticker_cat.get(t, "")
            hovers.append(f"<b>{t}</b><br>{cat}<br>Cluster {cid}<br>Eigenvector: {eig:.3f}")

        color = colors[cid % len(colors)]
        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=sizes, color=color, line=dict(width=1.5, color="#111111")),
            text=ctickers, textposition="top center",
            textfont=dict(size=9, color=color),
            name=cluster_labels.get(cid, f"Cluster {cid}"),
            hovertext=hovers, hoverinfo="text",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font_size=10, itemsizing="constant"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        **PRES_THEME,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pres-network")


def build_pres_heatmap() -> str:
    """Presentation-grade correlation heatmap."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    prices = pd.read_parquet(PROCESSED_DIR / "cleaned_prices.parquet")
    returns = prices.pct_change().dropna().tail(120)
    corr = returns.corr()

    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, 2)
    Z = linkage(squareform(dist), method="ward")
    order = leaves_list(Z)

    ordered = [corr.columns[i] for i in order]
    corr_ord = corr.loc[ordered, ordered]

    fig = go.Figure(data=go.Heatmap(
        z=corr_ord.values, x=ordered, y=ordered,
        colorscale=[[0, "#f85149"], [0.25, "#f0883e"], [0.5, "#111111"], [0.75, "#3fb950"], [1, "#58a6ff"]],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Corr", side="right"), tickfont=dict(color="#8b949e", size=10), len=0.8),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Corr: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=540, width=700, margin=dict(l=60, r=40, t=20, b=60),
        xaxis=dict(tickfont=dict(size=7), tickangle=45),
        yaxis=dict(tickfont=dict(size=7), autorange="reversed"),
        **PRES_THEME,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="pres-heatmap")
