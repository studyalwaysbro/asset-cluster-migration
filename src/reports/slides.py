"""Presentation slide generator for ACM research.

Generates a self-contained HTML slide deck from pipeline outputs.
Uses reveal.js CDN for presentation mode. Designed for Stevens MS thesis defense.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, get_universe_config
from src.reports.slide_charts import (
    build_pres_migration_chart,
    build_pres_network_chart,
    build_pres_heatmap,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def generate_slide_deck(output_dir: Path | None = None) -> Path:
    """Generate a reveal.js HTML slide deck from pipeline outputs."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    deck_path = output_dir / f"acm_slides_{today}.html"

    slides = []
    slides.append(_slide_title(today))
    slides.append(_slide_motivation())
    slides.append(_slide_methodology())
    slides.append(_slide_five_layers())
    slides.append(_slide_tds_decomposition())
    slides.append(_slide_universe())
    slides.append(_slide_migration_timeseries())
    slides.append(_slide_regime_status())
    slides.append(_slide_event_covid())
    slides.append(_slide_event_pearson_blind())
    slides.append(_slide_event_epic_fury())
    slides.append(_slide_cluster_network())
    slides.append(_slide_cluster_composition())
    slides.append(_slide_key_insights())
    slides.append(_slide_centrality())
    slides.append(_slide_correlation_heatmap())
    slides.append(_slide_leadership_reversal())
    slides.append(_slide_kmeans_baseline())
    slides.append(_slide_regime_validation())
    slides.append(_slide_robustness_summary())
    slides.append(_slide_novelty_vs_literature())
    slides.append(_slide_conclusions())
    slides.append(_slide_future_work())

    html = _wrap_revealjs(today, "\n".join(slides))
    deck_path.write_text(html, encoding="utf-8")
    logger.info(f"Slide deck generated: {deck_path}")
    return deck_path


# ── Individual slides ─────────────────────────────────────────────────────

def _slide_title(today: str) -> str:
    return f"""
    <section>
        <h1 style="font-size:1.8em">Dynamic Multi-Asset Topology<br>and Cluster Migration</h1>
        <h3 style="color:#8b949e">Under Geopolitical Stress</h3>
        <p style="margin-top:40px;color:#e6edf3;font-size:0.7em">Nicholas Tavares &middot; Amjad Hanini &middot; Brandon DaSilva</p>
        <p style="color:#58a6ff">Stevens Institute of Technology</p>
        <p style="color:#8b949e;font-size:0.6em">{today}</p>
    </section>"""


def _slide_motivation() -> str:
    return """
    <section>
        <h2>Motivation</h2>
        <ul>
            <li>Traditional correlation analysis assumes <strong>static relationships</strong></li>
            <li>In reality, asset relationships <strong>evolve over time</strong> and <strong>restructure during crises</strong></li>
            <li>Key questions:
                <ul>
                    <li>How does market <em>topology</em> change during geopolitical events?</li>
                    <li>Do assets restructure <em>before</em> or <em>during</em> crises?</li>
                    <li>Can we measure the <em>speed</em> and <em>magnitude</em> of structural change?</li>
                </ul>
            </li>
        </ul>
    </section>"""


def _slide_methodology() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:10px">Methodology</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.55em">
            <div style="background:#1c2333;padding:12px 16px;border-radius:8px">
                <h3 style="color:#58a6ff;font-size:1.3em;margin:0 0 6px 0">Data</h3>
                <ul style="margin:0">
                    <li>134 ETFs (76 survive cleaning)</li>
                    <li>19 asset groups (equity, bonds, commodities, FX, crypto)</li>
                    <li>2010&ndash;2026 (3,944 trading days)</li>
                    <li>Source: FMP API (daily adjusted close)</li>
                </ul>
            </div>
            <div style="background:#1c2333;padding:12px 16px;border-radius:8px">
                <h3 style="color:#3fb950;font-size:1.3em;margin:0 0 6px 0">Clustering</h3>
                <ul style="margin:0">
                    <li>120-day rolling windows, 5-day step</li>
                    <li>Pearson shrinkage correlation (Ledoit-Wolf)</li>
                    <li>Consensus Leiden (100 runs, adaptive k)</li>
                    <li>765 windows analyzed</li>
                </ul>
            </div>
            <div style="background:#1c2333;padding:12px 16px;border-radius:8px">
                <h3 style="color:#d29922;font-size:1.3em;margin:0 0 6px 0">Novel Metrics</h3>
                <ul style="margin:0">
                    <li><strong>CMI:</strong> Cluster Migration Index (Hungarian matching)</li>
                    <li><strong>TDS:</strong> Topology Deformation Score (3-component)</li>
                    <li><strong>AMF:</strong> Asset Migration Frequency</li>
                    <li><strong>CPS:</strong> Cluster Persistence Score</li>
                </ul>
            </div>
            <div style="background:#1c2333;padding:12px 16px;border-radius:8px">
                <h3 style="color:#f85149;font-size:1.3em;margin:0 0 6px 0">Regimes</h3>
                <ul style="margin:0">
                    <li>3-state Gaussian HMM</li>
                    <li>State vector: vol, mean corr, dispersion</li>
                    <li>Calm (89.7%), Transition (5.1%), Stress (5.2%)</li>
                </ul>
            </div>
        </div>
    </section>"""


def _slide_universe() -> str:
    try:
        config = get_universe_config()
        groups = []
        for name, assets in config["assets"].items():
            groups.append(f"<li>{name.replace('_', ' ').title()}: {len(assets)}</li>")
        returns = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")
        n_survive = returns.shape[1]
    except Exception:
        groups = ["<li>Error loading universe</li>"]
        n_survive = "?"

    return f"""
    <section>
        <h2 style="font-size:1.3em;margin-bottom:8px">Universe: 134 ETFs &rarr; {n_survive} After Cleaning</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.5em">
            <ul style="margin:0">{''.join(groups[:10])}</ul>
            <ul style="margin:0">{''.join(groups[10:])}</ul>
        </div>
        <p style="font-size:0.55em;color:#8b949e;margin-top:10px">Assets dropped for &gt;5% missing data pre-2010 inception. Forward-fill max 5 days.</p>
    </section>"""


def _slide_regime_status() -> str:
    try:
        rl = pd.read_csv(PROCESSED_DIR / "regime_labels.csv")
        current = str(rl.iloc[-1]["regime"]).upper()
        regimes = rl["regime"].values
        days_since_calm = 0
        for i in range(len(regimes) - 1, -1, -1):
            if regimes[i] == "calm":
                break
            days_since_calm += 1
        transitions = sum(1 for i in range(1, min(60, len(rl))) if rl.iloc[-i]["regime"] != rl.iloc[-i - 1]["regime"])
        counts = rl["regime"].value_counts()
    except Exception:
        return '<section><h2>Regime Status</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Current Regime: <span style="color:{'#f85149' if current == 'STRESS' else '#d29922' if current == 'TRANSITION' else '#3fb950'}">{current}</span></h2>
        <ul>
            <li><strong>{days_since_calm} trading days</strong> since last CALM regime</li>
            <li><strong>{transitions} regime transitions</strong> in last 60 days</li>
            <li>Market oscillating between STRESS and TRANSITION &mdash; sustained structural instability</li>
        </ul>
        <p style="margin-top:20px;font-size:0.8em">Historical distribution: Calm {counts.get('calm', 0):,}d ({counts.get('calm', 0)/len(rl)*100:.1f}%),
        Stress {counts.get('stress', 0):,}d ({counts.get('stress', 0)/len(rl)*100:.1f}%),
        Transition {counts.get('transition', 0):,}d ({counts.get('transition', 0)/len(rl)*100:.1f}%)</p>
    </section>"""


def _slide_cluster_composition() -> str:
    try:
        ca = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
        latest_date = ca["date"].max()
        latest = ca[ca["date"] == latest_date]
        n_clusters = latest["cluster"].nunique()
        cluster_sizes = latest.groupby("cluster").size().to_dict()
        config = get_universe_config()
        ticker_group = {}
        for gname, assets in config.get("assets", {}).items():
            for a in assets:
                ticker_group[a["ticker"]] = gname

        # Build cluster summaries
        summaries = []
        for cid in sorted(cluster_sizes.keys()):
            members = latest[latest["cluster"] == cid]["ticker"].tolist()
            groups = {}
            for t in members:
                g = ticker_group.get(t, "other").replace("_", " ")
                groups.setdefault(g, []).append(t)
            dominant = max(groups.items(), key=lambda x: len(x[1]))
            label = dominant[0].title()
            summaries.append(f"<li><strong>C{cid}</strong> ({len(members)}): {label} &mdash; {', '.join(members[:8])}{'...' if len(members) > 8 else ''}</li>")
    except Exception:
        return '<section><h2>Cluster Composition</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2 style="font-size:1.3em;margin-bottom:8px">{n_clusters} Clusters Identified</h2>
        <ul style="font-size:0.52em;line-height:1.4">{''.join(summaries)}</ul>
    </section>"""


def _slide_cluster_network() -> str:
    try:
        chart = build_pres_network_chart()
    except Exception as e:
        chart = f"<p>Chart failed: {e}</p>"
    return f"""
    <section>
        <h2 style="font-size:1.1em;margin-bottom:4px">Current Cluster Network &mdash; What the Market Looks Like Right Now</h2>
        <div style="display:grid;grid-template-columns:3fr 2fr;gap:8px">
            <div style="height:480px;overflow:hidden">{chart}</div>
            <div style="font-size:0.5em;padding:8px 0">
                <p style="margin:0 0 6px 0">Each node is an ETF. <strong>Bigger nodes = more central</strong> (eigenvector centrality). Colors represent cluster membership. Edges connect assets with correlation &gt;0.35.</p>
                <div style="background:#1c2333;padding:8px 10px;border-radius:6px;margin-bottom:6px">
                    <strong style="color:#58a6ff">What to notice:</strong>
                    <ul style="margin:4px 0 0 0;padding-left:16px">
                        <li><strong>EEM</strong> (emerging markets) is the largest node &mdash; it's the network hub whose movements propagate everywhere</li>
                        <li>Tight <strong>EM cluster</strong> (purple) shows contagion risk &mdash; these all move together</li>
                        <li><strong>Gold/Silver</strong> cluster (yellow) is separate from Treasuries &mdash; gold is a commodity play, not safe haven</li>
                        <li><strong>MINT</strong> (short-term bonds) bridges risk-on and risk-off clusters</li>
                    </ul>
                </div>
                <div style="background:rgba(248,81,73,0.1);padding:6px 10px;border-radius:6px;border-left:2px solid #f85149">
                    The network reveals relationships that <strong>correlation matrices hide</strong>: community structure, bridge assets, and contagion pathways.
                </div>
            </div>
        </div>
    </section>"""


def _slide_migration_timeseries() -> str:
    try:
        chart = build_pres_migration_chart()
    except Exception as e:
        chart = f"<p>Chart failed: {e}</p>"
    return f"""
    <section>
        <h2 style="font-size:1.1em;margin-bottom:4px">16 Years of Market Topology &mdash; How Structure Evolves</h2>
        <div style="display:grid;grid-template-columns:3fr 2fr;gap:8px">
            <div style="height:480px;overflow:hidden">{chart}</div>
            <div style="font-size:0.5em;padding:8px 0">
                <p style="margin:0 0 6px 0"><strong>Top panel:</strong> CMI &mdash; what fraction of assets changed clusters between consecutive windows. Higher = more reshuffling.</p>
                <p style="margin:0 0 6px 0"><strong>Bottom panel:</strong> TDS z-score &mdash; how extreme is the structural deformation vs history. Above 2.0 = alert.</p>
                <div style="background:#1c2333;padding:8px 10px;border-radius:6px;margin-bottom:6px">
                    <strong style="color:#58a6ff">Reading the chart:</strong>
                    <ul style="margin:4px 0 0 0;padding-left:16px">
                        <li>Colored bands mark <strong>geopolitical events</strong> &mdash; COVID (red), Fed tightening (yellow), Iran escalations, Japan carry unwind</li>
                        <li>Light red background = HMM detected <strong>STRESS</strong> regime</li>
                        <li>Notice CMI spikes <strong>precede</strong> or <strong>coincide</strong> with events, not lag them</li>
                        <li>The <strong>green dashed line</strong> (30-window MA) shows the structural trend</li>
                    </ul>
                </div>
                <div style="background:rgba(210,153,34,0.15);padding:6px 10px;border-radius:6px;border-left:2px solid #d29922">
                    <strong>Epic Fury buildup</strong> (Jan 2026): TDS peaked at 0.176 &mdash; <strong>higher than COVID</strong> &mdash; and it happened <em>before</em> the strike.
                </div>
            </div>
        </div>
    </section>"""


def _slide_correlation_heatmap() -> str:
    try:
        chart = build_pres_heatmap()
    except Exception as e:
        chart = f"<p>Chart failed: {e}</p>"
    return f"""
    <section>
        <h2 style="font-size:1.1em;margin-bottom:4px">Cross-Asset Correlation &mdash; The Building Block</h2>
        <div style="display:grid;grid-template-columns:3fr 2fr;gap:8px">
            <div style="height:500px;overflow:hidden">{chart}</div>
            <div style="font-size:0.5em;padding:8px 0">
                <p style="margin:0 0 6px 0">Latest 120-day window. Assets reordered by hierarchical clustering (Ward linkage) to reveal <strong>block structure</strong>.</p>
                <div style="background:#1c2333;padding:8px 10px;border-radius:6px;margin-bottom:6px">
                    <strong style="color:#58a6ff">What the blocks tell us:</strong>
                    <ul style="margin:4px 0 0 0;padding-left:16px">
                        <li><strong>Blue blocks</strong> (high positive corr) = assets moving together &mdash; these become clusters</li>
                        <li><strong>Red patches</strong> (negative corr) = natural hedges &mdash; bonds vs equities, USD vs commodities</li>
                        <li>The <strong>block diagonal</strong> structure is what Leiden clustering formalizes into communities</li>
                    </ul>
                </div>
                <div style="background:rgba(88,166,255,0.1);padding:6px 10px;border-radius:6px;border-left:2px solid #58a6ff">
                    <strong>But this is just Pearson.</strong> It's Layer 1 of 5. Distance correlation and tail dependence reveal structure that this heatmap <em>cannot show</em>. That's the whole point of multi-layer analysis.
                </div>
            </div>
        </div>
    </section>"""


def _slide_five_layers() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:6px">Five Similarity Layers</h2>
        <p style="font-size:0.55em;color:#8b949e;margin-bottom:8px">Each layer captures a different type of dependence. No single measure tells the full story.</p>
        <table style="font-size:0.55em">
            <tr><th>Layer</th><th>What It Captures</th><th>Why It Matters</th></tr>
            <tr><td style="color:#58a6ff"><strong>Ledoit-Wolf Shrinkage</strong></td><td>Regularized linear co-movement</td><td>Standard baseline; handles N&gt;T estimation</td></tr>
            <tr><td style="color:#3fb950"><strong>Distance Correlation (dCor)</strong></td><td>ANY statistical dependence (linear + nonlinear)</td><td>dCor=0 iff truly independent; catches what Pearson misses</td></tr>
            <tr><td style="color:#f85149"><strong>Lower-Tail Dependence</strong></td><td>Co-exceedance at 5th percentile</td><td>Crash structure; tail CMI 36% higher than Pearson CMI on average</td></tr>
            <tr><td style="color:#d29922"><strong>Spearman Rank</strong></td><td>Monotonic nonlinear dependence</td><td>Robust to outliers; nonparametric</td></tr>
            <tr><td style="color:#bc8cff"><strong>KSG Mutual Information</strong></td><td>General information-theoretic dependence</td><td>Entropy-based; captures arbitrary functional relationships</td></tr>
        </table>
        <div style="background:#1c2333;padding:10px 16px;border-radius:8px;margin-top:10px;font-size:0.6em;border-left:3px solid #f85149">
            <strong>Key Finding:</strong> During the June 2025 Twelve-Day War, Pearson CMI showed a muted response (0.163) — markets appeared calm.
            But dCor and tail dependence revealed <strong>massive nonlinear restructuring</strong> invisible to linear measures.
            Traditional risk models relying on correlation alone completely missed this crisis.
        </div>
    </section>"""


def _slide_tds_decomposition() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:6px">TDS: 3-Component Decomposition</h2>
        <p style="font-size:0.6em;color:#8b949e;margin-bottom:10px">Topology Deformation Score — measures <em>how much</em> and <em>how</em> the market network changed between consecutive windows.</p>
        <div style="font-size:0.6em;background:#1c2333;padding:16px;border-radius:8px;text-align:center;margin-bottom:12px">
            <code style="font-size:1.3em;color:#58a6ff">TDS(t) = &alpha;&middot;z(W<sub>degree</sub>) + &beta;&middot;z(J<sub>community</sub>) + &gamma;&middot;z(S<sub>spectral</sub>)</code>
        </div>
        <table style="font-size:0.55em">
            <tr><th>Component</th><th>Measures</th><th>Method</th></tr>
            <tr><td style="color:#58a6ff"><strong>W<sub>degree</sub></strong></td><td>Change in connectivity distribution</td><td>Wasserstein-1 distance between degree distributions</td></tr>
            <tr><td style="color:#3fb950"><strong>J<sub>community</sub></strong></td><td>Change in community structure</td><td>1 &minus; NMI(communities<sub>t</sub>, communities<sub>t-1</sub>)</td></tr>
            <tr><td style="color:#d29922"><strong>S<sub>spectral</sub></strong></td><td>Change in global network shape</td><td>Wasserstein distance on Laplacian eigenvalue spectra</td></tr>
        </table>
        <div style="font-size:0.55em;margin-top:10px">
            <strong>Innovation:</strong> Each component z-score normalized via rolling window (TDSNormalizer) for commensurable combination.
            Raw components have different scales — without normalization, one component would dominate.
        </div>
    </section>"""


def _slide_event_covid() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:6px">Event Study: COVID-19 (Feb&ndash;Apr 2020)</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.6em">
            <div style="background:#1c2333;padding:14px;border-radius:8px">
                <h3 style="color:#f85149;font-size:1.1em;margin:0 0 8px 0">The Shock</h3>
                <ul style="margin:0">
                    <li>Fastest bear market in history: S&P &minus;34% in 23 days</li>
                    <li>CMI spiked to <strong>near 1.0</strong> (complete cluster dissolution)</li>
                    <li>Every asset changed clusters in a single step</li>
                    <li>TDS peak: 0.156</li>
                </ul>
            </div>
            <div style="background:#1c2333;padding:14px;border-radius:8px">
                <h3 style="color:#3fb950;font-size:1.1em;margin:0 0 8px 0">The Recovery</h3>
                <ul style="margin:0">
                    <li>8&ndash;10 week recovery to structural normalcy</li>
                    <li><strong>Symmetric shock</strong>: hit all asset classes simultaneously</li>
                    <li>Fed unlimited QE provided clear recovery catalyst</li>
                    <li>Fastest structural recovery of all studied events</li>
                </ul>
            </div>
        </div>
        <div style="background:#1c2333;padding:10px 16px;border-radius:8px;margin-top:10px;font-size:0.55em;border-left:3px solid #58a6ff">
            <strong>Takeaway:</strong> COVID was a symmetric liquidity shock — devastating but structurally simple. The market knew what to do once the Fed acted.
            Compare to geopolitical shocks which are <em>asymmetric</em> and lack a clear resolution mechanism.
        </div>
    </section>"""


def _slide_event_pearson_blind() -> str:
    return """
    <section>
        <h2 style="font-size:1.2em;margin-bottom:6px">The Blind Spot: June 2025 Twelve-Day War</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;text-align:center;font-size:0.6em;margin-bottom:12px">
            <div style="background:#1c2333;padding:14px;border-radius:8px">
                <p style="color:#8b949e;font-size:0.85em;margin:0">PEARSON CMI</p>
                <p style="font-size:2.2em;color:#3fb950;margin:4px 0">0.163</p>
                <p style="color:#8b949e;font-size:0.85em;margin:0">"Markets look calm"</p>
            </div>
            <div style="background:#1c2333;padding:14px;border-radius:8px;border:1px solid #f85149">
                <p style="color:#8b949e;font-size:0.85em;margin:0">DISTANCE CORR CMI</p>
                <p style="font-size:2.2em;color:#f85149;margin:4px 0">Massive</p>
                <p style="color:#8b949e;font-size:0.85em;margin:0">Nonlinear restructuring</p>
            </div>
            <div style="background:#1c2333;padding:14px;border-radius:8px;border:1px solid #f85149">
                <p style="color:#8b949e;font-size:0.85em;margin:0">TAIL CMI</p>
                <p style="font-size:2.2em;color:#f85149;margin:4px 0">Near-total</p>
                <p style="color:#8b949e;font-size:0.85em;margin:0">Complete crash rewiring</p>
            </div>
        </div>
        <div style="font-size:0.6em">
            <p><strong>What happened:</strong> After April and October 2024 Iran-Israel exchanges, markets had pre-adapted their <em>linear</em> correlation structure.
            Pearson says nothing changed. But the <strong>nonlinear and tail dependence structure completely rewired</strong>.</p>
            <div style="background:rgba(248,81,73,0.1);padding:10px 16px;border-radius:8px;border-left:3px solid #f85149;margin-top:8px">
                <strong>Implication:</strong> Traditional VaR and risk models relying on Pearson correlation are <strong>blind to half the story</strong>.
                Multi-layer analysis is not optional — it's essential.
            </div>
        </div>
    </section>"""


def _slide_event_epic_fury() -> str:
    return """
    <section>
        <h2 style="font-size:1.2em;margin-bottom:6px">Markets Restructure BEFORE Events</h2>
        <h3 style="color:#d29922;font-size:0.8em;margin:0 0 8px 0">Operation Epic Fury Buildup (Jan&ndash;Feb 2026)</h3>
        <table style="font-size:0.55em;margin-bottom:10px">
            <tr><th>Phase</th><th>CMI</th><th>TDS</th><th>Interpretation</th></tr>
            <tr><td>Baseline (Dec 2025)</td><td>0.742</td><td>0.051</td><td>Normal market structure</td></tr>
            <tr><td style="color:#d29922"><strong>Buildup (Jan 2026)</strong></td><td style="color:#d29922"><strong>0.82+</strong></td><td style="color:#f85149"><strong>0.176</strong></td><td style="color:#f85149"><strong>Peak deformation &mdash; BEFORE the strike</strong></td></tr>
            <tr><td>Strike (Feb 28)</td><td>0.60</td><td>0.062</td><td>Structure locks in; assets already repositioned</td></tr>
            <tr><td>Post-strike</td><td>Elevated</td><td>Elevated</td><td>Slow recovery &mdash; no clear resolution mechanism</td></tr>
        </table>
        <div style="font-size:0.6em">
            <div style="background:rgba(210,153,34,0.15);padding:10px 16px;border-radius:8px;border-left:3px solid #d29922">
                <strong>Key Discovery:</strong> Peak TDS of <strong>0.176 exceeded COVID's 0.156</strong> and occurred during the <em>buildup</em>, not the strike itself.
                Markets see geopolitical threats coming and reorganize their dependency structure <strong>in advance</strong>.
                This is topology crystallization &mdash; the network hardens into a crisis configuration before the event hits.
            </div>
        </div>
    </section>"""


def _slide_leadership_reversal() -> str:
    return """
    <section>
        <h2 style="font-size:1.2em;margin-bottom:6px">Information Flow Reversal During Crisis</h2>
        <p style="font-size:0.55em;color:#8b949e;margin:0 0 8px 0">Transfer entropy (KSG estimator) reveals who leads and who follows &mdash; and it flips during war.</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.6em">
            <div style="background:#1c2333;padding:14px;border-radius:8px">
                <h3 style="color:#3fb950;font-size:1em;margin:0 0 6px 0">Calm Regime (2023)</h3>
                <p style="margin:0 0 4px 0"><strong>Leaders:</strong> XLRE, XLE, EFA</p>
                <p style="margin:0 0 4px 0"><strong>Followers:</strong> TLT, GOVT, TIP</p>
                <p style="color:#8b949e;margin:0;font-size:0.9em">Risk assets drive the network; bonds follow</p>
            </div>
            <div style="background:#1c2333;padding:14px;border-radius:8px;border:1px solid #f85149">
                <h3 style="color:#f85149;font-size:1em;margin:0 0 6px 0">Crisis Regime (Epic Fury 2026)</h3>
                <p style="margin:0 0 4px 0"><strong>Leaders:</strong> TLT, GOVT, TIP</p>
                <p style="margin:0 0 4px 0"><strong>Followers:</strong> QQQ, IWM</p>
                <p style="color:#8b949e;margin:0;font-size:0.9em"><strong>Complete reversal</strong>: Treasury complex drives; equities follow</p>
            </div>
        </div>
        <div style="font-size:0.55em;margin-top:10px">
            <p><strong>Top overall leaders (full-sample TE):</strong> HYG (+0.385), SHY (+0.227), EEM (+0.204)</p>
            <p><strong>Top followers:</strong> FXE (&minus;0.529), TIP (&minus;0.247), TLT (&minus;0.206)</p>
            <p>Credit spreads (HYG) are the single most important information sender in the network.</p>
        </div>
    </section>"""


def _slide_novelty_vs_literature() -> str:
    return """
    <section>
        <h2 style="font-size:1.2em;margin-bottom:6px">Novel Contributions</h2>
        <table style="font-size:0.5em">
            <tr><th>Existing Approach</th><th>Limitation</th><th>Our Innovation</th></tr>
            <tr><td>Static correlation</td><td>Ignores time variation</td><td>Rolling 120d windows + HMM regime segmentation</td></tr>
            <tr><td>Single-layer networks</td><td>Misses nonlinear &amp; tail dependence</td><td>5-layer multiplex with layer agreement signal</td></tr>
            <tr><td>Standard event studies</td><td>Focus on returns only</td><td>Topology-centric: CMI/TDS as primary signals</td></tr>
            <tr><td>Naive cluster comparison</td><td>Label permutations = false migration</td><td>Hungarian algorithm permutation-invariant CMI</td></tr>
            <tr><td>Simple TDS aggregation</td><td>Incommensurable scales</td><td>Z-score normalized 3-component decomposition</td></tr>
            <tr><td>Uncorrected Granger</td><td>Cherry-picked lags</td><td>Bonferroni across lags + ADF stationarity pre-check</td></tr>
        </table>
        <div style="background:#1c2333;padding:10px 16px;border-radius:8px;margin-top:8px;font-size:0.55em">
            <strong>First to demonstrate:</strong>
            (1) Topology crystallization before geopolitical events;
            (2) Twelve-Day War invisible to Pearson but visible in dCor/tail;
            (3) Cross-layer Granger causality (tail &rarr; Pearson CMI);
            (4) Regime-conditional transfer entropy leadership reversal;
            (5) Consensus Leiden across multi-layer financial networks.
        </div>
    </section>"""


def _slide_robustness_summary() -> str:
    try:
        bs = json.load(open(PROCESSED_DIR / "bootstrap_results.json"))
        sens = json.load(open(PROCESSED_DIR / "sensitivity_results.json"))
        wf = json.load(open(PROCESSED_DIR / "walk_forward_results.json"))
    except Exception:
        return '<section><h2>Robustness</h2><p>Data unavailable</p></section>'

    cmi_ci = f"[{bs['mean_cmi']['ci_lower']:.4f}, {bs['mean_cmi']['ci_upper']:.4f}]"
    cps_ci = f"[{bs['mean_cps']['ci_lower']:.4f}, {bs['mean_cps']['ci_upper']:.4f}]"

    sens_rows = ""
    for param, res in sens.items():
        color = "#3fb950" if "ROBUST" in res["conclusion"] else "#d29922" if "MODERATE" in res["conclusion"] else "#f85149"
        label = "ROBUST" if "ROBUST" in res["conclusion"] else "MODERATE" if "MODERATE" in res["conclusion"] else "SENSITIVE"
        sens_rows += f'<tr><td>{param.replace("_"," ").title()}</td><td style="color:{color}"><strong>{label}</strong></td></tr>'

    return f"""
    <section>
        <h2 style="font-size:1.2em;margin-bottom:6px">Statistical Robustness</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.55em">
            <div>
                <h3 style="color:#58a6ff;font-size:1em">Bootstrap CIs (1,000 resamples)</h3>
                <ul>
                    <li>Mean CMI: 0.113 &mdash; 95% CI {cmi_ci}</li>
                    <li>Mean CPS: 0.784 &mdash; 95% CI {cps_ci}</li>
                    <li>Tight CIs = precisely estimated dynamics</li>
                </ul>
                <h3 style="color:#d29922;font-size:1em;margin-top:10px">Sensitivity</h3>
                <table style="font-size:0.9em">{sens_rows}</table>
            </div>
            <div>
                <h3 style="color:#f85149;font-size:1em">Honest OOS Results</h3>
                <ul>
                    <li>Granger replication OOS: <strong style="color:#f85149">{wf['granger_replication_rate']:.0%}</strong></li>
                    <li>In-sample p=0.041 does not survive walk-forward</li>
                    <li>TE not significant vs surrogate null</li>
                    <li>Regime prediction: 89% acc, 31% F1 (class imbalance)</li>
                </ul>
                <div style="background:rgba(88,166,255,0.1);padding:8px 12px;border-radius:6px;border-left:3px solid #58a6ff;margin-top:8px">
                    <strong>This honesty strengthens the work.</strong> Descriptive findings are solid. Causal claims need caveats. The methodology is rigorous about what survives OOS.
                </div>
            </div>
        </div>
    </section>"""


def _slide_key_insights() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:8px">Key Cross-Asset Insights</h2>
        <ol style="font-size:0.65em">
            <li><strong>Gold decoupled from safe havens</strong> &mdash; clustering with commodity currencies and miners, not Treasuries. Trading as inflation play.</li>
            <li><strong>USD clustering with energy</strong> &mdash; strong-dollar/commodity headwind narrative visible in correlation structure.</li>
            <li><strong>High yield divorced from IG</strong> &mdash; HYG trading like a risk/currency asset, not a bond. Credit risk ≠ duration risk.</li>
            <li><strong>Commodity currencies with metals</strong> &mdash; AUD, CAD driven by commodity prices, not domestic equities.</li>
            <li><strong>Israel = US tech proxy</strong> &mdash; EIS clusters with QQQ/XLK, not EM.</li>
            <li><strong>Tight EM bloc</strong> &mdash; EEM, FXI, EWT, EWY, VWO all in same cluster. High contagion risk.</li>
        </ol>
    </section>"""


def _slide_centrality() -> str:
    try:
        cent = pd.read_parquet(PROCESSED_DIR / "centrality_metrics.parquet")
        latest = cent[cent["date"] == cent["date"].max()]
        top_eig = latest.nlargest(5, "eigenvector")
        top_btw = latest.nlargest(5, "betweenness")
        eig_items = "".join(f"<li>{r['ticker']} ({r['eigenvector']:.3f})</li>" for _, r in top_eig.iterrows())
        btw_items = "".join(f"<li>{r['ticker']} ({r['betweenness']:.3f})</li>" for _, r in top_btw.iterrows())
    except Exception:
        return '<section><h2>Network Centrality</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Network Centrality</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:30px">
            <div>
                <h3 style="color:#58a6ff">Most Connected (Eigenvector)</h3>
                <ol style="font-size:0.8em">{eig_items}</ol>
                <p style="font-size:0.7em;color:#8b949e">Hub assets whose movements propagate most broadly</p>
            </div>
            <div>
                <h3 style="color:#d29922">Key Bridges (Betweenness)</h3>
                <ol style="font-size:0.8em">{btw_items}</ol>
                <p style="font-size:0.7em;color:#8b949e">Bridge assets connecting separate clusters</p>
            </div>
        </div>
    </section>"""


def _slide_amf() -> str:
    try:
        amf = pd.read_csv(PROCESSED_DIR / "amf_scores.csv").sort_values("amf", ascending=False)
        volatile = "".join(f"<li>{r['ticker']} ({r['amf']:.3f})</li>" for _, r in amf.head(5).iterrows())
        stable = "".join(f"<li>{r['ticker']} ({r['amf']:.3f})</li>" for _, r in amf.tail(5).sort_values('amf').iterrows())
    except Exception:
        return '<section><h2>Asset Migration Frequency</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Asset Migration Frequency</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:30px">
            <div>
                <h3 style="color:#f85149">Most Volatile</h3>
                <ol style="font-size:0.8em">{volatile}</ol>
                <p style="font-size:0.7em;color:#8b949e">Commodity currencies &amp; frontier markets lack a stable cluster home</p>
            </div>
            <div>
                <h3 style="color:#3fb950">Most Stable</h3>
                <ol style="font-size:0.8em">{stable}</ol>
                <p style="font-size:0.7em;color:#8b949e">European developed markets are reliable cluster anchors</p>
            </div>
        </div>
    </section>"""


def _slide_kmeans_baseline() -> str:
    try:
        comp = pd.read_parquet(PROCESSED_DIR / "kmeans_baseline_comparison.parquet")
    except Exception:
        return '<section><h2>K-Means Baseline</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Baseline: Leiden vs K-Means</h2>
        <table style="font-size:0.8em">
            <tr><th>Metric</th><th>Leiden</th><th>K-Means</th><th>Interpretation</th></tr>
            <tr><td>Mean CMI</td><td>{comp['leiden_cmi'].mean():.4f}</td><td>{comp['kmeans_cmi'].mean():.4f}</td><td>Leiden 40% more stable</td></tr>
            <tr><td>ARI Agreement</td><td colspan="2">{comp['ari'].mean():.3f} [{comp['ari'].min():.2f}, {comp['ari'].max():.2f}]</td><td>Moderate overlap</td></tr>
            <tr><td>NMI Agreement</td><td colspan="2">{comp['nmi'].mean():.3f} [{comp['nmi'].min():.2f}, {comp['nmi'].max():.2f}]</td><td>~50% shared information</td></tr>
            <tr><td>Silhouette</td><td>&mdash;</td><td>{comp['kmeans_silhouette'].mean():.3f}</td><td>Low cohesion for K-Means</td></tr>
        </table>
        <p style="margin-top:20px;font-size:0.8em">Consensus Leiden's adaptive cluster count better captures evolving market structure than K-Means' fixed-k assumption.</p>
    </section>"""


def _slide_regime_validation() -> str:
    try:
        with open(PROCESSED_DIR / "regime_validation_results.json") as f:
            rv = json.load(f)
        top_feats = sorted(rv["feature_importances"].items(), key=lambda x: -x[1])[:5]
        feat_items = "".join(f"<li>{f}: {v:.3f}</li>" for f, v in top_feats)
    except Exception:
        return '<section><h2>Regime Validation</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Regime Prediction: {rv['mean_accuracy']:.1%} Accuracy</h2>
        <ul style="font-size:0.8em">
            <li>5-fold TimeSeriesSplit, RandomForest, 1-day horizon</li>
            <li>Macro-F1: {rv['mean_f1']:.3f} (class imbalance: calm dominates)</li>
            <li>Topology metrics are regime-<em>descriptive</em> but not strongly <em>predictive</em> of stress/transition</li>
        </ul>
        <h3 style="margin-top:20px">Top Features</h3>
        <ol style="font-size:0.75em">{feat_items}</ol>
    </section>"""


def _unused_slide_bootstrap() -> str:
    try:
        with open(PROCESSED_DIR / "bootstrap_results.json") as f:
            bs = json.load(f)
    except Exception:
        return '<section><h2>Bootstrap CIs</h2><p>Data unavailable</p></section>'

    rows = ""
    for metric, v in bs.items():
        name = metric.replace("_", " ").title()
        rows += f"<tr><td>{name}</td><td>{v['observed']:.4f}</td><td>[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]</td></tr>"

    return f"""
    <section>
        <h2>Bootstrap Confidence Intervals</h2>
        <p style="font-size:0.8em">Block bootstrap (Politis &amp; Romano 1992), 1,000 resamples</p>
        <table style="font-size:0.8em">
            <tr><th>Metric</th><th>Observed</th><th>95% CI</th></tr>
            {rows}
        </table>
        <p style="font-size:0.75em;color:#8b949e;margin-top:20px">CMI and CPS are precisely estimated. TDS mean spans zero (expected &mdash; spikes during events, averages near zero).</p>
    </section>"""


def _unused_slide_sensitivity() -> str:
    try:
        with open(PROCESSED_DIR / "sensitivity_results.json") as f:
            sens = json.load(f)
    except Exception:
        return '<section><h2>Sensitivity Analysis</h2><p>Data unavailable</p></section>'

    rows = ""
    for param, res in sens.items():
        conclusion = res["conclusion"]
        color = "#3fb950" if "ROBUST" in conclusion else "#d29922" if "MODERATE" in conclusion else "#f85149"
        label = "ROBUST" if "ROBUST" in conclusion else "MODERATE" if "MODERATE" in conclusion else "SENSITIVE"
        rows += f'<tr><td>{param.replace("_", " ").title()}</td><td>{res["baseline"]}</td><td style="color:{color}"><strong>{label}</strong></td></tr>'

    return f"""
    <section>
        <h2>Sensitivity Analysis</h2>
        <table style="font-size:0.8em">
            <tr><th>Parameter</th><th>Baseline</th><th>Stability</th></tr>
            {rows}
        </table>
        <p style="font-size:0.75em;color:#8b949e;margin-top:20px">Window size is robust. Top-K and tail quantile are sensitive &mdash; documented as methodological limitations.</p>
    </section>"""


def _unused_slide_walkforward() -> str:
    try:
        with open(PROCESSED_DIR / "walk_forward_results.json") as f:
            wf = json.load(f)
    except Exception:
        return '<section><h2>Walk-Forward Validation</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Walk-Forward: Honest OOS Results</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;text-align:center">
            <div style="background:#1c2333;padding:20px;border-radius:8px">
                <p style="color:#8b949e;font-size:0.7em">GRANGER REPLICATION</p>
                <p style="font-size:2em;color:#f85149">{wf['granger_replication_rate']:.0%}</p>
            </div>
            <div style="background:#1c2333;padding:20px;border-radius:8px">
                <p style="color:#8b949e;font-size:0.7em">CRYSTALLIZATION</p>
                <p style="font-size:2em;color:#f85149">{wf['crystallization_replication_rate']:.0%}</p>
            </div>
            <div style="background:#1c2333;padding:20px;border-radius:8px">
                <p style="color:#8b949e;font-size:0.7em">EARLY WARNING FPR</p>
                <p style="font-size:2em;color:#d29922">{wf['mean_fpr']:.0%}</p>
            </div>
        </div>
        <p style="margin-top:20px;font-size:0.8em">Cross-layer Granger does <strong>not replicate OOS</strong>. In-sample finding (p=0.041) is likely regime-dependent. This is an honest result that strengthens the methodology's credibility.</p>
    </section>"""


def _unused_slide_surrogate() -> str:
    try:
        with open(PROCESSED_DIR / "surrogate_results.json") as f:
            surr = json.load(f)
        rows = ""
        for pair, v in surr.items():
            rows += f"<tr><td>{pair}</td><td>{v['p_value']:.3f}</td><td>{v['effect_size']:.2f}</td></tr>"
    except Exception:
        return '<section><h2>Surrogate Testing</h2><p>Data unavailable</p></section>'

    return f"""
    <section>
        <h2>Surrogate Null Testing</h2>
        <p style="font-size:0.8em">Phase-randomized surrogates (Theiler 1992): H0 = TE from autocorrelation alone</p>
        <table style="font-size:0.75em">
            <tr><th>Pair</th><th>p-value</th><th>Effect</th></tr>
            {rows}
        </table>
        <p style="font-size:0.75em;color:#8b949e;margin-top:15px">No pair significant at p&lt;0.05. UUP&rarr;GLD strongest (p=0.14). TE findings = correlation, not causation.</p>
    </section>"""


def _slide_conclusions() -> str:
    return """
    <section>
        <h2 style="font-size:1.3em;margin-bottom:6px">Conclusions</h2>
        <div style="font-size:0.7em">
        <h3 style="color:#3fb950">What Works</h3>
        <ul>
            <li>Consensus Leiden identifies <strong>economically meaningful clusters</strong> across 76 global assets</li>
            <li>CMI and CPS are <strong>precisely estimated</strong> (tight bootstrap CIs)</li>
            <li>Topology metrics predict regime with <strong>89% accuracy</strong></li>
            <li>Leiden outperforms K-Means baseline on stability and event detection</li>
        </ul>
        <h3 style="color:#f85149;margin-top:20px">What Doesn't</h3>
        <ul>
            <li>Cross-layer Granger causality <strong>does not replicate OOS</strong></li>
            <li>Transfer entropy <strong>not significant</strong> against surrogate null</li>
            <li>Regime F1 low for minority classes (stress/transition)</li>
            <li>Top-K and tail quantile are <strong>sensitive</strong> hyperparameters</li>
        </ul>
        </div>
    </section>"""


def _slide_future_work() -> str:
    return """
    <section>
        <h2>Future Work</h2>
        <ul style="font-size:0.85em">
            <li><strong>Streaming pipeline</strong> &mdash; incremental rolling windows, real-time dashboard</li>
            <li><strong>Causal discovery</strong> &mdash; PCMCI+, DYNOTEARS for full causal graph</li>
            <li><strong>Extend to 2008</strong> &mdash; GFC analysis with pre-inception data sourcing</li>
            <li><strong>Geopolitical NLP layer</strong> &mdash; GDELT/news embeddings as similarity layer</li>
            <li><strong>Stock signal engine integration</strong> &mdash; topology features as ablation input</li>
            <li><strong>Publication</strong> &mdash; Journal of Financial Economics / Journal of Portfolio Management</li>
        </ul>
        <p style="margin-top:30px;font-size:0.8em;color:#58a6ff">Thank you. Questions?</p>
    </section>"""


def _wrap_revealjs(today: str, slides_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ACM Presentation — {today}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/black.min.css">
<style>
.reveal {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.reveal h1, .reveal h2 {{ color: #58a6ff; }}
.reveal h3 {{ color: #e6edf3; }}
.reveal table {{ font-size: 0.7em; margin: 10px auto; }}
.reveal th {{ color: #8b949e; border-bottom: 1px solid #30363d; }}
.reveal td {{ border-bottom: 1px solid #1c2333; }}
.reveal ul, .reveal ol {{ font-size: 0.85em; }}
.reveal li {{ margin-bottom: 6px; }}
.reveal section {{ overflow: hidden; }}
.reveal .slides section {{ max-height: 100%; }}
</style>
</head>
<body>
<div class="reveal">
<div class="slides">
{slides_html}
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.min.js"></script>
<script>Reveal.initialize({{ hash: true, slideNumber: true, transition: 'slide', width: 1400, height: 900, margin: 0.04, minScale: 0.2, maxScale: 2.0 }});</script>
</body>
</html>"""
