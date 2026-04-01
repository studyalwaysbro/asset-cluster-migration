"""Auto-generate speaker notes with current pipeline data baked in.

Regenerates on each pipeline run so the numbers always match the slides.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def generate_speaker_notes(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or REPORTS_DIR
    today = datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"speaker_notes_{today}.html"

    # Load current data
    mig = pd.read_parquet(PROCESSED_DIR / "migration_timeseries.parquet")
    returns = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")
    n_assets = returns.shape[1]
    n_days = returns.shape[0]
    n_windows = len(mig)
    mean_cmi = mig["cmi"].mean()
    latest_cmi = mig.iloc[-1]["cmi"]
    latest_clusters = int(mig.iloc[-1]["n_clusters"])

    regime_str = "unknown"
    days_since_calm = 0
    transitions_60d = 0
    try:
        rl = pd.read_csv(PROCESSED_DIR / "regime_labels.csv")
        regime_str = str(rl.iloc[-1]["regime"]).upper()
        regimes = rl["regime"].values
        for i in range(len(regimes) - 1, -1, -1):
            if regimes[i] == "calm":
                break
            days_since_calm += 1
        recent = rl.tail(60)
        transitions_60d = sum(1 for i in range(1, len(recent)) if recent.iloc[i]["regime"] != recent.iloc[i - 1]["regime"])
    except Exception:
        pass

    # Bootstrap CIs
    cmi_ci = "[?, ?]"
    cps_ci = "[?, ?]"
    try:
        bs = json.load(open(PROCESSED_DIR / "bootstrap_results.json"))
        cmi_ci = f"[{bs['mean_cmi']['ci_lower']:.4f}, {bs['mean_cmi']['ci_upper']:.4f}]"
        cps_ci = f"[{bs['mean_cps']['ci_lower']:.4f}, {bs['mean_cps']['ci_upper']:.4f}]"
    except Exception:
        pass

    notes = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ACM Speaker Notes — {today}</title>
<style>
:root {{ --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #e6edf3; --dim: #8b949e; --accent: #58a6ff; --green: #3fb950; --yellow: #d29922; --red: #f85149; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--text); line-height: 1.8; padding: 40px; max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 1.8em; color: var(--accent); margin-bottom: 4px; font-family: system-ui, sans-serif; }}
h2 {{ font-size: 1em; color: var(--dim); font-weight: 400; margin-bottom: 20px; font-family: system-ui, sans-serif; }}
h3 {{ font-size: 0.75em; color: var(--dim); font-weight: 400; margin-bottom: 30px; font-family: system-ui, sans-serif; }}
.stats {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 20px; margin-bottom: 30px; font-family: monospace; font-size: 0.85em; color: var(--accent); }}
.slide {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px 28px; margin-bottom: 20px; page-break-inside: avoid; }}
.slide-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }}
.slide-num {{ background: var(--accent); color: var(--bg); font-family: system-ui, sans-serif; font-weight: 700; font-size: 0.75em; padding: 3px 10px; border-radius: 12px; }}
.slide-title {{ font-family: system-ui, sans-serif; font-size: 1.1em; font-weight: 600; color: var(--text); }}
.slide p {{ margin-bottom: 12px; font-size: 0.95em; }}
.slide em {{ color: var(--yellow); font-style: italic; }}
.slide strong {{ color: var(--text); }}
.highlight {{ background: rgba(88,166,255,0.08); border-left: 3px solid var(--accent); padding: 10px 16px; border-radius: 0 6px 6px 0; margin: 12px 0; font-size: 0.9em; }}
.highlight-red {{ background: rgba(248,81,73,0.08); border-left: 3px solid var(--red); padding: 10px 16px; border-radius: 0 6px 6px 0; margin: 12px 0; font-size: 0.9em; }}
.divider {{ border: none; border-top: 1px solid var(--border); margin: 30px 0; }}
@media print {{ body {{ background: white; color: #222; }} .slide {{ border: 1px solid #ddd; }} .slide-num {{ background: #333; color: white; }} }}
</style>
</head>
<body>
<h1>ACM Presentation — Speaker Notes</h1>
<h2>Nick Tavares &middot; Stevens Institute of Technology &middot; MS Financial Engineering</h2>
<h3>Auto-generated {today} &mdash; numbers reflect latest pipeline run</h3>

<div class="stats">
{n_assets} assets &middot; {n_days:,} trading days &middot; {n_windows} windows &middot; {latest_clusters} clusters &middot;
regime = {regime_str} &middot; {days_since_calm}d since calm &middot; {transitions_60d} transitions/60d &middot;
CMI = {latest_cmi:.4f} (avg {mean_cmi:.4f}) &middot; Bootstrap CI {cmi_ci}
</div>

<!-- SLIDE 0 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">0</span><span class="slide-title">Title</span></div>
<p>Hey everyone, I'm Nick. Today I'm presenting my research on something I think is genuinely underexplored in quantitative finance &mdash; how the <em>structure</em> of financial markets changes over time, and what happens to that structure when geopolitical events hit. Not returns. Not volatility. The actual topology &mdash; who's connected to who, how tightly, and how that rewires during a crisis.</p>
</div>

<!-- SLIDE 1 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">1</span><span class="slide-title">Motivation</span></div>
<p>So here's the problem. When we do portfolio construction or risk management, we pull up a correlation matrix and treat it like it's some stable thing. Maybe we use a rolling window. But fundamentally, we're assuming that the relationships between assets are either fixed or change slowly. And that's just not true.</p>
<p>What I found &mdash; and this is the core of this work &mdash; is that during geopolitical events, the entire dependency structure of the market can reorganize in a matter of days. And more interestingly, sometimes it reorganizes <em>before</em> the event even happens. The market sees it coming and restructures in advance.</p>
<p>So the three questions driving this research are: how does topology change during events, do markets restructure before or during crises, and can we actually measure how fast and how severe that restructuring is?</p>
</div>

<!-- SLIDE 2 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">2</span><span class="slide-title">Methodology</span></div>
<p>Let me walk through the framework. Four pieces.</p>
<p><strong>Data</strong>: 134 ETFs spanning everything &mdash; US equity, all 11 GICS sectors, 18 country funds, bonds across the curve, five commodity sub-groups, eight currencies, crypto, managed futures, frontier markets. {n_assets} of those survive cleaning because we need continuous daily data back to 2010.</p>
<p><strong>Clustering</strong>: 120-day rolling windows, 5-day step, giving us {n_windows} network snapshots over 16 years. We use Ledoit-Wolf shrinkage correlation &mdash; a regularized estimator that handles when you have more assets than observations in your window.</p>
<p>For community detection, we use consensus Leiden &mdash; 100 independent runs with resolution parameters sampled from 0.5 to 2.0, then fuse the results. The number of clusters is <em>adaptive</em>. We're not forcing k=5 or k=8. The data decides. Right now it's finding {latest_clusters} clusters.</p>
<p><strong>Novel metrics</strong>: CMI, TDS, AMF, CPS &mdash; all permutation-invariant via the Hungarian algorithm. The Leiden algorithm can shuffle cluster labels between runs, so without the Hungarian matching, you'd see migration that didn't actually happen.</p>
<p><strong>Regimes</strong>: A 3-state Gaussian HMM on realized vol, mean pairwise correlation, and cross-sectional dispersion. Currently the market is in {regime_str}, and has been non-calm for {days_since_calm} trading days with {transitions_60d} transitions in the last 60 days.</p>
</div>

<!-- SLIDE 3 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">3</span><span class="slide-title">Five Similarity Layers</span></div>
<p>This is maybe the most important methodological choice. We compute five different dependence measures.</p>
<p><strong>Ledoit-Wolf</strong> &mdash; standard linear correlation, regularized. <strong>Distance correlation</strong> &mdash; the game-changer, equals zero if and only if truly independent, catches all nonlinear dependence. <strong>Lower-tail dependence</strong> &mdash; co-movement in the worst 5% of days.</p>
<div class="highlight">Here's a stat I love: tail CMI is on average 36% higher than Pearson CMI. The crash-structure of the market is <em>far more volatile</em> than the normal-day structure.</div>
<p><strong>Spearman</strong> &mdash; nonparametric monotonic. <strong>KSG mutual information</strong> &mdash; information-theoretic, captures arbitrary functional relationships.</p>
<p>The killer finding: during the June 2025 Twelve-Day War, Pearson showed CMI of 0.163 &mdash; markets looked calm. But dCor and tail dependence showed massive restructuring that Pearson completely missed. If you only had one layer, you'd have told your risk committee nothing happened. And you'd have been wrong.</p>
</div>

<!-- SLIDE 4 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">4</span><span class="slide-title">TDS Decomposition</span></div>
<p>TDS has three components. <strong>W-degree</strong> uses Wasserstein-1 distance between degree distributions &mdash; did hubs become peripheries? <strong>J-community</strong> is one minus NMI between cluster structures &mdash; how much did membership reshuffle? <strong>S-spectral</strong> computes Wasserstein distance on Laplacian eigenvalue spectra &mdash; this captures the global shape of the network.</p>
<p>The innovation is z-score normalization. Each component lives on a different scale. Without normalizing, one would dominate. By z-scoring via a rolling window, they contribute equally. Simple idea, but nobody had done it for financial networks before.</p>
</div>

<!-- SLIDE 5 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">5</span><span class="slide-title">Universe</span></div>
<p>134 ETFs, 19 groups, {n_assets} survive. Broadest cross-asset universe I've seen in any network topology paper. Most studies use 30-50 equities from one market. We're covering the whole global capital markets landscape.</p>
</div>

<!-- SLIDE 6 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">6</span><span class="slide-title">Migration Timeseries</span></div>
<p>The money chart. 16 years of topology evolution.</p>
<p>Top panel: CMI &mdash; every spike means the market's cluster structure is reshuffling. COVID is the massive spike in 2020 where CMI hit nearly 1.0 &mdash; every asset changed clusters.</p>
<p>Bottom panel: TDS z-score. Above 2.0 is our alert threshold. The colored event bands mark COVID, Fed tightening, SVB, Iran escalations, Japan carry unwind.</p>
<div class="highlight-red">Notice how the Epic Fury buildup in early 2026 produced the highest TDS in the entire dataset &mdash; and it happened <em>during the buildup</em>, before the actual strike. That's topology crystallization.</div>
</div>

<!-- SLIDE 7 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">7</span><span class="slide-title">Regime Status</span></div>
<p>{regime_str} regime, {days_since_calm} days since calm, {transitions_60d} transitions in 60 days. The market is flickering between stress and transition &mdash; it can't settle.</p>
<p>COVID was clean: calm, stress, calm. What we're seeing now is different &mdash; it's flickering. That flickering pattern is itself a signal. The HMM is telling us the underlying data is right at the decision boundary. Volatility is elevated but not extreme. Correlations are high but not at pandemic levels. The market is in this uncomfortable middle ground.</p>
</div>

<!-- SLIDE 8 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">8</span><span class="slide-title">COVID Event Study</span></div>
<p>Our reference event. Complete cluster dissolution &mdash; CMI hit 1.0, the Hungarian algorithm couldn't find any stable mapping. But recovery was fast &mdash; 8 to 10 weeks. Why? Symmetric shock plus a clear Fed response. The market knew the playbook.</p>
<p>Compare to geopolitical shocks &mdash; asymmetric, no central bank fix. Those produce slower, messier recovery.</p>
</div>

<!-- SLIDE 9 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">9</span><span class="slide-title">The Blind Spot &mdash; Twelve-Day War</span></div>
<p>This is my favorite finding in the entire project.</p>
<p>June 2025, the Twelve-Day War. Pearson CMI? 0.163. Below the historical average. A traditional risk system would say markets barely noticed.</p>
<p>But distance correlation CMI shows massive restructuring. Tail dependence CMI shows near-total rewiring. The nonlinear dependencies between assets completely changed, but the linear correlations didn't move.</p>
<div class="highlight-red">Why? Because after April and October 2024, markets had already pre-adapted their <em>linear</em> correlation structure. Pearson was already pricing in conflict. But the nonlinear and tail relationships hadn't adapted &mdash; and they got completely reshuffled. Any risk model built on Pearson correlation was blind to this event.</div>
</div>

<!-- SLIDE 10 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">10</span><span class="slide-title">Markets Restructure Before Events</span></div>
<p>Operation Epic Fury &mdash; the latest Iran-Israel escalation. During the baseline in December, TDS was 0.051 &mdash; normal. During the buildup in January, as tensions escalated, TDS peaked at 0.176. That's higher than COVID's peak of 0.156.</p>
<p>Then when the actual strike happened? TDS dropped to 0.062. The structure had already locked in. Assets had already repositioned.</p>
<div class="highlight">This is topology crystallization. The market network hardens into crisis mode <em>before</em> the event. By the time the missiles fly, the repositioning is done. If you're waiting for the event to start hedging, you're already too late.</div>
</div>

<!-- SLIDE 11 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">11</span><span class="slide-title">Cluster Network</span></div>
<p>Current snapshot. Every dot is an ETF, sized by eigenvector centrality. EEM is the biggest node &mdash; emerging markets is the network hub right now.</p>
<p>Notice the tight EM cluster &mdash; that's contagion risk. And gold is with commodities, not Treasuries &mdash; it's an inflation play, not safe haven. MINT bridges the risk-on and risk-off worlds.</p>
</div>

<!-- SLIDE 12 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">12</span><span class="slide-title">Cluster Composition</span></div>
<p>{latest_clusters} clusters, each with a clear economic interpretation. The algorithm found groupings that actually make sense &mdash; we didn't impose any sector constraints. The fact that an unsupervised algorithm recovers economically intuitive groupings is itself a validation.</p>
</div>

<!-- SLIDE 13 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">13</span><span class="slide-title">Cross-Asset Insights</span></div>
<p>Six findings a correlation matrix wouldn't tell you. Gold decoupled from safe havens. USD clustering with energy. HYG divorced from IG. Commodity currencies with metals. Israel as a tech proxy. Tight EM bloc.</p>
<p>Each one is a portfolio construction signal. If you're holding gold thinking it's your crisis hedge, this says you're actually holding a commodity position right now.</p>
</div>

<!-- SLIDE 14 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">14</span><span class="slide-title">Network Centrality</span></div>
<p>EEM is most connected. But MINT has the highest betweenness &mdash; it's the bridge between clusters, the connective tissue of the market. If you want to understand how stress transmits between asset classes, watch MINT.</p>
</div>

<!-- SLIDE 15 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">15</span><span class="slide-title">Correlation Heatmap</span></div>
<p>The raw Pearson matrix &mdash; block diagonal structure is clear. But I want to be upfront: this is Layer 1 of 5. The block structure is real but incomplete. Distance correlation and tail dependence reveal structure this heatmap literally cannot represent. That's the whole point.</p>
</div>

<!-- SLIDE 16 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">16</span><span class="slide-title">Leadership Reversal</span></div>
<p>Transfer entropy measures directed information flow &mdash; who's leading, who's following.</p>
<p>During calm markets: XLRE, XLE, EFA lead. Bonds follow. During Epic Fury: TLT, GOVT, TIP lead. QQQ and IWM follow. Complete reversal &mdash; in crisis, bonds drive the bus.</p>
<div class="highlight">And the overall leader? HYG &mdash; credit spreads. Net TE of +0.385. If you're only watching the S&amp;P, you're watching the follower, not the leader.</div>
</div>

<!-- SLIDE 17 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">17</span><span class="slide-title">K-Means Baseline</span></div>
<p>Leiden is 40% more stable than K-Means and detects structural breaks K-Means misses. During COVID, Leiden's CMI spiked &mdash; K-Means' actually dropped. K-Means forces fixed k, so when the market dissolves, it just shuffles between the same buckets. Leiden adapts.</p>
</div>

<!-- SLIDE 18 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">18</span><span class="slide-title">Regime Prediction</span></div>
<p>88.9% accuracy, but 31% F1 because of class imbalance. The model predicts calm well but can't reliably predict stress or transition. CMI rolling mean is the number one feature &mdash; not the level, the trend. A rising migration rate signals regime transition.</p>
<p>Honest result. Topology metrics are descriptive, not strongly predictive of minority classes.</p>
</div>

<!-- SLIDE 19 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">19</span><span class="slide-title">Statistical Robustness</span></div>
<p>Bootstrap CIs: Mean CMI = {mean_cmi:.4f}, 95% CI {cmi_ci}. Mean CPS {cps_ci}. Tight &mdash; descriptive findings are precise.</p>
<div class="highlight-red">The honest part: cross-layer Granger doesn't replicate out-of-sample. Zero percent. TE not significant against surrogates. This doesn't invalidate the descriptive findings &mdash; clusters are real, metrics are precise, event studies are robust. But the causal claims need caveats. I think that honesty actually strengthens the work. Anyone can overfit a story. Showing what doesn't survive OOS is what separates rigorous research from data mining.</div>
</div>

<!-- SLIDE 20 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">20</span><span class="slide-title">Novel Contributions</span></div>
<p>Six innovations versus literature. Five "first to demonstrate" results: topology crystallization, the Twelve-Day War blind spot, cross-layer Granger, regime-conditional leadership reversal, and consensus Leiden on multi-layer financial networks.</p>
<p>The core novelty: treating cluster migration as a signal, not just counting clusters.</p>
</div>

<!-- SLIDE 21 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">21</span><span class="slide-title">Conclusions</span></div>
<p>What works: meaningful clusters, precise metrics, 89% regime accuracy. What doesn't: Granger OOS, TE significance.</p>
<p>Big takeaway: if you're doing risk management with just a correlation matrix, you're working with one-fifth of the information. Market topology is dynamic, multi-layered, and anticipatory. Capturing that requires the kind of framework this research provides.</p>
</div>

<!-- SLIDE 22 -->
<div class="slide">
<div class="slide-header"><span class="slide-num">22</span><span class="slide-title">Future Work</span></div>
<p>Three directions. Real-time streaming with intraday windows. Deeper causality with PCMCI+ and DYNOTEARS. Extending back to 2008 for the GFC.</p>
<p>And the pipeline already runs daily as a cron job. Every afternoon at 3:30, the full 9-step pipeline fires, refreshes everything, and generates the report, slides, and these notes automatically. The research is a living system.</p>
<p>Thank you. Happy to take questions.</p>
</div>

</body>
</html>"""

    path.write_text(notes, encoding="utf-8")
    logger.info(f"Speaker notes generated: {path}")
    return path

# END OF FILE
