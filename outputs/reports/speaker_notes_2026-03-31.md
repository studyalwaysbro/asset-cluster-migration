# ACM Presentation — Speaker Notes
## Nick Tavares | Stevens Institute of Technology | MS Financial Engineering
### Auto-generated 2026-03-31 — numbers reflect latest pipeline run

*These notes update daily with the pipeline. The data references always reflect the latest run.*

**Current state:** 76 assets, 3,944 trading days, 764 windows, 8 clusters, regime = STRESS (41 days since calm, 41 transitions in 60d), latest CMI = 0.2105, mean CMI = 0.1129

---

### SLIDE 0: Title

> Hey everyone, I'm Nick. Today I'm presenting my research on something I think is genuinely underexplored in quantitative finance — how the *structure* of financial markets changes over time, and what happens to that structure when geopolitical events hit. Not returns. Not volatility. The actual topology — who's connected to who, how tightly, and how that rewires during a crisis.

---

### SLIDE 1: Motivation

> So here's the problem. When we do portfolio construction or risk management, we pull up a correlation matrix and treat it like it's some stable thing. Maybe we use a rolling window. But fundamentally, we're assuming that the relationships between assets are either fixed or change slowly. And that's just not true.

> What I found — and this is the core of this work — is that during geopolitical events, the entire dependency structure of the market can reorganize in a matter of days. And more interestingly, sometimes it reorganizes *before* the event even happens. The market sees it coming and restructures in advance.

> So the three questions driving this research are: how does topology change during events, do markets restructure before or during crises, and can we actually measure how fast and how severe that restructuring is?

---

### SLIDE 2: Methodology

> Let me walk through the framework. Four pieces.

> **Data**: 134 ETFs spanning everything — US equity, all 11 GICS sectors, 18 country funds, bonds across the curve, five commodity sub-groups, eight currencies, crypto, managed futures, frontier markets. 76 of those survive cleaning because we need continuous daily data back to 2010.

> **Clustering**: 120-day rolling windows, 5-day step, giving us 764 network snapshots over 16 years. We use Ledoit-Wolf shrinkage correlation — a regularized estimator that handles when you have more assets than observations in your window.

> For community detection, we use consensus Leiden — 100 independent runs with resolution parameters sampled from 0.5 to 2.0, then fuse the results. The number of clusters is *adaptive*. We're not forcing k=5 or k=8. The data decides. Right now it's finding 8 clusters.

> **Novel metrics**: CMI, TDS, AMF, CPS — all permutation-invariant via the Hungarian algorithm. The Leiden algorithm can shuffle cluster labels between runs, so without the Hungarian matching, you'd see migration that didn't actually happen.

> **Regimes**: A 3-state Gaussian HMM on realized vol, mean pairwise correlation, and cross-sectional dispersion. Currently the market is in STRESS, and has been non-calm for 41 trading days with 41 transitions in the last 60 days.

---

### SLIDE 3: Five Similarity Layers

> This is maybe the most important methodological choice. We compute five different dependence measures.

> **Ledoit-Wolf** — standard linear correlation, regularized. **Distance correlation** — the game-changer, equals zero if and only if truly independent, catches all nonlinear dependence. **Lower-tail dependence** — co-movement in the worst 5% of days. Here's a stat I love: tail CMI is on average 36% higher than Pearson CMI. The crash-structure of the market is *far more volatile* than the normal-day structure.

> **Spearman** — nonparametric monotonic measure. **KSG mutual information** — information-theoretic, captures arbitrary functional relationships.

> The killer finding: during the June 2025 Twelve-Day War, Pearson showed CMI of 0.163 — markets looked calm. But dCor and tail dependence showed massive restructuring that Pearson completely missed. If you only had one layer, you'd have told your risk committee nothing happened. And you'd have been wrong.

---

### SLIDE 4: TDS Decomposition

> TDS has three components. W-degree uses Wasserstein-1 distance between degree distributions — did hubs become peripheries? J-community is one minus NMI between cluster structures — how much did membership reshuffle? S-spectral computes Wasserstein distance on Laplacian eigenvalue spectra — this captures the global shape of the network.

> The innovation is z-score normalization. Each component lives on a different scale. Without normalizing, one would dominate. By z-scoring via a rolling window, they contribute equally. Simple idea, but nobody had done it for financial networks before.

---

### SLIDE 5: Universe

> 134 ETFs, 19 groups, 76 survive. Broadest cross-asset universe I've seen in any network topology paper. Most studies use 30-50 equities from one market. We're covering the whole global capital markets landscape.

---

### SLIDE 6: Migration Timeseries

> The money chart. 16 years of topology evolution.

> Top panel: CMI — every spike means the market's cluster structure is reshuffling. COVID is the massive spike in 2020 where CMI hit nearly 1.0 — every asset changed clusters.

> Bottom panel: TDS z-score. Above 2.0 is our alert threshold. The event bands are annotated — COVID, Fed tightening, SVB, Iran escalations, Japan carry unwind.

> The key thing: notice how the Epic Fury buildup in early 2026 produced the highest TDS z-score in the entire dataset, and it happened *during the buildup*, before the actual strike. That's topology crystallization.

---

### SLIDE 7: Regime Status

> STRESS regime, 41 days since calm, 41 transitions in 60 days. The market is flickering between stress and transition — it can't settle. That flickering is itself a signal. The HMM is telling us the underlying data is right at the decision boundary.

---

### SLIDE 8: COVID

> Our reference event. Complete cluster dissolution — CMI hit 1.0, the Hungarian algorithm couldn't find any stable mapping. But recovery was fast — 8-10 weeks. Why? Symmetric shock plus a clear Fed response. The market knew the playbook.

> Compare to geopolitical shocks — asymmetric, no central bank fix. Those produce slower, messier recovery.

---

### SLIDE 9: The Blind Spot

> My favorite finding. Pearson CMI 0.163 — "markets look calm." Distance correlation and tail dependence — massive restructuring. Traditional VaR was blind to this event. Multi-layer analysis is essential.

---

### SLIDE 10: Epic Fury

> Peak TDS of 0.176 exceeded COVID's 0.156 and happened during the *buildup*, not the strike. This is topology crystallization — the market hardens into crisis mode in advance. If you're waiting for the event to hedge, you're already too late.

---

### SLIDE 11-15: Network, Clusters, Insights, Centrality, Heatmap

> [Walk through the current market structure. Use the narration panels on each slide. Key talking points: EEM as hub, gold decoupled from safe havens, HYG divorced from IG, commodity currencies with metals, MINT as bridge asset, heatmap as "just Layer 1 of 5."]

---

### SLIDE 16: Leadership Reversal

> Transfer entropy shows who leads and who follows — and it completely flips during crisis. Calm: risk assets lead, bonds follow. Crisis: bonds lead, equities follow. HYG (credit spreads) is the overall #1 information sender. If you're only watching the S&P, you're watching the follower.

---

### SLIDE 17: K-Means Baseline

> Leiden is 40% more stable than K-Means and detects structural breaks that K-Means misses. K-Means forces fixed k — when the market dissolves into chaos, it just shuffles assets between the same buckets. Leiden adapts.

---

### SLIDE 18: Regime Prediction

> 88.9% accuracy but 31% F1 because of class imbalance. The model predicts calm well but can't reliably predict stress/transition. CMI rolling mean is the #1 feature. Topology metrics are descriptive, not strongly predictive of minority classes. Honest result.

---

### SLIDE 19: Robustness

> Bootstrap CIs: Mean CMI [0.1052, 0.1214], Mean CPS [0.7695, 0.7977]. Tight — descriptive findings are precise.

> The honest part: Granger doesn't replicate OOS. TE not significant against surrogates. This doesn't invalidate the descriptive findings. But the causal claims need caveats. I think that honesty strengthens the work.

---

### SLIDE 20: Novel Contributions

> Six innovations vs literature. Five "first to demonstrate" results. The core novelty: treating cluster migration as a signal, not just counting clusters.

---

### SLIDE 21: Conclusions

> What works: meaningful clusters, precise metrics, 89% regime accuracy. What doesn't: Granger OOS, TE significance. Big takeaway: one correlation matrix gives you one-fifth of the information. Market topology is dynamic, multi-layered, and anticipatory.

---

### SLIDE 22: Future Work

> Real-time streaming, causal discovery, extend to 2008. The pipeline already runs daily at 3:30 PM as a cron. The research is a living system. Thank you — happy to take questions.
