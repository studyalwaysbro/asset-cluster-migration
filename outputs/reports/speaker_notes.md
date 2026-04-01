# ACM Presentation — Speaker Notes
## Nick Tavares | Stevens Institute of Technology | MS Financial Engineering

*These notes update daily with the pipeline. Read through before presenting — the data references will always reflect the latest run.*

---

### SLIDE 0: Title

> Hey everyone, I'm Nick. Today I'm presenting my research on something I think is genuinely underexplored in quantitative finance — how the *structure* of financial markets changes over time, and what happens to that structure when geopolitical events hit. Not returns. Not volatility. The actual topology — who's connected to who, how tightly, and how that rewires during a crisis.

---

### SLIDE 1: Motivation

> So here's the problem. When we do portfolio construction or risk management, we pull up a correlation matrix and treat it like it's some stable thing. Maybe we use a rolling window. But fundamentally, we're assuming that the *relationships between assets* are either fixed or change slowly. And that's just not true.

> What I found — and this is the core of this work — is that during geopolitical events, the entire dependency structure of the market can reorganize in a matter of days. And more interestingly, sometimes it reorganizes *before* the event even happens. The market sees it coming and restructures in advance.

> So the three questions driving this research are: how does topology change during events, do markets restructure before or during crises, and can we actually measure how fast and how severe that restructuring is?

---

### SLIDE 2: Methodology

> Let me walk through the framework. We've got four pieces here.

> **Data**: 134 ETFs spanning everything — US equity, all 11 GICS sectors, 18 country funds, bonds across the curve, commodities broken out by precious metals, energy, agriculture, industrial metals, eight currencies, crypto, managed futures, even frontier markets like Africa. 76 of those survive our data cleaning process — we need continuous daily data going back to 2010 to have enough history for the rolling windows.

> **Clustering**: We use 120-day rolling windows stepping forward 5 days at a time. That gives us 765 network snapshots over 16 years. For the similarity measure, we use Ledoit-Wolf shrinkage correlation — it's a regularized estimator that handles the case where you have more assets than observations in your window, which is exactly our situation.

> For community detection, we use the Leiden algorithm, which is an improvement over Louvain that guarantees connected communities. But here's the key — we don't run it once. We run consensus Leiden: 100 independent runs with resolution parameters sampled uniformly from 0.5 to 2.0, then fuse the results. This means the number of clusters is *adaptive* — the algorithm decides how many groups the market naturally falls into at each point in time. We're not forcing k=5 or k=8. The data tells us.

> **Novel metrics**: CMI, TDS, AMF, CPS — I'll explain each of these as we go, but the key innovation is that they're all permutation-invariant. The Leiden algorithm can shuffle cluster labels between runs, so a naive comparison would show migration that didn't actually happen. We solve this with the Hungarian algorithm to find the optimal label matching before computing any metrics.

> **Regimes**: A 3-state Gaussian Hidden Markov Model on a state vector of realized volatility, mean pairwise correlation, and cross-sectional dispersion. It identifies calm, transition, and stress states. About 90% of the time the market is calm. Stress is roughly 5% of trading days. But that 5% is where all the interesting topology changes happen.

---

### SLIDE 3: Five Similarity Layers

> This is maybe the most important methodological choice in the whole project. We don't just compute one correlation matrix — we compute five different measures of dependence between assets.

> **Ledoit-Wolf** is our baseline — it's the standard linear correlation, regularized so it doesn't blow up when N is close to T.

> **Distance correlation** is the game-changer. Unlike Pearson, dCor equals zero *if and only if* two series are truly statistically independent. Pearson can be zero even when there's a strong nonlinear relationship. dCor catches everything — quadratic, threshold effects, any functional form.

> **Lower-tail dependence** measures how assets co-move in the worst 5% of days. This is the crash structure. And here's a stat I love — tail CMI is on average 36% higher than Pearson CMI. That means the crash-structure of the market is *far more volatile* than the normal-day structure. The market's topology during bad days is fundamentally different and less stable than during good days.

> **Spearman** gives us a nonparametric monotonic measure — robust to outliers. And **KSG mutual information** uses k-nearest-neighbor entropy estimation to capture arbitrary functional relationships.

> And the key finding that justifies all five layers — during the June 2025 Twelve-Day War, Pearson showed a CMI of 0.163. Markets looked calm. But distance correlation and tail dependence showed massive restructuring that Pearson completely missed. If you only had one layer, you'd have told your risk committee that nothing happened. And you'd have been wrong.

---

### SLIDE 4: TDS Decomposition

> TDS is our measure of *how much* the market's network structure changed between consecutive windows. It has three components, and I want to explain why each one matters.

> **W-degree** uses the Wasserstein-1 distance — optimal transport distance — between the degree distributions of consecutive graphs. This tells you if the connectivity pattern changed. Did hubs become peripheries? Did isolated nodes suddenly get connected?

> **J-community** is one minus the Normalized Mutual Information between the community structures. NMI of 1 means identical clustering; NMI of 0 means completely different. So J measures how much the cluster membership reshuffled.

> **S-spectral** computes the Wasserstein distance on the Laplacian eigenvalue spectra. This is the most elegant one — the eigenvalues of the graph Laplacian encode the global shape of the network. Two networks with identical spectra have the same topology up to isomorphism. So this captures structural changes that might not show up in degree distributions or community labels.

> The innovation is the z-score normalization. Each component lives on a completely different scale. Without normalizing, the spectral component would dominate just because its raw values are larger. By z-scoring each component via a rolling normalizer, we make them commensurable — each contributes equally to the final score.

---

### SLIDE 5: Universe

> 134 ETFs across 19 groups — this is the broadest cross-asset universe I've seen in any network topology paper. Most studies use maybe 30-50 equities from one market. We're covering global equity, full Treasury curve, IG and HY credit, EM bonds, five types of commodities, eight currencies, crypto, frontier markets, managed futures.

> 76 survive cleaning because we need continuous data back to 2010. The ones that drop are mostly newer ETFs — the crypto funds, some thematic stuff, newer EM bonds. But 76 is still a very rich universe for network analysis.

---

### SLIDE 6: Migration Timeseries

> This is the money chart. 16 years of market topology evolution.

> The top panel is CMI — every time there's a spike, the market's cluster structure is reshuffling. You can see COVID as the massive spike in early 2020 — that's where CMI hit nearly 1.0, meaning *every single asset* changed clusters.

> But look at the colored event bands. The Iran escalations in 2024, the Japan carry trade unwind, the SVB banking crisis — each one produces a distinct CMI signature. And what's really interesting is the Epic Fury buildup in early 2026 — the TDS z-score in the bottom panel spikes to its highest value in the entire dataset, and it happens *during the buildup*, not during the actual strike.

> The green dashed line is the 30-window moving average — it shows the structural trend. You can see that since late 2025, the MA has been elevated. The market's baseline topology has been less stable than historical norms.

---

### SLIDE 7: Regime Status

> The HMM says we're in STRESS right now, and we have been oscillating between stress and transition for over 40 trading days. That's not a brief crisis — that's a sustained period where the market can't find a stable state.

> COVID was clean: calm, stress, calm. What we're seeing now is different — it's flickering. And that flickering pattern is itself a signal. When the HMM can't decide between two states, it means the underlying data is right at the decision boundary. Volatility is elevated but not extreme. Correlations are high but not at pandemic levels. The market is in this uncomfortable middle ground.

---

### SLIDE 8: COVID Event Study

> COVID is our reference event — the cleanest structural shock in the dataset. S&P dropped 34% in 23 trading days, fastest bear market in history.

> From a topology perspective, it produced complete cluster dissolution. CMI hit 1.0 — that means the Hungarian algorithm couldn't find *any* stable mapping between the before and after cluster structures. Every asset changed groups.

> But here's what's interesting — the recovery was fast. 8-10 weeks and the topology was back to something resembling normal. Why? Because COVID was a *symmetric* shock. It hit every asset class, every country, every sector simultaneously. And the Fed's unlimited QE gave a clear resolution mechanism. The market knew what the playbook was.

> Compare that to geopolitical shocks, which are asymmetric — they hit some assets more than others, and there's no central bank putting that fires out. Those produce slower, messier topology recovery.

---

### SLIDE 9: The Blind Spot (Twelve-Day War)

> This is my favorite finding in the entire project.

> June 2025, the Twelve-Day War between Iran and Israel. We compute CMI across all five similarity layers. Pearson CMI? 0.163. That's below the historical average. A traditional risk system would say markets barely noticed.

> But distance correlation CMI shows massive restructuring. Tail dependence CMI shows near-total rewiring of the crash structure. The nonlinear dependencies between assets completely changed, but the linear correlations didn't move.

> Why? Because after the April and October 2024 exchanges, markets had already pre-adapted their *linear* correlation structure to the Iran-Israel risk. The Pearson relationships were already pricing in conflict. But the *nonlinear* relationships — the tail dependencies, the information-theoretic connections — those hadn't adapted, and they got completely reshuffled.

> The implication is serious. Any risk model built on Pearson correlation — which is basically all of them, every VaR model, every mean-variance optimizer — was blind to this event. Multi-layer analysis isn't an academic luxury. It's essential for actually understanding what's happening.

---

### SLIDE 10: Epic Fury (Markets Restructure Before Events)

> This is the other paradigm-shifting finding. Operation Epic Fury — the latest Iran-Israel escalation in early 2026.

> Look at the TDS values in this table. During the baseline in December 2025, TDS was 0.051 — normal. During the buildup in January 2026, as tensions escalated and the strike was anticipated, TDS peaked at 0.176. That's *higher than COVID's peak of 0.156*.

> And then when the actual strike happened on February 28th? TDS dropped to 0.062. The structure had already locked in. Assets had already repositioned into their crisis configuration.

> This is what I call topology crystallization. The market network hardens into a crisis mode *before* the event. By the time the missiles fly, the repositioning is done. If you're waiting for the event to start hedging, you're already too late — the topology has already moved against you.

---

### SLIDE 11: Cluster Network

> This is the current snapshot of the market's network structure. Every dot is an ETF. The size reflects how central it is — bigger nodes have higher eigenvector centrality, meaning their movements propagate most broadly through the network.

> EEM — emerging markets — is the biggest node. It's the network hub right now. When EM moves, everything else feels it.

> Notice the tight purple EM cluster — EEM, FXI, EWT, EWY, VWO, EWZ, all packed together. That's contagion risk. If one of those breaks, they all break. They're essentially one bet right now.

> And look at where gold is. It's in the commodity cluster with copper, steel, palladium — *not* with Treasuries. Gold is trading as an inflation hedge and commodity play, not as a safe haven. That's a meaningful regime signal for anyone running a portfolio.

---

### SLIDE 12: Cluster Composition

> Eight clusters, each with a clear economic interpretation. That's one of the most satisfying results — the algorithm found groupings that actually make sense. We didn't impose any sector or asset class constraints. The Leiden algorithm just looked at the correlation structure and naturally separated bonds from equities from commodities from currencies.

> The fact that an unsupervised algorithm recovers economically intuitive groupings is itself a validation of the methodology.

---

### SLIDE 13: Cross-Asset Insights

> Six findings that a correlation matrix wouldn't tell you.

> Gold decoupled from safe havens. USD clustering with energy. HYG divorced from IG. Commodity currencies with metals. Israel as a tech proxy. Tight EM bloc.

> Each of these is a portfolio construction signal. If you're holding gold thinking it's your crisis hedge, this says you're actually holding a commodity position right now. If you think AUD and CAD diversify you away from equities, they don't — they're moving with copper and miners.

---

### SLIDE 14: Centrality

> EEM is the most connected asset by eigenvector and degree centrality. But MINT — a short-maturity bond ETF — has the highest *betweenness*. That means it's the bridge between clusters. It sits at the crossroads of risk-on and risk-off.

> In practical terms, MINT is the connective tissue of the market right now. It connects the equity world to the bond world. If you want to understand how stress transmits from one asset class to another, watch MINT.

---

### SLIDE 15: Correlation Heatmap

> This is the raw Pearson correlation matrix — the starting point. You can see the block diagonal structure clearly: equity bloc, bond bloc, commodity bloc, currency bloc.

> But I want to be upfront about what this *doesn't* show. This is Layer 1 of 5. The block structure here is real, but it's incomplete. Distance correlation and tail dependence reveal structure that this heatmap literally cannot represent. The whole point of multi-layer analysis is that no single view tells the full story.

---

### SLIDE 16: Leadership Reversal

> Transfer entropy measures directed information flow — not just correlation, but who's *leading* and who's *following*.

> During calm markets in 2023, the leaders were XLRE (real estate), XLE (energy), EFA (developed international). Risk assets were driving the network. Bonds were followers.

> During Epic Fury in 2026, it completely flipped. TLT, GOVT, TIP — the entire Treasury complex — became the primary information senders. QQQ and IWM became pure followers. In a crisis, bonds drive the bus and equities react.

> And the overall leader across the full sample? HYG — high-yield credit. With a net transfer entropy of +0.385, credit spreads are the single most important information sender in the global financial network. If you're only watching the S&P, you're watching the follower, not the leader.

---

### SLIDE 17: K-Means Baseline

> We compare our consensus Leiden approach against K-Means as a baseline. Leiden produces 40% less migration — more stable partitions. The ARI agreement is 0.277 — moderate overlap but distinct structure.

> The clincher is event detection. During COVID, Leiden's CMI spiked — it detected the structural break clearly. K-Means' CMI actually *dropped*. K-Means couldn't see it because it forces a fixed number of clusters. When the market dissolves into chaos, K-Means just shuffles assets between the same k buckets. Leiden adapts — it can go from 8 clusters to 3 to 12, tracking the actual structure.

---

### SLIDE 18: Regime Prediction

> Can topology metrics predict regime? 88.9% accuracy across 5 time-series folds. But I want to be honest about the F1 — it's 0.313, because the model crushes the calm class (89% of data) but can't reliably predict stress or transition.

> The interesting finding is the feature importance. CMI rolling mean is the #1 predictor at 0.325. Not the level of CMI — the *trend*. A rising migration rate signals regime transition. CMI volatility is #2. And cluster count — how many clusters exist — has zero importance. The number of clusters doesn't matter; what matters is how fast assets are moving between them.

---

### SLIDE 19: Robustness

> I'm going to be transparent about what survives out-of-sample and what doesn't.

> Bootstrap CIs are tight. Mean CMI is precisely estimated at 0.113 ± 0.008. Cluster persistence at 0.784 ± 0.014. The descriptive findings are rock solid.

> Sensitivity analysis: window size is moderate — conclusions hold whether you use 60-day or 252-day windows. Top-K and tail quantile are sensitive — those are hyperparameters that matter and I document them as methodological choices rather than universal truths.

> Now the honest part. The cross-layer Granger causality — tail CMI Granger-causes Pearson CMI with p=0.041 in-sample — does not replicate out-of-sample. Zero percent replication. Same with the crystallization pattern in walk-forward. Transfer entropy is not significant against phase-randomized surrogates.

> This doesn't invalidate the descriptive findings. The clusters are real, the migration metrics are precise, the event studies are robust. But the *causal* claims — that tail dependence *drives* Pearson restructuring, that TE reveals true directional causality — those need caveats. I think that honesty actually strengthens the work. Anyone can overfit a story. Showing what doesn't survive OOS testing is what separates rigorous research from data mining.

---

### SLIDE 20: Novel Contributions

> Six specific innovations versus the existing literature. Rolling windows with HMM instead of static correlation. Five-layer multiplex instead of single-layer. Topology-centric event studies instead of return-focused. Hungarian-matched CMI instead of naive label comparison. Z-score normalized TDS instead of raw aggregation. Bonferroni-corrected Granger instead of cherry-picked lags.

> And five "first to demonstrate" results: topology crystallization, the Twelve-Day War blind spot, cross-layer Granger, regime-conditional leadership reversal, and consensus Leiden on multi-layer financial networks.

---

### SLIDE 21: Conclusions

> What works: economically meaningful clusters, precisely estimated migration dynamics, 89% regime prediction accuracy, and Leiden outperforming K-Means on every metric.

> What doesn't: Granger doesn't replicate OOS, TE isn't significant against surrogates, regime F1 is low for minority classes.

> The big takeaway: if you're doing risk management or portfolio construction with just a correlation matrix, you're working with one-fifth of the information. Market topology is dynamic, it's multi-layered, and it restructures in advance of crises. Capturing that requires the kind of framework this research provides.

---

### SLIDE 22: Future Work

> Three directions. First, real-time — streaming data, incremental rolling windows, a dashboard that updates intraday. Second, deeper causality — PCMCI+ and DYNOTEARS for full causal graph learning, extending the Granger analysis that didn't survive OOS. Third, extending the data — back to 2008 for the GFC, higher-frequency intraday analysis during crisis windows.

> And the pipeline already runs daily as a cron job. Every afternoon at 3:30, the full 9-step pipeline fires, refreshes the data, re-runs clustering, regimes, migration, centrality, and generates this report automatically. So the research isn't static — it's a living system that updates every trading day.

> Thank you. Happy to take questions.
