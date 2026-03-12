# Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress

> **Disclaimer:** This project is produced solely for educational and academic research purposes. Nothing in this repository constitutes investment advice, a solicitation, or a recommendation to buy, sell, or hold any security or financial instrument. This project is not connected to, endorsed by, or produced in any capacity related to UBS or any broker-dealer. The authors are presenting independent academic research on asset clustering methodologies. Past performance and historical patterns do not guarantee future results.

## Authors

- **Nicholas Tavares**
- **Amjad Hanini**
- **Brandon Dasilva**

## Overview

A quantitative research framework that tracks how multi-asset portfolio structures reorganize during geopolitical crises. We apply graph theory, community detection, information theory, and multi-layer network analysis to a universe of 91 ETFs spanning equities, fixed income, commodities, currencies, crypto, managed futures, thematics, and 18 country-specific funds from January 2019 through the most recent trading day.

### Three-Phase Architecture

| Phase | Focus | Methods |
|-------|-------|---------|
| **Phase 1** | Single-layer topology | Shrinkage correlation, Leiden clustering, CMI/TDS metrics, HMM regime detection |
| **Phase 2** | Multi-layer analysis | Distance correlation, tail dependence, multiplex consensus clustering, layer agreement |
| **Phase 3** | Directional flow | Transfer entropy (KSG), Granger causality, lead-lag networks, cross-layer causality |

### Key Findings

1. **Markets restructure BEFORE the event** - Peak topology deformation (TDS=0.176, exceeding COVID) occurred during the buildup to Operation Epic Fury, not during the strikes
2. **Correlation alone is insufficient** - The June 2025 Twelve-Day War was invisible to Pearson correlation but showed massive restructuring in nonlinear and tail-dependence measures
3. **Tail-dependence CMI Granger-causes Pearson CMI** (p=0.041) - Crash-structure changes predict normal-correlation changes, providing a potential early warning signal
4. **Leadership reversal during war** - Calm markets: credit/real estate lead. War: Treasury complex (TLT, GOVT, TIP) becomes the primary information sender
5. **COVID produced complete cluster dissolution** (CMI=1.0) but recovered faster than the asymmetric geopolitical shocks

## Repository Structure

```
Asset Cluster Migration/
├── config/
│   ├── universe.yaml          # 43-ETF universe definition
│   └── settings.yaml          # API and pipeline settings
├── src/
│   ├── data/
│   │   ├── fmp_client.py      # Async FMP API client (rate-limited, cached)
│   │   ├── ingestion.py       # Data fetching orchestrator
│   │   ├── cleaning.py        # Alignment and forward-fill
│   │   └── universe.py        # Universe config loader
│   ├── features/
│   │   ├── returns.py         # Log/simple/excess returns
│   │   ├── similarity.py      # 5 similarity measures (shrinkage, dCor, tail dep, etc.)
│   │   └── lead_lag.py        # Transfer entropy (KSG), Granger causality, info flow
│   ├── graphs/
│   │   ├── construction.py    # Threshold graph, MST, multilayer
│   │   ├── filtering.py       # PMFG (Planar Maximally Filtered Graph)
│   │   └── topology.py        # Laplacian eigenvalues, spectral distance
│   ├── clustering/
│   │   ├── community.py       # Leiden, spectral, consensus
│   │   ├── multiplex.py       # Multiplex consensus + layer agreement
│   │   └── kmeans.py          # K-Means baseline comparison engine
│   ├── migration/
│   │   ├── metrics.py         # CMI, AMF, CPS, TDS (novel metrics)
│   │   └── tracking.py        # Migration path tracking, flow matrices
│   ├── regimes/
│   │   ├── hmm.py             # Hidden Markov Model regime detection
│   │   ├── changepoint.py     # PELT changepoint detection
│   │   └── validation.py      # OOS regime validation (TimeSeriesSplit)
│   ├── robustness/            # Phase 4: Statistical robustness framework
│   │   ├── walk_forward.py    # Walk-forward validation (train/test splits)
│   │   ├── bootstrap.py       # Block bootstrap CIs (Politis & Romano 1992)
│   │   ├── sensitivity.py     # Hyperparameter sensitivity sweeps
│   │   ├── multiple_testing.py # Bonferroni, BH-FDR, Storey q-value
│   │   └── surrogate_testing.py # Surrogate data + power analysis
│   └── pipeline/
│       └── orchestrator.py    # Full pipeline orchestrator
├── outputs/
│   ├── final_report.pdf       # Complete research report (24 figures, glossary, disclaimers)
│   └── figures/               # All 24 publication-quality figures
├── data/
│   ├── raw/                   # Cached API responses (gitignored)
│   └── processed/             # Parquet files: returns, correlations, assignments, TE matrices
├── CHANGELOG.md               # Version history (patch notes)
├── Makefile                   # Pipeline automation
└── pyproject.toml             # Dependencies
```

## Asset Universe (91 ETFs)

| Category | Tickers | Count |
|----------|---------|-------|
| US Equity & Value | SPY, QQQ, IWM, DIA, SCHD, VTV | 6 |
| US Sectors | XLE, XLF, XLV, XLU, XLI, XLK, XLP, XLRE, RSPN | 9 |
| International | EFA, EEM, FXI, EWZ, EWJ, VGK, CQQQ, VYMI | 8 |
| Country ETFs | EIS, INDA, EIDO, GREK, EWI, EWN, EWG, EWU, EWW, COLO, ECH, ARGT, EWY, VNM, THD, EWS, EWT, EWA | 18 |
| Fixed Income | TLT, IEF, SHY, LQD, HYG, EMB, TIP, GOVT | 8 |
| Commodities | GLD, SLV, GDX, USO, DBA, DBC, PDBC, VNQ, COPX, URA | 10 |
| FX & Volatility | UUP, FXE, FXY, VIXY | 4 |
| Thematic & Defense | ITA, XAR, QTUM, BLOK, DRNZ, AIPO | 6 |
| Global X Thematic | BOTZ, LIT, DRIV, SOCL, CLOU, BUG, AIQ, HERO, PAVE, KRMA, FINX, SNSR, EBIZ, GNOM, DTCR, SHLD | 16 |
| Managed Futures | DBMF, KMLM, CTA, WTMF | 4 |
| Crypto | BITO, IBIT | 2 |

## Quick Start

```bash
# Clone
git clone https://github.com/studyalwaysbro/asset-cluster-migration.git
cd asset-cluster-migration

# Setup
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env with your FMP API key

# Run full pipeline
make run-all

# Or step by step
make fetch-data
make build-features
make run-clustering
make run-migration
make generate-figures
```

## Novel Metrics

- **CMI (Cluster Migration Index)** - Fraction of assets that changed cluster assignment between windows
- **TDS (Topology Deformation Score)** - Composite of Wasserstein degree distance + NMI community divergence + spectral distance
- **Layer Agreement** - Pairwise NMI between Pearson, dCor, and tail-dependence clusterings
- **Net Transfer Entropy** - Directional information flow ranking via KSG estimator
- **Cross-Layer Granger Causality** - Tests whether tail-dependence CMI predicts Pearson CMI

## Report

The full research report is available at [`outputs/final_report.pdf`](outputs/final_report.pdf). It includes:
- Executive summary and glossary of 19 terms
- Layman-language summaries in every section
- COVID-19 and Iran-Israel conflict event studies
- 24 publication-quality figures
- All methods described with plain-English explanations
- Research observations (explicitly NOT investment recommendations)

---

## Roadmap

> See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Phase 3.5: Baseline Comparisons & Validation (v0.2.0 — Completed)

#### 3.5.1 K-Means Baseline
- [x] K-Means clustering on same rolling windows as Leiden (`src/clustering/kmeans.py`)
- [x] Per-window CMI comparison (K-Means vs Leiden)
- [x] Cross-method agreement metrics (ARI, NMI, silhouette)
- [x] Event-window summary table (pre / event / post aggregation)
- [ ] Generate side-by-side figures for paper (run `rolling_kmeans_baseline` on full dataset)

#### 3.5.2 Out-of-Sample Regime Validation
- [x] Forward-chaining TimeSeriesSplit validation (`src/regimes/validation.py`)
- [x] Topology metrics → regime prediction with constrained RF (no overfitting)
- [x] Per-fold accuracy + macro-F1 reporting
- [x] Feature importance ranking (which topology metrics matter most)
- [ ] Run validation on full dataset and document results

#### 3.5.3 Removed
- [x] ~~Supervised Gradient Boosting predictive layer~~ — removed (out of scope for descriptive research; reserved for future real-time forecasting extension)

### Phase 4: Statistical Robustness (v0.4.0 — Completed)

Framework implemented + critical methodological fixes. See [CHANGELOG.md](CHANGELOG.md) for full details.

#### 4.0 Methodological Audit & Fixes
- [x] CMI permutation invariance — Hungarian algorithm for cluster label matching (`migration/metrics.py`)
- [x] TDS component z-score normalization — `TDSNormalizer` class for commensurable combination
- [x] TDS spectral distance — Wasserstein on Laplacian spectra (replaces zero-padded L2)
- [x] CPS bidirectional matching — Hungarian-based instead of greedy best-Jaccard
- [x] MST weight double-counting fix (`graphs/construction.py`)
- [x] Granger Bonferroni across lags + ADF stationarity pre-check (`features/lead_lag.py`)
- [x] PELT penalty fix — proper BIC: `d * log(n)` (`regimes/changepoint.py`)

#### 4.1 Walk-Forward Validation (`src/robustness/walk_forward.py`)
- [x] Split: train on 2019-2022, test on 2023-2024. Re-train on 2019-2024, test on 2025-2026
- [x] Cross-layer Granger causality (tail -> Pearson CMI) OOS replication test
- [x] Topology crystallization pattern replication (restructuring before events)
- [x] Early warning signal detection with false positive rate tracking
- [ ] Run on full dataset and document results

#### 4.2 Bootstrap & Confidence Intervals (`src/robustness/bootstrap.py`)
- [x] Block bootstrap (Politis & Romano 1992) with configurable block size
- [x] Generic `bootstrap_metric()` for any scalar metric CI
- [x] `bootstrap_te_rankings()`: TE leadership stability across 1000 resamples
- [x] `bootstrap_granger_f_stat()`: cross-layer Granger F-stat robustness
- [ ] Run on full dataset and publish confidence bands

#### 4.3 Sensitivity Analysis (`src/robustness/sensitivity.py`)
- [x] Window size sweep: 60, 90, 120, 150, 180, 252 days
- [x] Top-k threshold sweep: 3, 5, 7, 10 edges per node
- [x] Leiden resolution sweep: 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0
- [x] Tail quantile sweep: 0.01, 0.03, 0.05, 0.10
- [x] Automatic stability assessment (ROBUST / MODERATE / SENSITIVE)
- [ ] Run on full dataset and document parameter stability

#### 4.4 Multiple Testing Correction (`src/robustness/multiple_testing.py`)
- [x] Bonferroni correction (FWER control)
- [x] Benjamini-Hochberg FDR
- [x] Storey's q-value (adaptive FDR with pi_0 estimation)
- [x] Aggregate binomial test (more significant pairs than chance?)
- [x] `summarize_corrections()` for publication-ready comparison table
- [ ] Run on full Granger matrix (8,190 pairs at 91 assets) and document survival rate

#### 4.5 Small-Sample Robustness (`src/robustness/surrogate_testing.py`)
- [x] Phase-randomized surrogates (Theiler et al. 1992) — preserves power spectrum
- [x] IAAFT surrogates (Schreiber & Schmitz 1996) — preserves spectrum + distribution
- [x] Surrogate TE significance test (null: TE from autocorrelation alone)
- [x] Stationary block bootstrap (Politis & Romano 1994) — geometric block lengths
- [x] Monte Carlo minimum sample size estimation (power analysis)
- [ ] Run surrogate tests on regime-conditional TE and document results

### Phase 5: Real-Time Extension

#### 5.1 Streaming Pipeline
- [ ] Replace batch FMP fetch with streaming price feed (WebSocket or polling)
- [ ] Incremental rolling window update (append new day, drop oldest)
- [ ] Intraday granularity option (hourly windows for faster signal detection)

#### 5.2 Real-Time Computation
- [ ] Incremental shrinkage correlation (rank-1 update instead of full recompute)
- [ ] Online Leiden with warm-start from previous partition
- [ ] Streaming transfer entropy with exponential decay weighting

#### 5.3 Dashboard & Alerting
- [ ] Web dashboard (Streamlit or Dash): live CMI, TDS, layer agreement
- [ ] Configurable alert thresholds on tail CMI, layer agreement, TE leadership reversals
- [ ] Historical comparison overlay (current window vs. COVID, vs. Epic Fury buildup)
- [ ] Interactive cluster network visualization (D3.js or Plotly)

### Phase 6: Extended Research

- [ ] **Extend to 2008**: Source pre-2010 data for ETFs that existed (SPY, QQQ, TLT, GLD) to include the GFC
- [ ] **Higher-frequency analysis**: Intraday 5-min returns during crisis windows
- [ ] **Cross-market extension**: Sovereign CDS, VIX term structure, yield curve factors
- [ ] **Causal discovery**: PCMCI+ or DYNOTEARS for full causal graph learning
- [ ] **Geopolitical NLP layer**: GDELT or news embeddings as an additional similarity layer
- [ ] **Crypto deep-dive**: Individual tokens (BTC, ETH, SOL) + DeFi indices
- [ ] **Alternative clustering**: Infomap, Stochastic Block Models, compare to Leiden
- [ ] **Publication**: Target Journal of Financial Economics, Review of Financial Studies, or Journal of Portfolio Management

---

## Research Workflow

The mutable logical workflow:

```
1. FOUNDATION (Completed)
   ├── Multi-asset universe construction (91 ETFs, 7 years)
   ├── Rolling-window similarity computation (3 layers)
   ├── Community detection + migration tracking (CMI, TDS, AMF)
   └── Baseline event studies (COVID, Iran-Israel)

2. MULTI-LAYER ANALYSIS (Completed)
   ├── Distance correlation + tail dependence layers
   ├── Multiplex consensus clustering
   ├── Layer agreement as a meta-signal
   └── Cross-layer divergence analysis

3. INFORMATION FLOW (Completed)
   ├── Transfer entropy (KSG estimator)
   ├── Granger causality network
   ├── Regime-conditional leadership reversal
   └── Cross-layer Granger causality (key discovery)

4. STATISTICAL ROBUSTNESS (Framework Complete — v0.4.0)
   ├── Methodological audit: fixed CMI permutation invariance, TDS scaling, Granger corrections
   ├── Walk-forward validation (train 2019-2022/test 2023-2024, expand + retest)
   ├── Block bootstrap confidence intervals (Politis & Romano 1992)
   ├── Sensitivity sweeps: window size, top-k, resolution, tail quantile
   ├── Multiple testing correction: Bonferroni, BH-FDR, Storey q-value
   ├── Surrogate data testing: phase-randomized + IAAFT null distributions
   └── Monte Carlo power analysis for minimum sample size estimation

5. REAL-TIME EXTENSION
   ├── Streaming data pipeline + incremental rolling window
   ├── Dashboard (Streamlit/Dash): live CMI, TDS, layer agreement
   ├── Alert system on tail CMI, TE leadership reversals
   └── Live validation against emerging events

6. EXTENDED RESEARCH
   ├── Extend to 2008 (GFC) with available ETFs
   ├── Intraday 5-min analysis during crisis windows
   ├── Causal discovery (PCMCI+, DYNOTEARS)
   ├── Geopolitical NLP layer (GDELT, news embeddings)
   └── Crypto deep-dive (BTC, ETH, SOL, DeFi indices)

7. PUBLICATION
   ├── Working paper with full methodology
   ├── Replication package (this repository)
   ├── Conference presentations (AFA, EFA, INFORMS)
   └── Journal submission (JFE, RFS, JPM)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work in academic research, please cite:

```bibtex
@misc{tavares2026topology,
  title={Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress},
  author={Tavares, Nicholas and Hanini, Amjad and Dasilva, Brandon},
  year={2026},
  note={Available at: https://github.com/studyalwaysbro/asset-cluster-migration}
}
```

---

> **Reminder:** This project is for educational and research purposes only. It does not constitute investment advice and is not produced in any broker-dealer or investment advisory capacity. See the full disclaimer in the [research report](outputs/final_report.pdf).
