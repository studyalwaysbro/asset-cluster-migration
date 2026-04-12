# Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress

> **Disclaimer:** This project is produced solely for educational and academic research purposes. Nothing in this repository constitutes investment advice, a solicitation, or a recommendation to buy, sell, or hold any security or financial instrument. The authors are presenting independent academic research on asset clustering methodologies. Past performance and historical patterns do not guarantee future results.

## Authors

- **Nicholas Tavares**
- **Amjad Hanini**
- **Brandon Dasilva**

## Overview

A quantitative research framework that tracks how multi-asset portfolio structures reorganize during geopolitical crises. We apply graph theory, community detection, information theory, and multi-layer network analysis to a universe of ETFs spanning equities, fixed income, commodities, currencies, crypto, managed futures, thematics, safe havens, agriculture, industrial metals, and country-specific funds.

The pipeline runs automatically via cron, producing daily HTML reports with interactive charts, presentation slide decks, and speaker notes.

### Three-Phase Architecture

| Phase | Focus | Methods |
|-------|-------|---------|
| **Phase 1** | Single-layer topology | Shrinkage correlation, Leiden clustering, CMI/TDS metrics, HMM regime detection |
| **Phase 2** | Multi-layer analysis | Distance correlation, tail dependence, multiplex consensus clustering, layer agreement |
| **Phase 3** | Directional flow | Transfer entropy (KSG), Granger causality, lead-lag networks, cross-layer causality |

### Key Findings

<!-- BEGIN:DYNAMIC pipeline_stats -->
This section is auto-generated. Do not edit manually.
Run `python scripts/generate_readme.py` to refresh.
<!-- END:DYNAMIC pipeline_stats -->

## Latest Pipeline Results

<!-- BEGIN:DYNAMIC latest_results -->
This section is auto-generated. Do not edit manually.
Run `python scripts/generate_readme.py` to refresh.
<!-- END:DYNAMIC latest_results -->

## Repository Structure

```
Asset Cluster Migration/
├── config/
│   ├── universe.yaml          # ETF universe definition
│   ├── settings.yaml          # API and pipeline settings
│   ├── methodology.yaml       # All hyperparameters and seeds
│   └── event_windows.yaml     # Geopolitical event windows
├── src/
│   ├── data/
│   │   ├── fmp_client.py      # Async FMP API client (rate-limited, cached, live extension)
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
│   ├── reports/               # Automated report generation
│   │   ├── daily_report.py    # 18-section HTML daily report (~400 KB)
│   │   ├── charts.py          # Interactive Plotly charts (CMI/TDS, network, heatmap)
│   │   ├── slides.py          # Presentation slide deck generator
│   │   ├── slide_charts.py    # Slide-specific chart generators
│   │   └── speaker_notes.py   # Automated speaker notes from pipeline data
│   └── pipeline/
│       ├── orchestrator.py    # Full pipeline orchestrator (CLI)
│       ├── steps.py           # 9-step pipeline + report/slides generation
│       └── council_logger.py  # Training/council run logging
├── outputs/
│   ├── final_report.pdf       # Complete research report (24 figures, glossary, disclaimers)
│   ├── figures/               # All 24 publication-quality figures
│   └── reports/               # Daily HTML reports, slide decks, speaker notes
├── data/
│   ├── raw/                   # Cached API responses (gitignored)
│   └── processed/             # Parquet files: returns, correlations, assignments, TE matrices
├── logs/
│   ├── pipeline/              # Per-day pipeline execution logs
│   ├── council/               # Council session logs
│   ├── run_summary.jsonl      # Pipeline run metadata
│   └── training_runs.jsonl    # Training step details
├── paper.tex                  # LaTeX research paper source
├── CHANGELOG.md               # Version history (patch notes)
├── Makefile                   # Pipeline automation
└── pyproject.toml             # Dependencies
```

## Asset Universe

<!-- BEGIN:DYNAMIC universe_summary -->
This section is auto-generated. Do not edit manually.
Run `python scripts/generate_readme.py` to refresh.
<!-- END:DYNAMIC universe_summary -->

## Quick Start

```bash
# Clone
git clone https://github.com/studyalwaysbro/asset-cluster-migration.git
cd asset-cluster-migration

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env with your FMP API key

# Run full pipeline (fetches data, clusters, regimes, migration, centrality, exports, report, slides)
python -m src.pipeline.orchestrator run-all

# Run with cached data (skip API fetch)
python -m src.pipeline.orchestrator run-all --skip-fetch

# Run individual steps
python -m src.pipeline.orchestrator run-step fetch-data
python -m src.pipeline.orchestrator run-step run-clustering
python -m src.pipeline.orchestrator run-step export-topology
python -m src.pipeline.orchestrator run-step generate-report

# Export topology to external analysis cache only
python -m src.pipeline.orchestrator export-topology
```

## Pipeline Steps

The automated daily pipeline runs 9 steps sequentially (~2 minutes total):

| Step | Name | Description |
|------|------|-------------|
| 1 | `fetch-data` | Fetch all universe tickers from FMP API |
| 2 | `validate-data` | Check staleness, coverage, missing data |
| 3 | `build-features` | Clean/align prices, compute log returns |
| 4 | `run-clustering` | Rolling window Leiden clustering |
| 5 | `run-regimes` | HMM regime detection (calm/transition/stress) |
| 6 | `run-migration` | CMI, AMF, CPS, TDS metrics per window |
| 7 | `compute-centrality` | Betweenness, eigenvector, degree, closeness |
| 8 | `export-topology` | Parquet files to external cache |
| 9 | `generate-report` | Daily HTML report + slide deck + speaker notes |

## Novel Metrics

- **CMI (Cluster Migration Index)** - Fraction of assets that changed cluster assignment between windows (Hungarian-matched for permutation invariance)
- **TDS (Topology Deformation Score)** - Composite of Wasserstein degree distance + NMI community divergence + spectral distance (z-score normalized)
- **Layer Agreement** - Pairwise NMI between Pearson, dCor, and tail-dependence clusterings
- **Net Transfer Entropy** - Directional information flow ranking via KSG estimator
- **Cross-Layer Granger Causality** - Tests whether tail-dependence CMI predicts Pearson CMI

## Daily Reports

The pipeline generates three output artifacts daily:

1. **Daily Report** (`outputs/reports/acm_daily_report_YYYY-MM-DD.html`) — 18-section HTML report (~400 KB) with interactive Plotly charts, regime analysis, cluster composition, migration metrics, centrality leaders, AMF rankings, automated discussion/interpretation, robustness summary, and data quality metrics
2. **Slide Deck** (`outputs/reports/acm_slides_YYYY-MM-DD.html`) — Presentation-ready HTML slides generated from pipeline data
3. **Speaker Notes** (`outputs/reports/speaker_notes_YYYY-MM-DD.html`) — Companion notes for each slide

A `latest` symlink always points to the most recent report.

## Report

The full research report is available at [`outputs/final_report.pdf`](outputs/final_report.pdf). It includes:
- Executive summary and glossary of 19 terms
- Layman-language summaries in every section
- COVID-19 and Iran-Israel conflict event studies
- 24 publication-quality figures
- All methods described with plain-English explanations
- Research observations (explicitly NOT investment recommendations)

A LaTeX research paper is also available at [`paper.tex`](paper.tex).

---

## Roadmap

> See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Phase 3.5: Baseline Comparisons & Validation (v0.2.0 — Completed)

#### 3.5.1 K-Means Baseline
- [x] K-Means clustering on same rolling windows as Leiden (`src/clustering/kmeans.py`)
- [x] Per-window CMI comparison (K-Means vs Leiden)
- [x] Cross-method agreement metrics (ARI, NMI, silhouette)
- [x] Event-window summary table (pre / event / post aggregation)
- [x] Side-by-side figures in daily report (K-Means vs Leiden CMI comparison)

#### 3.5.2 Out-of-Sample Regime Validation
- [x] Forward-chaining TimeSeriesSplit validation (`src/regimes/validation.py`)
- [x] Topology metrics → regime prediction with constrained RF (no overfitting)
- [x] Per-fold accuracy + macro-F1 reporting
- [x] Feature importance ranking (which topology metrics matter most)
- [x] Run validation on full dataset: 88.9% accuracy, F1 macro 0.313

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
- [x] Run on full dataset (2010-2026) — Granger replication 0% OOS, honest caveat documented

#### 4.2 Bootstrap & Confidence Intervals (`src/robustness/bootstrap.py`)
- [x] Block bootstrap (Politis & Romano 1992) with configurable block size
- [x] Generic `bootstrap_metric()` for any scalar metric CI
- [x] `bootstrap_te_rankings()`: TE leadership stability across 1000 resamples
- [x] `bootstrap_granger_f_stat()`: cross-layer Granger F-stat robustness
- [x] Run on full dataset (2010-2026) — CMI CI [0.105, 0.121], CPS CI [0.770, 0.798]

#### 4.3 Sensitivity Analysis (`src/robustness/sensitivity.py`)
- [x] Window size sweep: 60, 90, 120, 150, 180, 252 days — MODERATE
- [x] Top-k threshold sweep: 3, 5, 7, 10 edges per node — SENSITIVE
- [x] Leiden resolution sweep: 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0 — Mixed
- [x] Tail quantile sweep: 0.01, 0.03, 0.05, 0.10 — SENSITIVE
- [x] Automatic stability assessment (ROBUST / MODERATE / SENSITIVE)
- [x] Run on full dataset (2010-2026)

#### 4.4 Multiple Testing Correction (`src/robustness/multiple_testing.py`)
- [x] Bonferroni correction (FWER control)
- [x] Benjamini-Hochberg FDR
- [x] Storey's q-value (adaptive FDR with pi_0 estimation)
- [x] Aggregate binomial test (more significant pairs than chance?)
- [x] `summarize_corrections()` for publication-ready comparison table
- [x] Run on full dataset (2010-2026)

#### 4.5 Small-Sample Robustness (`src/robustness/surrogate_testing.py`)
- [x] Phase-randomized surrogates (Theiler et al. 1992) — preserves power spectrum
- [x] IAAFT surrogates (Schreiber & Schmitz 1996) — preserves spectrum + distribution
- [x] Surrogate TE significance test (null: TE from autocorrelation alone)
- [x] Stationary block bootstrap (Politis & Romano 1994) — geometric block lengths
- [x] Monte Carlo minimum sample size estimation (power analysis)
- [x] Run on full dataset (2010-2026) — no pair significant at p<0.05 (TE is contemporaneous, not causal)

#### 4.6 Pipeline Automation & Integration (v0.5.0 — Completed)
- [x] Full 9-step pipeline with CLI (fetch, validate, build-features, clustering, regimes, migration, centrality, export-topology, generate-report)
- [x] Topology export to external analysis cache for downstream model consumption
- [x] Disconnected graph handling for eigenvector centrality
- [x] Run logging with JSONL summaries

### Phase 4.7: Reporting & Presentation (v0.6.0 — Completed)

- [x] Daily HTML report generator — 18 sections, ~400 KB, interactive Plotly charts
- [x] CMI/TDS timeseries chart with regime shading
- [x] Interactive cluster network visualization (force-directed Plotly)
- [x] Correlation heatmap with hierarchical clustering
- [x] Automated discussion/interpretation section
- [x] Statistical robustness summary integrated into report
- [x] Presentation slide deck auto-generation from pipeline data
- [x] Speaker notes generator for each slide
- [x] Daily cron automation (3:30 PM ET, Mon-Sat)

### Phase 5: Real-Time Extension

#### 5.0 Data & Coverage Expansion (v0.6.0 — Completed)
- [x] Extended data range to 2010 (was 2019)
- [x] Expanded universe from 96 to 136 ETFs (76 survive cleaning)
- [x] Completed GICS sector coverage (XLY, XLB, XLC, SOXX)
- [x] Full commodity decomposition: precious, energy, agriculture, industrial, broad
- [x] 7 FX currencies, 4 EM/intl bonds, 3 safe havens, 6 agriculture ETFs
- [x] 8 geopolitical event windows (COVID, EU debt crisis 2011, Fed 2022, SVB 2023, Japan carry 2024, Iran-Israel x2, DeepSeek 2025)

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

- [ ] **Market holiday awareness** — `market_calendar.py` to gate pipelines on non-trading days
- [ ] **Extend to 2008**: Partially achieved — now starts 2010. GFC extension requires sourcing pre-inception data for newer ETFs.
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
   ├── Multi-asset universe construction (ETFs across multiple groups, 16+ years)
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

4. STATISTICAL ROBUSTNESS (Completed — Full Suite Executed)
   ├── Methodological audit: fixed CMI permutation invariance, TDS scaling, Granger corrections
   ├── Walk-forward validation (train/test splits)
   ├── Block bootstrap confidence intervals (Politis & Romano 1992)
   ├── Sensitivity sweeps: window size, top-k, resolution, tail quantile
   ├── Multiple testing correction: Bonferroni, BH-FDR, Storey q-value
   ├── Surrogate data testing: phase-randomized + IAAFT null distributions
   ├── Monte Carlo power analysis for minimum sample size estimation
   ├── K-Means baseline comparison (Leiden 40% more stable)
   ├── OOS regime validation (89% accuracy, 5-fold TimeSeriesSplit)
   └── Pipeline automation: 9-step CLI, topology export, run logging

5. REPORTING & PRESENTATION (Completed)
   ├── 18-section daily HTML report with interactive Plotly charts
   ├── Automated discussion and interpretation section
   ├── Presentation slide deck and speaker notes auto-generation
   ├── Daily cron automation (3:30 PM ET)
   └── LaTeX research paper

6. DATA EXPANSION (Completed)
   ├── Data range extended to 2010, GICS sectors completed, 8 event windows
   ├── Universe expanded (76 ETFs survive cleaning)
   ├── Full commodity decomposition (5 sub-groups)
   ├── Expanded FX, EM bonds, safe havens, agriculture, industrial metals
   └── Topology export for downstream ML model integration

7. REAL-TIME EXTENSION
   ├── Streaming data pipeline + incremental rolling window
   ├── Dashboard (Streamlit/Dash): live CMI, TDS, layer agreement
   ├── Alert system on tail CMI, TE leadership reversals
   └── Live validation against emerging events

8. EXTENDED RESEARCH
   ├── Market holiday calendar integration
   ├── Extend to 2008 (GFC) — partially achieved, now starts 2010
   ├── Intraday 5-min analysis during crisis windows
   ├── Causal discovery (PCMCI+, DYNOTEARS)
   ├── Geopolitical NLP layer (GDELT, news embeddings)
   └── Crypto deep-dive (BTC, ETH, SOL, DeFi indices)

9. PUBLICATION
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

> **Reminder:** This project is for educational and research purposes only. It does not constitute investment advice. See the full disclaimer in the [research report](outputs/final_report.pdf).

<!-- BEGIN:DYNAMIC generation_stamp -->
<!-- END:DYNAMIC generation_stamp -->
