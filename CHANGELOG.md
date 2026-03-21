# Changelog

All notable changes to the Asset Cluster Migration project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Versioning convention:**
> - **Major (X.0.0):** Breaking API changes or fundamental methodology shifts
> - **Minor (0.X.0):** New modules, metrics, or analysis capabilities
> - **Patch (0.0.X):** Bug fixes, docs, config tweaks, cleanup

---

## [0.5.0] — 2026-03-21

### Added — Phase 4.6: Pipeline Automation & Data Expansion

**Full Pipeline Implementation** (`src/pipeline/steps.py`)
- `step_fetch_data()`: Fetches all universe tickers from FMP API through last trading day
- `step_validate_data()`: Checks staleness, coverage, and missing data percentages
- `step_build_features()`: Cleans/aligns prices, computes log returns, saves parquets
- `step_run_clustering()`: Rolling window Leiden clustering over full history (764 windows)
- `step_run_regimes()`: Fits HMM regime detector (calm/transition/stress) on full returns
- `step_run_migration()`: Computes CMI, AMF, CPS, TDS metrics per window with graph history
- `step_compute_centrality()`: Betweenness, eigenvector, degree, closeness per rolling window
- `step_export_topology()`: Exports 4 parquet files to external cache directory/cache/topology/
- `run_full_pipeline()`: End-to-end orchestrator calling all 8 steps sequentially

**Pipeline Orchestrator** (`src/pipeline/orchestrator.py`)
- CLI via Typer: `run-all`, `run-step <step>`, `export-topology`
- `--skip-fetch` flag for using cached data
- `--skip-export` flag to skip topology export
- Run logging with JSONL summaries to `logs/run_summary.jsonl`
- Persistent file logging with rotation (`logs/pipeline/`)

**Topology Export** (NEW: topology data export for reproducibility)
- `cluster_membership.parquet`: 45,076 rows (date, ticker, cluster_id, days_since_migration, cluster_size)
- `centrality_metrics.parquet`: 45,076 rows (date, ticker, degree, betweenness, eigenvector, closeness)
- `regime_states.parquet`: 3,918 rows (date, regime, regime_probability, days_in_regime)
- `topology_deformation.parquet`: 763 rows (date, tds_score, tds_zscore, layer_agreement)
- Exports to `external cache directory/cache/topology/` for downstream topology analysis

**Event Windows Expansion** (`config/event_windows.yaml`)
- 2 → 8 geopolitical events with full sub-windows and confounders:
  - EU Sovereign Debt Crisis & US Downgrade 2011
  - COVID-19 Market Crash 2020
  - Fed Hawkish Repricing 2022
  - SVB Banking Stress 2023
  - Iran-Israel April 2024 (existing)
  - Iran-Israel October 2024 (existing)
  - Japan Carry Trade Unwind August 2024
  - DeepSeek AI Cost Shock January 2025

**Universe Updates** (`config/universe.yaml`)
- Added 7 ETFs: VTI (Total US Market), XLY (Consumer Discretionary), XLB (Materials), XLC (Communication Services), SOXX (Semiconductors), JEPI (Equity Premium Income), COWZ (Cash Cow Value)
- Completes all 11 GICS sectors (XLY, XLB, XLC were missing)
- Removed DRNZ, AIPO to watch list (< 18 months history, unreliable for 120-day rolling windows)
- Net change: 91 → 96 tickers in universe

**Council Logger** (`src/pipeline/council_logger.py`)
- Training run logging for clustering and HMM steps
- JSONL format for automated analysis

### Changed
- **Data range**: 2019-01-01 → 2010-01-01 (16 years vs 7 years)
  - Captures EU debt crisis 2011, taper tantrum 2013, China deval 2015, vol-pocalypse 2018, COVID 2020, Fed hiking 2022, SVB 2023
  - Newer ETFs auto-excluded from early windows via max_missing_pct filter
- **`methodology.yaml`**: `min_inception_date` updated to 2010-01-01
- **`src/graphs/topology.py`**: Added `_safe_eigenvector_centrality()` fallback for disconnected graphs
  - Computes per-connected-component eigenvector centrality with scale normalization
  - Prevents `AmbiguousSolution` crash on sparse rolling windows
- **`src/pipeline/orchestrator.py`**: Skeleton → fully wired step dispatch with run logging

### Fixed
- Eigenvector centrality crash on disconnected graphs (`AmbiguousSolution` exception)
- Empty cluster_assignments.parquet when cached data had wrong date range (cache clearing resolves)

### Full Pipeline Run Results (2010-2026)
- 59 assets survived cleaning (from 96 universe)
- 3,938 trading days
- 764 rolling windows (120-day, 5-day step)
- 6 active clusters in latest window
- 3 HMM regimes: calm (89.7%), transition (5.1%), stress (5.2%)
- Mean CMI: 0.094, Max CMI: 0.441
- Mean TDS: 0.022, Max TDS: 10.195
- Most migration-prone: COLO (0.216), DBA (0.208), USO (0.206)
- Most stable: VTV (0.016), DIA (0.018), RSPN (0.022)
- Information flow leaders: HYG (+0.385 net), SHY (+0.227), EEM (+0.204)

---

## [0.4.0] — 2026-03-11

### Added — Phase 4: Statistical Robustness
- **`src/robustness/` package** — 5 new modules for rigorous statistical validation:
  - `walk_forward.py`: Walk-forward validation (train 2019-2022/test 2023-2024, re-train 2019-2024/test 2025-2026)
    - Tests cross-layer Granger causality replication out-of-sample
    - Tests topology crystallization pattern replication (restructuring before events)
    - Early warning signal detection with false positive rate tracking
    - `WalkForwardResult` / `WalkForwardSummary` dataclasses
  - `bootstrap.py`: Block bootstrap confidence intervals (Politis & Romano 1992)
    - `bootstrap_metric()`: generic block bootstrap CI for any scalar metric
    - `bootstrap_te_rankings()`: stability of TE leadership rankings across 1000 resamples
    - `bootstrap_granger_f_stat()`: robustness of cross-layer Granger F-statistic (p=0.041)
    - `BootstrapCI` / `TEStabilityResult` dataclasses
  - `sensitivity.py`: Systematic hyperparameter sensitivity sweeps
    - Window size: 60, 90, 120, 150, 180, 252 days
    - Top-K threshold: 3, 5, 7, 10 edges per node
    - Leiden resolution: 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0
    - Tail dependence quantile: 0.01, 0.03, 0.05, 0.10
    - Automatic stability assessment (ROBUST / MODERATE / SENSITIVE)
    - `SensitivityPoint` / `SensitivityResult` dataclasses
  - `multiple_testing.py`: Multiple testing correction for pairwise causality
    - Bonferroni correction (FWER control)
    - Benjamini-Hochberg (FDR control)
    - Storey's q-value (adaptive FDR with pi_0 estimation)
    - Aggregate significance test (binomial test: are there more significant pairs than chance?)
    - `summarize_corrections()` for publication-ready comparison table
  - `surrogate_testing.py`: Small-sample robustness via surrogate data
    - Phase-randomized surrogates (Theiler et al. 1992) — preserves power spectrum
    - IAAFT surrogates (Schreiber & Schmitz 1996) — preserves spectrum AND distribution
    - Surrogate TE significance test (null: TE explainable by autocorrelation alone)
    - Stationary block bootstrap (Politis & Romano 1994) — geometric block lengths
    - Monte Carlo power analysis for minimum sample size estimation
    - `SurrogateTestResult` / `MinSampleResult` dataclasses

### Fixed — Critical Methodological Issues
- **CMI permutation invariance** (`migration/metrics.py`): CMI previously counted label permutations as real migrations. Now uses Hungarian algorithm (scipy `linear_sum_assignment`) to find optimal cluster-to-cluster matching before computing migration. Label swap {0,1,2} → {2,0,1} now correctly yields CMI=0.
- **TDS component incommensurability** (`migration/metrics.py`): Wasserstein degree distance, NMI distance, and spectral distance have different natural scales. Added `TDSNormalizer` class for running z-score normalization of components before weighted combination.
- **TDS spectral distance** (`migration/metrics.py`): Replaced zero-padded L2 norm (destroys spectral gap structure) with Wasserstein distance on Laplacian spectra (handles different-sized graphs naturally). Normalized by max eigenvalue bound (2.0 for normalized Laplacian).
- **CPS bidirectional matching** (`migration/metrics.py`): Cluster persistence score now uses Hungarian matching instead of naive greedy best-Jaccard, avoiding many-to-one matching bugs that miss cluster fragmentation.
- **AMF with Hungarian matching** (`migration/metrics.py`): Asset migration frequency now uses per-step Hungarian matching for consistency.
- **MST weight double-counting** (`graphs/construction.py`): `build_mst()` was summing upper and lower triangle weights (doubling edge weights). Fixed to take max of symmetric entries.
- **Granger causality multiple testing** (`features/lead_lag.py`): Previously took min p-value across lags (cherry-picking). Now applies Bonferroni correction across lags before selecting best.
- **Granger stationarity pre-check** (`features/lead_lag.py`): Added ADF (Augmented Dickey-Fuller) stationarity test. Non-stationary pairs flagged with p=1.0 and logged as warning.
- **PELT penalty formula** (`regimes/changepoint.py`): Was using ad-hoc `log(n) * var(x)`. Fixed to proper BIC penalty: `d * log(n)` where d=2 (mean + variance parameters per segment).

### Changed
- `migration/metrics.py`: `topology_deformation_score()` now accepts optional `TDSNormalizer` argument
- `features/lead_lag.py`: `granger_causality_matrix()` now accepts `check_stationarity` flag (default True)
- Updated test suite: 15 → 32 tests (13 new robustness tests, 4 new migration tests replacing 3 old ones)

---

## [0.3.0] — 2026-03-11

### Added
- **91-ETF universe** (up from 42) — 49 new tickers across 5 new groups:
  - `country_etfs` (18): EIS, INDA, EIDO, GREK, EWI, EWN, EWG, EWU, EWW, COLO, ECH, ARGT, EWY, VNM, THD, EWS, EWT, EWA
  - `global_x_thematic` (16): BOTZ, LIT, DRIV, SOCL, CLOU, BUG, AIQ, HERO, PAVE, KRMA, FINX, SNSR, EBIZ, GNOM, DTCR, SHLD
  - `managed_futures` (4): DBMF, KMLM, CTA, WTMF
  - `dividend_value` (3): SCHD, VTV, VYMI
  - Extended `thematic` (+6): CQQQ, RSPN, DRNZ, AIPO, QTUM, BLOK
  - Extended `commodities` (+2): COPX, URA
  - Extended `international` (+1): CQQQ
- **Live data extension**: FMP client now auto-extends to today via real-time quote endpoint
  - `get_quote()` method for real-time/latest price
  - `_maybe_extend_today()` appends current-day row using close as adjClose fallback
  - Cache staleness check: stale entries (>1 business day old) automatically re-fetched
  - `_last_business_day()` and `_is_us_market_open()` helpers
- **Interactive Plotly visualizations** — all 6 stub modules implemented:
  - `networks.py`: spring-layout cluster network (nodes sized by centrality, colored by cluster, hover details)
  - `migration.py`: CMI comparison line chart (K-Means vs Leiden with event shading) + Sankey flow diagram
  - `heatmaps.py`: correlation heatmap with hierarchical clustering order + category color bar
  - `timeseries.py`: dual-subplot metric timeseries with regime shading + event markers
  - `centrality.py`: top-N highlighted evolution (all assets as gray background, top movers colored)
  - `regimes.py`: regime timeline with colored background bands + metric overlays
  - All output interactive HTML files (zoom, pan, hover, legend toggle)

### Changed
- `FMPClient._check_cache()` now accepts `max_staleness_days` parameter — stale caches are automatically invalidated
- `FMPClient.get_historical_prices()` accepts `extend_to_today` flag (default True)
- `FMPClient.batch_fetch()` passes `extend_to_today` through to individual fetches
- README updated: 91-ETF universe table, overview text
- Short-history ETFs (DRNZ, AIPO, DTCR, SHLD, etc.) have `inception` dates in config — auto-excluded by cleaner if >5% missing

---

## [0.2.0] — 2026-03-11

### Added
- `src/clustering/kmeans.py` — K-Means baseline comparison module
  - `kmeans_communities()`: single-window K-Means clustering (same `dict[str, int]` output as Leiden)
  - `rolling_kmeans_baseline()`: runs K-Means across identical rolling windows as Leiden, computing per-window CMI, ARI, NMI, silhouette
  - `baseline_event_summary()`: aggregates comparison into pre/event/post phases for publication tables
- `src/regimes/validation.py` — Out-of-sample regime validation module
  - `ValidationResult` dataclass with per-fold accuracy, macro-F1, classification report, feature importances
  - `align_features_and_target()`: proper forward-shift alignment (today's topology → tomorrow's regime)
  - `validate_regime_detection()`: forward-chaining `TimeSeriesSplit` with constrained Random Forest (max_depth=5, min_samples_leaf=5)
- `CHANGELOG.md` — this file

### Changed
- Updated README.md roadmap: checked off K-Means baseline and OOS validation tasks

### Removed
- `src/regimes/predictive.py` — Supervised Gradient Boosting predictive layer (scope creep; this project is descriptive/diagnostic, not predictive. Supervised forecasting reserved for future real-time extension if backtest to 2008 is pursued)

### Notes
- The trainee's original `kmeans.py` was a single function with no comparison logic — rewritten as a full rolling-window apples-to-apples baseline engine
- The trainee's original `validation.py` used a raw RF with no structure — rewritten with `ValidationResult` dataclass, macro-F1, feature importances, and proper docstrings clarifying this is a validation exercise not a forecasting model
- The trainee's `predictive.py` trained a GBM on regime labels with no train/test split and only reported training accuracy — removed entirely

---

## [0.1.2] — 2026-03-11 *(remote only — trainee push, superseded by 0.2.0)*

### Added *(reverted in 0.2.0)*
- `src/clustering/kmeans.py` — basic K-Means function (no comparison logic)
- `src/regimes/validation.py` — basic TimeSeriesSplit + RandomForest (no structure)
- `src/regimes/predictive.py` — GradientBoosting regime predictor (no OOS evaluation)

### Notes
- Commits `60047c7a` and `dc5401f8` by trainee
- All three files replaced or removed in 0.2.0

---

## [0.1.1] — 2026-03-09

### Changed
- README.md: removed PhD-level phrasing, added author names, updated citation URLs
- README.md: added statistical robustness phase to roadmap (Phase 4), restructured research workflow diagram
- RESEARCH_DESIGN.md: added author names and domino cascade section

### Removed
- CONTRIBUTING.md (consolidated contributor info into README)

### Notes
- Commits `43e1c12` through `83f220c`
- Three additional README edits on remote (`1c8409c6`, `04c8c93f`, `ca084385`) — minor formatting

---

## [0.1.0] — 2026-03-08

### Added — Initial Release
- **Data pipeline** (`src/data/`)
  - `fmp_client.py`: async FMP API client with token-bucket rate limiting, bandwidth tracking, parquet caching
  - `ingestion.py`: batch universe fetcher with retry logic
  - `cleaning.py`: alignment, forward-fill (max 5 days), >5% missing exclusion
  - `universe.py`: YAML-driven universe loader
  - `cache.py`: SHA256-keyed parquet cache layer

- **Feature computation** (`src/features/`)
  - `returns.py`: log, simple, excess returns + winsorization
  - `similarity.py`: 5 similarity layers — Ledoit-Wolf shrinkage, Spearman, distance correlation (dcor), mutual information (KSG), tail dependence (empirical co-exceedance)
  - `rolling.py`: `RollingWindowEngine` with configurable window sizes [60, 120, 252], step size 5
  - `lead_lag.py`: transfer entropy (KSG k-NN), Granger causality, cross-correlation, information flow ranking
  - `distribution.py`: pairwise Wasserstein distance

- **Graph construction** (`src/graphs/`)
  - `construction.py`: threshold graph, MST, multilayer top-k sparsification, angular/abs distance conversion
  - `filtering.py`: PMFG (Planar Maximally Filtered Graph)
  - `topology.py`: centrality metrics (degree, betweenness, eigenvector, closeness), modularity, Laplacian eigenvalues
  - `multilayer.py`: adjacency tensor [layer × i × j]

- **Clustering** (`src/clustering/`)
  - `community.py`: Leiden (igraph + leidenalg), spectral (eigengap heuristic), consensus (multi-resolution)
  - `multiplex.py`: multiplex consensus clustering, NMI-based layer agreement matrix
  - `temporal.py`: cluster evolution tracking (long-form DataFrame)

- **Novel migration metrics** (`src/migration/`)
  - `metrics.py`: CMI (cluster migration index), AMF (asset migration frequency), CPS (cluster persistence score), TDS (topology deformation score — Wasserstein + NMI + spectral composite), `MigrationSnapshot` dataclass
  - `bridges.py`: bridge score = betweenness × AMF × (1 − clustering_coeff)
  - `tracking.py`: migration path tracking, flow matrices, dominant direction analysis

- **Regime detection** (`src/regimes/`)
  - `hmm.py`: `MarketRegimeDetector` — Gaussian HMM, 3 regimes, auto-ordered by volatility (calm → transition → stress)
  - `changepoint.py`: PELT algorithm (ruptures) with BIC-based penalty selection
  - `segmentation.py`: regime feature builder (realized vol, mean correlation, dispersion)

- **Event study framework** (`src/event_study/`)
  - `windows.py`: `EventWindow` dataclass with pre/event/post slicing, loaded from `config/event_windows.yaml`
  - `analysis.py`: `EventStudyAnalyzer` comparing topology across event phases

- **Pipeline** (`src/pipeline/`)
  - `orchestrator.py`: Typer CLI with 10-step pipeline, Rich logging

- **Configuration** (`config/`)
  - `universe.yaml`: 43-ETF universe across 14 asset categories
  - `methodology.yaml`: all hyperparameters (windows, layers, clustering, regimes, TDS weights, seeds)
  - `event_windows.yaml`: Iran-Israel April 2024 (5 sub-windows) + October 2024 (4 sub-windows)
  - `settings.yaml`: API and pipeline settings

- **Infrastructure**
  - `pyproject.toml`: hatchling build, all dependencies, dev extras (pytest, ruff, mypy, pre-commit)
  - `Makefile`: pipeline automation targets
  - `src/constants.py`: type aliases (`CommunityAssignment`, `LayerDict`, etc.) and enums
  - `src/config.py`: YAML loader, API key management, project root resolution
  - 15 unit tests across 6 test files

- **Documentation**
  - `README.md`: full project overview, architecture, asset universe, quick start, novel metrics, 7-phase roadmap
  - `RESEARCH_DESIGN.md`: detailed methodology, event windows, phased roadmap
  - `outputs/final_report.pdf`: 24-figure research report with glossary and disclaimers

- **Data artifacts** (`data/processed/`)
  - 40 assets × 1,804 days (2019-01-02 to 2026-03-06)
  - Cleaned prices, log returns, correlation matrices (parquet)
  - Static baseline: 4 Leiden clusters (domestic equity, bonds/safe-haven, intl equity, commodities)
  - Top centrality: DIA, SPY (degree); VIXY, USO, UUP (betweenness)

### Notes
- Commit `fb2dad4c` — initial release
- IBIT and BITO excluded from baseline (>5% missing data, short history)
- All random seeds fixed at 42 for reproducibility
