# Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress

> **Disclaimer:** This document is produced solely for educational and academic research purposes. Nothing herein constitutes investment advice, a solicitation, or a recommendation to buy, sell, or hold any security or financial instrument. The authors are presenting independent academic research on asset clustering methodologies. Past performance and historical patterns do not guarantee future results.

## Authors

- **Nicholas Tavares**
- **Amjad Hanini**
- **Brandon Dasilva**

---

## 1. Research Title

**Dynamic Multi-Layer Asset Topology and Cluster Migration Under Geopolitical Stress: A Regime-Aware Framework for Detecting Structural Breaks in Cross-Asset Dependency Networks**

---

## 2. Abstract

We develop a dynamic multi-layer network framework for characterizing the evolving topology of global cross-asset relationships and detecting structural cluster migration during geopolitical stress episodes. Unlike static correlation-based clustering, our approach constructs parallel dependency layers — linear (shrinkage correlation), nonlinear (distance correlation), and tail (empirical co-exceedance) — updated over rolling windows, and applies Leiden community detection with temporal persistence tracking to identify when asset clusters form, fragment, merge, or undergo lasting regime shifts versus transient panic co-movement. We introduce novel metrics: a permutation-invariant Cluster Migration Index (CMI) using Hungarian matching to distinguish genuine structural migration from label noise, and a normalized Topology Deformation Score (TDS) combining Wasserstein degree distance, NMI community divergence, and Wasserstein spectral distance with z-score normalization for commensurable combination. Phase 3 adds directional information flow via KSG transfer entropy and Granger causality networks with Bonferroni correction and ADF stationarity pre-checks, revealing regime-conditional leadership reversals (calm: credit/real estate lead; war: Treasury complex leads). Phase 4 provides a comprehensive statistical robustness framework: walk-forward validation, block bootstrap confidence intervals, hyperparameter sensitivity sweeps, multiple testing correction (Bonferroni/BH-FDR/Storey q-value), and surrogate data testing. We validate on a diversified 96-ETF universe spanning equities, fixed income, commodities, currencies, crypto, managed futures, thematics, and 18 country-specific funds from January 2010 through the most recent trading day, spanning 16 years of regime diversity including the European sovereign debt crisis, taper tantrum, COVID-19 crash, Fed hiking cycle, SVB banking stress, and multiple geopolitical shocks, using the Iran-Israel conflict cycle (2024-2025) and COVID-19 as primary event studies.

---

## 3. Novelty Statement

The primary contribution is treating cluster migration as a first-class, measurable signal rather than a byproduct of rolling correlation analysis. We unify:

1. **Multi-layer graph construction** — simultaneously capturing linear, nonlinear, and tail dependency with cross-layer agreement as a meta-signal
2. **Permutation-invariant migration tracking** — Hungarian algorithm for optimal cluster label matching eliminates false positives from Leiden's arbitrary relabeling
3. **Normalized topology deformation** — z-scored TDS components ensure commensurable combination regardless of natural scale differences
4. **Cross-layer causality** — demonstrating that tail-dependence CMI Granger-causes Pearson CMI (p=0.041), providing a potential early warning signal
5. **Regime-conditional information flow** — transfer entropy reveals leadership reversals during geopolitical stress that are invisible to static methods
6. **Rigorous robustness framework** — walk-forward OOS validation, block bootstrap CIs, surrogate null distributions, and multiple testing correction across 8,000+ pairwise tests

---

## 4. Phased Architecture

### Phase 1: Single-Layer Topology ✅

| Component | Method | Status |
|-----------|--------|--------|
| Data pipeline | Async FMP client, parquet caching, alignment | Complete |
| Similarity | Ledoit-Wolf shrinkage correlation | Complete |
| Graph | Threshold (top-K), MST, PMFG | Complete |
| Community detection | Leiden (igraph + leidenalg) | Complete |
| Migration metrics | CMI, AMF, CPS, TDS (with Hungarian matching) | Complete |
| Regime detection | Gaussian HMM (3 states), PELT changepoints | Complete |

### Phase 2: Multi-Layer Analysis ✅

| Component | Method | Status |
|-----------|--------|--------|
| Distance correlation | dcor library, rolling windows | Complete |
| Tail dependence | Empirical co-exceedance at configurable quantiles | Complete |
| Multiplex clustering | Consensus across layers | Complete |
| Layer agreement | Pairwise NMI between Pearson/dCor/tail clusterings | Complete |

### Phase 3: Directional Information Flow ✅

| Component | Method | Status |
|-----------|--------|--------|
| Transfer entropy | KSG k-NN estimator | Complete |
| Granger causality | With Bonferroni across lags + ADF pre-check | Complete |
| Lead-lag networks | Directed graph from TE rankings | Complete |
| Cross-layer Granger | Tail-dep CMI → Pearson CMI causality test | Complete |

### Phase 3.5: Baseline Comparisons ✅

| Component | Method | Status |
|-----------|--------|--------|
| K-Means baseline | Rolling window comparison engine | Complete |
| Comparison metrics | Per-window ARI, NMI, silhouette, CMI | Complete |
| OOS regime validation | TimeSeriesSplit + constrained RF | Complete |

### Phase 4: Statistical Robustness ✅ (Framework)

| Component | Method | Status |
|-----------|--------|--------|
| Walk-forward | Train 2010-2022/test 2023-2024, expand + retest | Framework complete |
| Bootstrap CIs | Block bootstrap (Politis & Romano 1992) | Framework complete |
| Sensitivity | Window, top-k, resolution, tail quantile sweeps | Framework complete |
| Multiple testing | Bonferroni, BH-FDR, Storey q-value | Framework complete |
| Surrogate testing | Phase-randomized, IAAFT, power analysis | Framework complete |
| **Run on full dataset** | All 5 modules on 96-ETF data | **Pending** |

### Phase 5: Pipeline Automation & Data Expansion ✅

| Component | Method | Status |
|-----------|--------|--------|
| Full pipeline | 8-step CLI orchestrator (Typer) | Complete |
| Data expansion | 2019 → 2010 start date (16 years) | Complete |
| Universe | +7 ETFs (VTI, XLY, XLB, XLC, SOXX, JEPI, COWZ), -2 (DRNZ, AIPO) | Complete |
| Event windows | 2 → 8 events (EU 2011, COVID, Fed 2022, SVB, Japan carry, DeepSeek) | Complete |
| Topology export | 4 parquets for downstream reproducibility and analysis | Complete |
| Centrality fix | Disconnected graph eigenvector centrality fallback | Complete |

### Phase 6: Real-Time Extension (Future)

- Streaming data pipeline + incremental rolling window
- Dashboard (Streamlit/Dash): live CMI, TDS, layer agreement
- Alert system on tail CMI, TE leadership reversals
- Live validation against emerging events

### Phase 7: Extended Research (Future)

- Extend to 2008 GFC with available ETFs
- Intraday 5-min returns during crisis windows
- Causal discovery (PCMCI+, DYNOTEARS)
- Geopolitical NLP layer (GDELT, news embeddings)
- Crypto deep-dive (BTC, ETH, SOL, DeFi indices)

---

## 5. Asset Universe (96 ETFs)

### Selection Criteria
- Liquid ETFs with sufficient history (inception < 2020 preferred)
- Cross-asset coverage: equities, fixed income, commodities, FX, alternatives, crypto, thematics
- Assets known to behave differently during geopolitical stress
- Reproducible via FMP stable API
- Short-history ETFs (inception dates noted) auto-excluded by cleaner if >5% missing

### US Equity & Value (6)
| Ticker | Name | Rationale |
|--------|------|-----------|
| SPY | S&P 500 | Core U.S. large cap |
| QQQ | Nasdaq 100 | Tech/growth benchmark |
| IWM | Russell 2000 | Small cap / domestic cyclical |
| DIA | Dow Jones Industrial | Value/industrial tilt |
| SCHD | US Dividend Equity | Dividend/value factor |
| VTV | US Large-Cap Value | Value tilt |

### US Sectors (9)
| Ticker | Name | Rationale |
|--------|------|-----------|
| XLE | Energy | Direct oil/geopolitical exposure |
| XLF | Financials | Rate/risk sensitivity |
| XLV | Health Care | Defensive sector |
| XLU | Utilities | Defensive / rate sensitive |
| XLI | Industrials | Cyclical / defense overlap |
| XLK | Technology | Growth / momentum proxy |
| XLP | Consumer Staples | Defensive consumer |
| XLRE | Real Estate | Rate sensitive / alternative |
| RSPN | Equal Weight Industrials | Alternative weighting |

### International Equities (8)
| Ticker | Name | Rationale |
|--------|------|-----------|
| EFA | MSCI EAFE | Developed international |
| EEM | MSCI EM | EM risk proxy |
| FXI | China Large Cap | China-specific risk |
| EWZ | Brazil | LatAm / commodity EM |
| EWJ | Japan | Safe haven / carry proxy |
| VGK | FTSE Europe | European exposure |
| CQQQ | China Technology | Tech / EM overlap |
| VYMI | Intl High Dividend | International value |

### Country ETFs (18)
| Ticker | Name | Rationale |
|--------|------|-----------|
| EIS | Israel | Direct geopolitical exposure |
| INDA | India | Large EM, distinct cycle |
| EIDO | Indonesia | SE Asian EM |
| GREK | Greece | Peripheral European |
| EWI | Italy | Eurozone periphery |
| EWN | Netherlands | Core Eurozone / tech |
| EWG | Germany | Eurozone anchor |
| EWU | United Kingdom | Post-Brexit dynamics |
| EWW | Mexico | LatAm / nearshoring |
| COLO | Colombia | Frontier EM / commodity |
| ECH | Chile | Commodity EM (copper) |
| ARGT | Argentina | Frontier volatility |
| EWY | South Korea | Asian tech / semi exposure |
| VNM | Vietnam | Frontier manufacturing |
| THD | Thailand | SE Asian tourism/trade |
| EWS | Singapore | Asian financial hub |
| EWT | Taiwan | Semiconductor / geopolitical |
| EWA | Australia | Commodity DM |

### Fixed Income (8)
| Ticker | Name | Rationale |
|--------|------|-----------|
| TLT | 20+ Yr Treasury | Duration / flight-to-safety |
| IEF | 7-10 Yr Treasury | Intermediate duration |
| SHY | 1-3 Yr Treasury | Short duration / cash proxy |
| LQD | IG Corp Bonds | Credit risk |
| HYG | High Yield Corp | Risk-on credit |
| EMB | EM USD Bonds | EM credit / geopolitical |
| TIP | TIPS | Inflation protection |
| GOVT | US Treasury Bond | Broad treasury exposure |

### Commodities (10)
| Ticker | Name | Rationale |
|--------|------|-----------|
| GLD | Gold | Safe haven / geopolitical hedge |
| SLV | Silver | Precious metals / industrial |
| GDX | Gold Miners | Leveraged gold proxy |
| USO | Crude Oil | Direct geopolitical exposure |
| DBA | Agriculture | Food/commodity inflation |
| DBC | Broad Commodities | Commodity complex |
| PDBC | Optimum Yield | Commodity alternative |
| VNQ | Real Estate REIT | Real assets / rate sensitivity |
| COPX | Copper Miners | Industrial metals / green transition |
| URA | Uranium | Nuclear / energy transition |

### FX & Volatility (4)
| Ticker | Name | Rationale |
|--------|------|-----------|
| UUP | Dollar Bull | USD strength proxy |
| FXE | Euro Currency | EUR/USD proxy |
| FXY | Japanese Yen | JPY safe haven |
| VIXY | VIX Short-Term | Volatility / fear gauge |

### Thematic & Defense (6)
| Ticker | Name | Rationale |
|--------|------|-----------|
| ITA | Aerospace & Defense | Direct defense exposure |
| XAR | S&P Aerospace Defense | Alternative defense ETF |
| QTUM | Quantum Computing & AI | Emerging tech |
| BLOK | Blockchain | Crypto infrastructure |
| DRNZ | Drone Technology | Defense tech (inception Oct 2025) |
| AIPO | AI & Power Infrastructure | AI capex theme (inception Jul 2025) |

### Global X Thematic (16)
| Ticker | Name | Rationale |
|--------|------|-----------|
| BOTZ | Robotics & AI | Automation theme |
| LIT | Lithium & Battery | EV supply chain |
| DRIV | Autonomous & EV | Transportation disruption |
| SOCL | Social Media | Digital consumption |
| CLOU | Cloud Computing | Enterprise SaaS |
| BUG | Cybersecurity | Security spending |
| AIQ | AI & Big Data | AI infrastructure |
| HERO | Video Games & Esports | Digital entertainment |
| PAVE | US Infrastructure | Fiscal spending |
| KRMA | Conscious Companies | ESG factor |
| FINX | FinTech | Financial disruption |
| SNSR | Internet of Things | Connected devices |
| EBIZ | E-Commerce | Digital commerce |
| GNOM | Genomics & Biotech | Healthcare innovation |
| DTCR | Data Center Infrastructure | AI infrastructure (inception Jun 2024) |
| SHLD | Defense Technology | Defense tech (inception Mar 2024) |

### Managed Futures (4)
| Ticker | Name | Rationale |
|--------|------|-----------|
| DBMF | Managed Futures Strategy | Trend-following alternative |
| KMLM | Mount Lucas Managed Futures | Systematic momentum |
| CTA | Simplify Managed Futures | CTA replication |
| WTMF | WisdomTree Managed Futures | Multi-factor managed futures |

### Crypto (2)
| Ticker | Name | Rationale |
|--------|------|-----------|
| IBIT | Bitcoin Trust | Direct BTC exposure (inception Jan 2024) |
| BITO | Bitcoin Strategy | Bitcoin futures proxy (inception Oct 2021) |

### Data Specs
- History: 2010-01-01 to most recent trading day (auto-extended via FMP stable API)
- Frequency: Daily adjusted closes
- Returns: Log returns computed locally with 1% Winsorization
- Short-history handling: ETFs with >5% missing data auto-excluded by `align_and_clean()`
- API: FMP stable endpoint (`historical-price-eod/dividend-adjusted`)
- Estimated ~96 API calls for initial fetch, ~96/day for incremental updates

---

## 6. Methodology Stack

### Layer 1: Similarity / Dependency Measures

| Layer | Measure | Purpose |
|-------|---------|---------|
| Linear | Ledoit-Wolf shrinkage correlation | Regularized linear dependence |
| Rank | Spearman rank correlation | Robust to outliers, nonparametric |
| Nonlinear | Distance correlation (dCor) | Captures nonlinear dependence; zero iff independent |
| Mutual Info | KSG k-NN MI estimator | General nonlinear dependence |
| Tail | Lower-tail dependence (empirical co-exceedance) | Crisis co-movement at configurable quantiles (1%, 3%, 5%, 10%) |
| Transfer entropy | KSG conditional MI at lag-1 | Directional causal proxy |
| Granger causality | VAR with Bonferroni across lags + ADF stationarity pre-check | Directed linear causality |

### Layer 2: Graph Construction
1. Threshold graphs (top-K edges per node, K ∈ {3, 5, 7, 10})
2. MST (structural backbone, fixed: `max()` for symmetric entries, no double-counting)
3. PMFG (richer filtered topology)
4. Multi-layer adjacency tensor A[layer, i, j]

### Layer 3: Community Detection
- **Leiden algorithm** (primary, resolution parameter sweep: 0.3–2.0)
- **K-Means baseline** (rolling window comparison, same windows as Leiden)
- Spectral clustering (eigengap heuristic for k)
- Consensus clustering (100 Leiden runs, consensus matrix)
- Multiplex community detection (cross-layer consensus)

### Layer 4: Temporal Dynamics
- Rolling windows: 60, 90, 120, 150, 180, 252 days, step = 5 days
- PELT change-point detection with proper BIC penalty: `d * log(n)` where d=2
- 3-state Gaussian HMM on [realized_vol, mean_corr, dispersion], auto-ordered by volatility

### Layer 5: Novel Migration Metrics

**CMI — Cluster Migration Index** (permutation-invariant):
```
CMI(t) = (1/N) * sum_i [1 - delta(σ(c_i(t)), c_i(t-1))]
```
where σ is the Hungarian-optimal bijection between current and previous label sets. [0,1], higher = more genuine structural migration. Pure label permutations correctly yield CMI=0.

**AMF — Asset Migration Frequency** (per-step Hungarian matching):
```
AMF_i = (1/T) * sum_t [1 - delta(σ_t(c_i(t)), c_i(t-1))]
```
Identifies boundary/bridge assets with genuinely unstable cluster membership.

**CPS — Cluster Persistence Score** (Hungarian-matched Jaccard):
```
CPS_k(t) = |C_σ(k)(t) ∩ C_k(t-1)| / |C_σ(k)(t) ∪ C_k(t-1)|
```
Uses Hungarian matching for bidirectional cluster alignment, avoiding many-to-one bugs.

**TDS — Topology Deformation Score** (z-score normalized):
```
TDS(t) = α * z(W_degree(t)) + β * z(J_community(t)) + γ * z(S_spectral(t))
```
- W_degree: Wasserstein-1 distance between degree distributions
- J_community: 1 - NMI(communities_t, communities_baseline)
- S_spectral: **Wasserstein distance on Laplacian eigenvalue spectra** (handles different-sized graphs naturally, normalized by max eigenvalue bound 2.0 for normalized Laplacian)
- z(): running z-score normalization via `TDSNormalizer` class for commensurable combination
- Weights calibrated for equal contribution on calibration window

**BS — Bridge Score**:
```
BS_i(t) = betweenness_i(t) * AMF_i(t) * (1 - clustering_coeff_i(t))
```
High = contagion conduit between macro groups.

**Net Transfer Entropy**:
```
NTE(X→Y) = TE(X→Y) - TE(Y→X)
```
Directional information flow ranking via KSG estimator. Identifies regime-conditional leadership.

**Cross-Layer Granger Causality**:
Tests whether tail-dependence CMI Granger-causes Pearson CMI (found: p=0.041), providing a potential early warning signal for structural change.

### Layer 6: Distributional Distance
- Wasserstein-1 on rolling return distributions for regime segmentation
- Wasserstein on Laplacian eigenvalue spectra for topological distance

### Layer 7: Statistical Robustness (Phase 4)

| Method | Reference | Purpose |
|--------|-----------|---------|
| Walk-forward validation | — | OOS replication of cross-layer Granger + topology crystallization |
| Block bootstrap | Politis & Romano (1992) | Confidence intervals for TE rankings, Granger F-stat |
| Stationary bootstrap | Politis & Romano (1994) | Geometric block lengths for autocorrelated data |
| Phase-randomized surrogates | Theiler et al. (1992) | Null: TE from autocorrelation alone (preserves spectrum) |
| IAAFT surrogates | Schreiber & Schmitz (1996) | Null preserving spectrum AND marginal distribution |
| Bonferroni correction | — | FWER control for pairwise Granger (9,120 pairs at 96 assets) |
| Benjamini-Hochberg | Benjamini & Hochberg (1995) | FDR control |
| Storey q-value | Storey (2002) | Adaptive FDR with pi_0 estimation |
| Monte Carlo power analysis | — | Minimum sample size for TE significance |
| Sensitivity sweeps | — | Window size, top-k, resolution, tail quantile stability |

### Baseline Hierarchy
- B0: Static Pearson + hierarchical clustering
- B1: Rolling 120d Pearson + Leiden
- B1-KM: Rolling 120d Pearson + K-Means (baseline comparison)
- M1: Rolling multi-layer (Pearson + dCor + tail) + multiplex Leiden
- M2: M1 + regime segmentation + CMI/TDS (with Hungarian matching + z-score normalization)
- M3: M2 + transfer entropy + Granger causality + cross-layer causality
- M4: M3 + full robustness suite (bootstrap CIs, surrogate tests, sensitivity sweeps, multiple testing correction)

---

## 7. Repository Structure

```
Asset Cluster Migration/
├── config/
│   ├── universe.yaml          # 96-ETF universe definition (13 groups)
│   ├── methodology.yaml       # All hyperparameters (windows, layers, clustering, regimes, TDS, seeds)
│   ├── event_windows.yaml     # Iran-Israel Apr/Oct 2024, COVID, June 2025 events
│   └── settings.yaml          # API and pipeline settings
├── src/
│   ├── data/
│   │   ├── fmp_client.py      # Async FMP stable API client (rate-limited, cached, auto-extend to today)
│   │   ├── ingestion.py       # Batch universe fetcher with retry
│   │   ├── cleaning.py        # Alignment, forward-fill, >5% missing exclusion
│   │   ├── universe.py        # YAML-driven universe loader
│   │   └── cache.py           # SHA256-keyed parquet cache
│   ├── features/
│   │   ├── returns.py         # Log/simple/excess returns + Winsorization
│   │   ├── similarity.py      # 5 similarity measures (shrinkage, dCor, tail dep, etc.)
│   │   ├── rolling.py         # Rolling window engine
│   │   ├── lead_lag.py        # Transfer entropy (KSG), Granger (Bonferroni + ADF), cross-corr
│   │   └── distribution.py    # Pairwise Wasserstein distance
│   ├── graphs/
│   │   ├── construction.py    # Threshold, MST (fixed double-counting), multilayer top-k
│   │   ├── filtering.py       # PMFG
│   │   ├── topology.py        # Centrality, modularity, Laplacian eigenvalues
│   │   └── multilayer.py      # Adjacency tensor [layer x i x j]
│   ├── clustering/
│   │   ├── community.py       # Leiden, spectral, consensus, multiplex
│   │   ├── multiplex.py       # Multiplex consensus + NMI layer agreement
│   │   ├── kmeans.py          # K-Means baseline comparison engine
│   │   └── temporal.py        # Cluster evolution tracking
│   ├── migration/
│   │   ├── metrics.py         # CMI (Hungarian), AMF, CPS, TDS (z-normalized, Wasserstein spectral)
│   │   ├── bridges.py         # Bridge score detection
│   │   └── tracking.py        # Migration flow matrices, dominant direction
│   ├── regimes/
│   │   ├── hmm.py             # Gaussian HMM (3 states, auto-ordered)
│   │   ├── changepoint.py     # PELT with proper BIC penalty (d * log(n))
│   │   ├── validation.py      # OOS TimeSeriesSplit + constrained RF
│   │   └── segmentation.py    # Regime feature builder
│   ├── robustness/            # Phase 4: Statistical robustness framework
│   │   ├── walk_forward.py    # Walk-forward validation (train/test splits)
│   │   ├── bootstrap.py       # Block bootstrap CIs (Politis & Romano 1992)
│   │   ├── sensitivity.py     # Hyperparameter sensitivity sweeps
│   │   ├── multiple_testing.py # Bonferroni, BH-FDR, Storey q-value
│   │   └── surrogate_testing.py # Phase-randomized/IAAFT surrogates + power analysis
│   ├── event_study/
│   │   ├── windows.py         # EventWindow dataclass, YAML loader
│   │   └── analysis.py        # Cross-event topology comparison
│   ├── visualization/         # Interactive Plotly HTML
│   │   ├── networks.py        # Spring-layout cluster network
│   │   ├── migration.py       # CMI comparison + Sankey flows
│   │   ├── heatmaps.py        # Correlation heatmap with clustering order
│   │   ├── timeseries.py      # Metric timeseries with regime shading
│   │   ├── centrality.py      # Top-N centrality evolution
│   │   └── regimes.py         # Regime timeline with metric overlays
│   ├── pipeline/
│   │   └── orchestrator.py    # Typer CLI, 10-step pipeline, Rich logging
│   ├── constants.py           # Type aliases, enums
│   └── config.py              # YAML loader, API key management
├── outputs/
│   ├── final_report.pdf       # Complete research report
│   ├── figures/               # Publication-quality figures
│   └── PHASE_PROGRESS_REPORT.md
├── data/
│   ├── raw/                   # Cached API responses (gitignored)
│   └── processed/             # Parquet: returns, correlations, assignments, TE matrices
├── tests/
│   ├── test_migration_metrics.py  # 7 tests (Hungarian matching)
│   ├── test_robustness.py         # 13 tests (all robustness modules)
│   └── ... (6 additional test files, 32 total tests)
├── CHANGELOG.md
├── Makefile
└── pyproject.toml             # v0.4.0, Python >=3.10
```

---

## 8. Security Plan

1. API keys in `.env` only (never tracked in git)
2. `python-dotenv` loads at runtime; `src/config.py` validates at startup
3. Also supports `FMP_API_KEY` as system environment variable
4. Zero hardcoded API key strings anywhere in codebase
5. `.gitignore`: `.env`, `.env.*`, `data/raw/`, `*.parquet`, `__pycache__/`, `.ipynb_checkpoints/`, `outputs/`
6. Pre-commit hooks: `detect-secrets`, `detect-private-key`, `check-added-large-files`, `nbstripout`
7. CI: trufflehog on all PRs for verified secret detection
8. Policy: All PRs pass secret scan; no API key in notebook output
9. **Branch protection**: GitHub ruleset active — contributors must submit PRs for review before merging to master

---

## 9. Reproducibility Plan

### Data
- `config/universe.yaml`: exact tickers (96), date ranges, source, inception dates for short-history ETFs
- Data manifests with SHA-256 checksums after each fetch
- Processed parquet snapshots version-controlled (< 5MB for Phase 1)
- Larger universes: GitHub Releases for data bundles
- Auto-extension to most recent trading day via FMP stable API

### Environment
- `pyproject.toml` with pinned dependency versions
- Python >= 3.10 required
- Lock file for deterministic installs

### Randomness
- All stochastic methods seeded via `config/methodology.yaml`
- Seeds: clustering=42, hmm=42, consensus=42, bootstrap=42

### Workflow (Makefile)
`fetch-data`, `validate-data`, `build-features`, `run-baseline`, `run-clustering`,
`run-regimes`, `run-event-study`, `run-migration`, `generate-figures`, `generate-report`,
`run-all`, `test`, `lint`, `clean`

---

## 10. Implementation Blueprint

### Core Dependencies
pandas>=2.1, numpy>=1.26, scipy>=1.12, scikit-learn>=1.4, statsmodels>=0.14,
networkx>=3.2, igraph>=0.11, leidenalg>=0.10, hmmlearn>=0.3, ruptures>=1.1,
dcor>=0.7, POT>=0.9, pyarrow>=14.0, httpx>=0.27, tenacity>=8.2,
python-dotenv>=1.0, pyyaml>=6.0, matplotlib>=3.8, seaborn>=0.13, plotly>=5.18,
rich>=13.7, typer>=0.9

### Methodological Rigor

| Concern | Mitigation |
|---------|------------|
| Nonstationarity | Rolling windows + regime segmentation |
| Look-ahead bias | Strictly backward-looking windows |
| Survivorship bias | Inception date validation, auto-exclusion of short-history ETFs |
| Calendar alignment | Forward-fill max 5 days, async close lag check |
| Missing data | Fill 5 days then NaN; >5% missing = exclude |
| Window sensitivity | Sensitivity sweep: 60, 90, 120, 150, 180, 252 days |
| Outlier robustness | Shrinkage, rank corr, Winsorization (1%) |
| Stress correlation breakdown | Tail dependence layer (quantile sweep: 1%, 3%, 5%, 10%) |
| Linear insufficiency | dCor + MI + tail layers |
| Cluster label permutation | Hungarian algorithm matching (CMI, AMF, CPS) |
| TDS component scales | Running z-score normalization (TDSNormalizer) |
| Spectral distance sizing | Wasserstein on spectra (handles different graph sizes) |
| MST weight bias | Fixed double-counting (max, not sum of symmetric entries) |
| Granger lag selection | Bonferroni correction across tested lags |
| Granger on nonstationary data | ADF stationarity pre-check (non-stationary → p=1.0) |
| PELT over-segmentation | Proper BIC penalty: d * log(n), d=2 |
| Multiple testing (9,120 pairs) | Bonferroni, BH-FDR, Storey q-value comparison |
| Small sample TE significance | Phase-randomized + IAAFT surrogate null distributions |
| TE leadership robustness | Block bootstrap CIs (1000 resamples) |
| OOS validity | Walk-forward validation (two expanding windows) |
| Hyperparameter sensitivity | Full sweep + automatic stability assessment |
| Statistical power | Monte Carlo minimum sample size estimation |

---

## Appendix A: Event Studies

### Primary Event: Iran-Israel Conflict Cycle (2024-2025)

#### Operation True Promise I (April 2024)
First-ever direct Iran-Israel military exchange. Israeli strike on Iranian consulate in Damascus (Apr 1) triggered escalation culminating in Iranian launch of ~170 drones, 30+ cruise missiles, and 120+ ballistic missiles (Apr 13-14). Israeli counter-strike on Isfahan (Apr 19). De-escalation followed.

| Window | Dates | Trading Days | Purpose |
|--------|-------|-------------|---------|
| Pre-event calibration | 2024-01-02 to 2024-03-31 | ~60 | Baseline topology |
| Pre-calm | 2024-03-15 to 2024-03-31 | ~12 | Immediate pre-shock baseline |
| Escalation buildup | 2024-04-01 to 2024-04-12 | ~9 | Telegraphed retaliation pricing |
| Acute shock | 2024-04-13 to 2024-04-15 | ~1 (Mon) | First trading day reaction |
| Counter-response | 2024-04-16 to 2024-04-22 | ~5 | Israeli counter-strike, de-escalation |
| Digestion | 2024-04-23 to 2024-05-10 | ~14 | Risk premium fade |
| Normalization | 2024-05-11 to 2024-05-31 | ~15 | Return to macro-driven regime |
| Post-event full | 2024-05-01 to 2024-06-28 | ~42 | Extended post-event |

#### Operation True Promise II (October 2024)
~180 ballistic missiles launched Oct 1. Israeli counter-strike Oct 26 avoided oil infrastructure, triggering ~5-6% oil crash.

| Window | Dates | Trading Days | Purpose |
|--------|-------|-------------|---------|
| Pre-escalation | 2024-09-15 to 2024-09-30 | ~11 | Includes Nasrallah assassination Sep 27 |
| Acute shock | 2024-10-01 to 2024-10-03 | ~3 | Missile barrage reaction |
| Anticipation | 2024-10-04 to 2024-10-25 | ~16 | Waiting for Israeli response |
| Resolution relief | 2024-10-26 to 2024-10-28 | ~1 | Oil crash on restrained response |
| Normalization | 2024-10-29 to 2024-11-15 | ~14 | Risk premium unwind |

#### Operation Epic Fury / Twelve-Day War (2025)
Key finding: Peak topology deformation (TDS=0.176, exceeding COVID) occurred during the BUILDUP, not during strikes. June 2025 events were invisible to Pearson correlation but showed massive restructuring in nonlinear and tail-dependence measures.

### Secondary Event: COVID-19 (March 2020)
Complete cluster dissolution (CMI=1.0) but recovered faster than the asymmetric geopolitical shocks.

---

## Appendix B: Key Findings (Pre-v0.4.0 Corrections)

> These findings were computed before the v0.4.0 methodological fixes (Hungarian matching, TDS normalization, etc.). Results will be recalculated with corrected methodology on the full 91-ETF universe. Qualitative patterns expected to hold; quantitative values may shift.

1. **Markets restructure BEFORE the event** — Peak TDS (0.176) occurred during buildup to Operation Epic Fury, not during strikes
2. **Correlation alone is insufficient** — June 2025 Twelve-Day War invisible to Pearson but massive in dCor/tail layers
3. **Tail-dependence CMI Granger-causes Pearson CMI** (p=0.041) — Crash-structure changes predict normal-correlation changes
4. **Leadership reversal during war** — Calm: credit/real estate lead. War: Treasury complex (TLT, GOVT, TIP) becomes primary information sender
5. **COVID produced complete cluster dissolution** (CMI=1.0) but recovered faster than asymmetric geopolitical shocks

---

## Appendix C: Approach Comparison

| Existing Approach | Limitation | Our Improvement |
|-------------------|-----------|-----------------|
| Static correlation clustering | Ignores time variation | Rolling + regime-conditional |
| Rolling correlation heatmaps | Pairwise, no community structure | Community detection + lifecycle |
| Single-layer network | Misses nonlinear/tail | Multi-layer graph with layer agreement |
| HMM on returns | Asset-level, not topology | HMM on topology state vector |
| Standard event studies | Returns focus | Topology-based event analysis |
| Graph-only methods | No migration tracking | CMI, AMF, CPS, TDS metrics (permutation-invariant) |
| Naive CMI | Counts label swaps as migration | Hungarian matching for true structural change |
| Simple TDS aggregation | Incommensurable components | z-score normalization + Wasserstein spectral |
| Uncorrected Granger | Cherry-picked lags, non-stationary data | Bonferroni + ADF pre-check |
| Single hypothesis testing | Inflated false discovery rate | Bonferroni/BH-FDR/Storey across 9,120 pairs |

---

## Appendix D: Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

| Version | Date | Highlights |
|---------|------|------------|
| v0.1.0 | 2026-03-08 | Initial release: 42 ETFs, full pipeline, 15 tests |
| v0.1.1 | 2026-03-09 | README/docs cleanup |
| v0.2.0 | 2026-03-11 | K-Means baseline, OOS validation, removed predictive.py |
| v0.3.0 | 2026-03-11 | 91-ETF universe, live data extension, Plotly visualizations |
| v0.4.0 | 2026-03-11 | Phase 4 robustness framework, 7 critical methodology fixes, 32 tests |
| v0.5.0 | 2026-03-21 | Phase 5: 8-step CLI orchestrator, 2010 start date (16 years), 96-ETF universe (+7/-2), 8 event windows, topology parquet export, centrality fix |
