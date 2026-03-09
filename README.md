# Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress

> **Disclaimer:** This project is produced solely for educational and academic research purposes. Nothing in this repository constitutes investment advice, a solicitation, or a recommendation to buy, sell, or hold any security or financial instrument. This project is not connected to, endorsed by, or produced in any capacity related to UBS or any broker-dealer. The authors are presenting independent academic research on asset clustering methodologies. Past performance and historical patterns do not guarantee future results.

## Authors

- **Nicholas Tavares** - Lead Researcher
- **Amjad Hanini** - Contributor
- **Brandon DaSilva** - Contributor

## Overview

A quantitative research framework that tracks how multi-asset portfolio structures reorganize during geopolitical crises. We apply graph theory, community detection, information theory, and multi-layer network analysis to a universe of 43 ETFs spanning equities, bonds, commodities, currencies, crypto, and the Israeli market (EIS) from January 2019 through March 2026.

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
│   │   └── multiplex.py       # Multiplex consensus + layer agreement
│   ├── migration/
│   │   ├── metrics.py         # CMI, AMF, CPS, TDS (novel metrics)
│   │   └── tracking.py        # Migration path tracking, flow matrices
│   ├── regimes/
│   │   ├── hmm.py             # Hidden Markov Model regime detection
│   │   └── changepoints.py    # PELT changepoint detection
│   └── pipeline/
│       └── orchestrator.py    # Full pipeline orchestrator
├── outputs/
│   ├── final_report.pdf       # Complete research report (24 figures, glossary, disclaimers)
│   └── figures/               # All 24 publication-quality figures
├── data/
│   ├── raw/                   # Cached API responses (gitignored)
│   └── processed/             # Parquet files: returns, correlations, assignments, TE matrices
├── Makefile                   # Pipeline automation
└── pyproject.toml             # Dependencies
```

## Asset Universe (43 ETFs)

| Category | Tickers | Purpose |
|----------|---------|---------|
| US Equity | SPY, QQQ, IWM, DIA | Core benchmarks |
| US Sectors | XLE, XLF, XLV, XLU, XLI, XLK, XLP, XLRE | Sector rotation |
| International | EFA, EEM, FXI, EWZ, EWJ, VGK, **EIS** | Global + Israel |
| Fixed Income | TLT, IEF, SHY, LQD, HYG, EMB, TIP, GOVT | Duration / safe-haven |
| Commodities | GLD, SLV, GDX, USO, DBA, DBC, PDBC, VNQ | Inflation / geopolitical |
| Currencies | UUP, FXE, FXY | Dollar / risk sentiment |
| Vol & Defense | VIXY, ITA, XAR | Tail risk / defense sector |
| Crypto | **BITO** (Oct 2021+), **IBIT** (Jan 2024+) | Digital assets |

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

## Roadmap: Real-Time Extension

### Phase 4: Live Monitoring System (Proposed)

The natural next step is to make this framework operate in real-time. Below is the proposed architecture:

#### 4.1 Streaming Data Pipeline
- [ ] Replace batch FMP fetch with streaming price feed (WebSocket or polling)
- [ ] Implement incremental rolling window update (append new day, drop oldest)
- [ ] Add intraday granularity option (hourly windows for faster signal detection)
- [ ] Integrate alternative data: news sentiment (NLP), options implied vol surface, CDS spreads

#### 4.2 Real-Time Metric Computation
- [ ] Incremental shrinkage correlation update (rank-1 update instead of full recompute)
- [ ] Online Leiden clustering with warm-start from previous partition
- [ ] Streaming transfer entropy with exponential decay weighting
- [ ] Real-time layer agreement and cross-layer causality monitoring

#### 4.3 Dashboard & Alerting
- [ ] Web dashboard (Streamlit or Dash) showing live CMI, TDS, layer agreement
- [ ] Alert system: configurable thresholds on tail CMI, layer agreement, TE leadership reversals
- [ ] Historical comparison overlay (current vs. COVID, vs. Epic Fury buildup)
- [ ] Interactive cluster network visualization (D3.js or Plotly)

#### 4.4 Backtesting Framework
- [ ] Walk-forward validation of cross-layer Granger causality signal
- [ ] Out-of-sample testing of topology crystallization pattern
- [ ] Monte Carlo simulation of cluster-aware allocation strategies
- [ ] Transaction cost and turnover analysis for rebalancing triggers

### Phase 5: Extended Research (Proposed)

- [ ] **Extend to 2008**: Source pre-2010 data for ETFs that existed (SPY, QQQ, TLT, GLD) to include the GFC
- [ ] **Higher-frequency analysis**: Intraday 5-min returns during crisis windows
- [ ] **Cross-market extension**: Add sovereign CDS, VIX term structure, yield curve factors
- [ ] **Machine learning regime detection**: Replace HMM with neural regime-switching models
- [ ] **Causal discovery**: Apply PCMCI+ or DYNOTEARS for full causal graph learning
- [ ] **Crypto deep-dive**: Expand to individual tokens (BTC, ETH, SOL) + DeFi indices
- [ ] **Geopolitical NLP layer**: Integrate GDELT or news embeddings as an additional similarity layer
- [ ] **Publication**: Target Journal of Financial Economics, Review of Financial Studies, or Journal of Portfolio Management

---

## PhD-Level Research Workflow

For researchers looking to extend this work, here is the recommended logical workflow:

```
1. FOUNDATION (Completed)
   ├── Multi-asset universe construction (43 ETFs, 7 years)
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

4. VALIDATION & ROBUSTNESS (Next Priority)
   ├── Walk-forward out-of-sample testing
   ├── Bootstrap confidence intervals on TE rankings
   ├── Sensitivity to window size, threshold, resolution
   ├── Alternative clustering algorithms (Infomap, SBM)
   └── Multiple testing correction (Bonferroni, FDR)

5. REAL-TIME EXTENSION
   ├── Streaming architecture design
   ├── Incremental computation algorithms
   ├── Dashboard + alerting system
   └── Live validation against emerging events

6. THEORETICAL CONTRIBUTION
   ├── Formal proof: conditions under which tail CMI leads Pearson CMI
   ├── Topology crystallization: formalize as a phase transition
   ├── Information-theoretic bounds on early warning lead time
   └── Connection to market microstructure (order flow, liquidity)

7. PUBLICATION
   ├── Working paper with full methodology + proofs
   ├── Replication package (this repository)
   ├── Conference presentations (AFA, EFA, INFORMS)
   └── Journal submission
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work in academic research, please cite:

```bibtex
@misc{tavares2026topology,
  title={Dynamic Multi-Asset Topology and Cluster Migration Under Geopolitical Stress},
  author={Tavares, Nicholas and Hanini, Amjad and DaSilva, Brandon},
  year={2026},
  note={Available at: https://github.com/studyalwaysbro/asset-cluster-migration}
}
```

---

> **Reminder:** This project is for educational and research purposes only. It does not constitute investment advice and is not produced in any broker-dealer or investment advisory capacity. See the full disclaimer in the [research report](outputs/final_report.pdf).
