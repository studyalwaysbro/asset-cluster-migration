# Phase Progress Report — Asset Cluster Migration
## Generated: 2026-03-12 | Version: v0.4.0

---

## Executive Summary

The project has completed **Phases 1–4 (framework)** across 4 versions. All methodology is implemented, audited, and unit-tested (32/32 passing). The 91-ETF universe is configured but not yet fetched. The next step is to **run the full pipeline on real data** and populate results.

---

## Phase Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Single-layer topology (shrinkage corr, Leiden, CMI/TDS, HMM) | ✅ Complete |
| **Phase 2** | Multi-layer analysis (dCor, tail dep, multiplex consensus) | ✅ Complete |
| **Phase 3** | Directional flow (transfer entropy, Granger, lead-lag) | ✅ Complete |
| **Phase 3.5** | K-Means baseline + OOS validation framework | ✅ Complete |
| **Phase 4** | Statistical robustness framework | ✅ Complete (framework) |
| **Phase 4 — Run** | Execute robustness on full 91-ETF dataset | ⬜ Pending |
| **Phase 5** | Real-time streaming extension | ⬜ Future |
| **Phase 6** | Extended research (GFC, intraday, causal discovery) | ⬜ Future |

---

## What Was Done (This Session — v0.4.0)

### Critical Methodological Fixes
1. **CMI permutation invariance** — Hungarian algorithm prevents label-swap false positives
2. **TDS component scaling** — z-score normalization via `TDSNormalizer` class
3. **TDS spectral distance** — Wasserstein on Laplacian spectra (replaces broken L2)
4. **CPS bidirectional matching** — Hungarian instead of greedy best-Jaccard
5. **MST weight double-counting** — `max()` instead of `sum()` for symmetric entries
6. **Granger multiple testing** — Bonferroni across lags + ADF stationarity pre-check
7. **PELT penalty** — proper BIC: `d * log(n)` where d=2

### New Robustness Modules (5 files, ~800 lines)
| Module | File | What It Does |
|--------|------|--------------|
| Walk-Forward | `src/robustness/walk_forward.py` | Train 2019-2022/test 2023-2024, expand + retest 2025-2026 |
| Bootstrap | `src/robustness/bootstrap.py` | Block bootstrap CIs (Politis & Romano 1992), TE ranking stability |
| Sensitivity | `src/robustness/sensitivity.py` | Sweeps: window size, top-k, resolution, tail quantile |
| Multiple Testing | `src/robustness/multiple_testing.py` | Bonferroni, BH-FDR, Storey q-value, aggregate binomial |
| Surrogates | `src/robustness/surrogate_testing.py` | Phase-randomized, IAAFT, power analysis |

### Tests
- 32/32 passing (13 new robustness + 4 rewritten migration + 15 original)

### Infrastructure
- Branch protection active (GitHub ruleset #13815239) — contributors must submit PRs
- All code pushed to GitHub (commit `cb417d3`)

---

## What Needs to Run Next (Morning Workflow)

### Step-by-Step Execution Plan

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Fetch 91-ETF Data                              │
│  Command: make fetch-data                               │
│  Expected: ~91 tickers fetched via FMP stable API       │
│  Time: ~5-10 min (API rate limits)                      │
│  Note: Short-history ETFs auto-excluded by cleaner      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  STEP 2: Build Features                                 │
│  Command: make build-features                           │
│  Expected: Log returns, correlation matrices,           │
│            dCor, tail dependence, transfer entropy       │
│  Time: ~15-30 min (TE is O(n²) with KSG)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  STEP 3: Run Clustering + Migration                     │
│  Command: make run-clustering && make run-migration     │
│  Expected: Rolling Leiden + K-Means, CMI/TDS/AMF/CPS   │
│  Time: ~10-20 min                                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  STEP 4: Run Robustness Suite                           │
│  4a. Sensitivity sweeps (window, top-k, resolution,     │
│      tail quantile) — identifies ROBUST/SENSITIVE params│
│  4b. Bootstrap CIs (1000 resamples for TE rankings,     │
│      Granger F-stat confidence bands)                   │
│  4c. Multiple testing (Bonferroni/BH/Storey on full     │
│      Granger matrix — 8,190 pairs at 91 assets)         │
│  4d. Surrogate tests (phase-randomized + IAAFT null     │
│      distributions for regime-conditional TE)           │
│  4e. Walk-forward validation (train/test splits,        │
│      cross-layer Granger OOS, early warning signals)    │
│  Time: ~30-60 min (bootstrap is the bottleneck)         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  STEP 5: Generate Figures                               │
│  Command: make generate-figures                         │
│  Expected: 24+ figures (interactive Plotly HTML +        │
│            static PNGs for report)                      │
│  Includes: K-Means vs Leiden side-by-side               │
│  Time: ~5 min                                           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  STEP 6: Update Report + Push                           │
│  - Update outputs/final_report.pdf with new results     │
│  - Check off roadmap items in README                    │
│  - Push to GitHub                                       │
└─────────────────────────────────────────────────────────┘
```

### Quick Start (Single Command)
```bash
cd "C:\Users\regin\OneDrive\Desktop\Machine Learning\Asset Cluster Migration"
source .venv/Scripts/activate
make run-all
```

---

## Unchecked Roadmap Items (README.md)

These are the "run on real data" items that remain after the framework is built:

### Phase 3.5
- [ ] Generate side-by-side figures for paper (run `rolling_kmeans_baseline` on full dataset)
- [ ] Run validation on full dataset and document results

### Phase 4
- [ ] Run walk-forward on full dataset and document results
- [ ] Run bootstrap on full dataset and publish confidence bands
- [ ] Run sensitivity on full dataset and document parameter stability
- [ ] Run multiple testing on full Granger matrix (8,190 pairs at 91 assets) and document survival rate
- [ ] Run surrogate tests on regime-conditional TE and document results

### Phase 5 (Future)
- [ ] Streaming pipeline, incremental computation, dashboard, alerting

### Phase 6 (Future)
- [ ] Extend to 2008, intraday analysis, causal discovery, NLP layer, crypto deep-dive

---

## Key Results So Far (from 40-asset baseline, pre-v0.4.0 fixes)

These will be recalculated with corrected methodology on 91 assets:

| Finding | Metric | Value | Note |
|---------|--------|-------|------|
| Pre-event restructuring | TDS peak | 0.176 | Before Operation Epic Fury, exceeded COVID |
| COVID cluster dissolution | CMI | 1.0 | Complete dissolution, fast recovery |
| Tail → Pearson causality | Granger p | 0.041 | Tail-dep CMI predicts Pearson CMI |
| Leadership reversal | TE ranking | — | Calm: credit/RE lead. War: Treasuries lead |
| Nonlinear-only events | June 2025 | — | Invisible to Pearson, massive in dCor/tail |

**Important**: These findings were computed BEFORE the v0.4.0 methodological fixes. The corrected results may change quantitatively but should be qualitatively consistent based on prior visual inspection.

---

## Repository State

- **Branch**: `master`
- **Latest commit**: `cb417d3` (v0.4.0 — Phase 4 robustness framework)
- **Tests**: 32/32 passing
- **Branch protection**: Active (PRs required for non-admin contributors)
- **Universe**: 91 ETFs configured in `config/universe.yaml`
- **Data**: 40-asset baseline in `data/processed/` (needs refresh with 91 assets)

---

## File Map (Key Files)

```
src/
├── robustness/           ← NEW (Phase 4)
│   ├── walk_forward.py   — Train/test splits, early warning evaluation
│   ├── bootstrap.py      — Block bootstrap CIs, TE ranking stability
│   ├── sensitivity.py    — Hyperparameter sweeps + stability assessment
│   ├── multiple_testing.py — Bonferroni, BH-FDR, Storey q-value
│   └── surrogate_testing.py — Phase-randomized/IAAFT surrogates, power analysis
├── migration/
│   └── metrics.py        ← FIXED (Hungarian matching, TDS normalizer)
├── features/
│   └── lead_lag.py       ← FIXED (Bonferroni across lags, ADF pre-check)
├── graphs/
│   └── construction.py   ← FIXED (MST double-counting)
├── regimes/
│   └── changepoint.py    ← FIXED (BIC penalty)
└── clustering/
    └── kmeans.py          — K-Means baseline engine

tests/
├── test_migration_metrics.py  ← REWRITTEN (7 tests for Hungarian matching)
└── test_robustness.py         ← NEW (13 tests)
```

---

*Report generated automatically. Next action: run `make run-all` to execute the full pipeline on 91-ETF data.*
