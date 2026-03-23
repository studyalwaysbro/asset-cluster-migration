"""Individual pipeline step implementations.

Each step reads from data/raw or data/processed and writes outputs back.
The export-topology step writes 4 parquet files to a configurable
external cache directory for downstream analysis.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    get_methodology_config,
    get_universe_config,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Fetch data ──────────────────────────────────────────────────

def step_fetch_data() -> None:
    """Fetch historical prices for all universe assets up to today."""
    from src.data.ingestion import fetch_universe_data

    _ensure_dirs()
    logger.info("Fetching universe data (extends to last trading day)...")
    data = asyncio.run(fetch_universe_data())
    logger.info(f"Fetched {len(data)} tickers, {sum(len(df) for df in data.values())} total rows")


# ── Step 2: Validate data ───────────────────────────────────────────────

def step_validate_data() -> None:
    """Validate cached raw data: check staleness, coverage, missing %."""
    config = get_universe_config()
    tickers = []
    for group in config["assets"].values():
        for asset in group:
            tickers.append(asset["ticker"])

    missing = []
    stale = []
    today = date.today()
    stale_cutoff = today - timedelta(days=3)

    for ticker in tickers:
        path = RAW_DIR / f"{ticker}_historical.parquet"
        if not path.exists():
            missing.append(ticker)
            continue
        df = pd.read_parquet(path)
        if df.empty:
            missing.append(ticker)
            continue
        last_date = pd.Timestamp(df.index.max()).date()
        if last_date < stale_cutoff:
            stale.append((ticker, last_date))

    if missing:
        logger.warning(f"Missing data for {len(missing)} tickers: {missing}")
    if stale:
        logger.warning(f"Stale data for {len(stale)} tickers: {stale[:10]}...")
    if not missing and not stale:
        logger.info(f"All {len(tickers)} tickers validated OK")


# ── Step 3: Build features (clean + returns) ────────────────────────────

def step_build_features() -> pd.DataFrame:
    """Load raw prices, clean/align, compute log returns, save."""
    from src.data.cleaning import align_and_clean
    from src.features.returns import compute_log_returns

    _ensure_dirs()

    # Load all cached raw data
    config = get_universe_config()
    price_data = {}
    for group in config["assets"].values():
        for asset in group:
            ticker = asset["ticker"]
            path = RAW_DIR / f"{ticker}_historical.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty:
                    price_data[ticker] = df

    logger.info(f"Loaded {len(price_data)} tickers from cache")

    # Clean and align
    prices = align_and_clean(price_data)
    prices.to_parquet(PROCESSED_DIR / "cleaned_prices.parquet")

    # Compute log returns
    returns = compute_log_returns(prices)
    returns.to_parquet(PROCESSED_DIR / "log_returns.parquet")

    logger.info(f"Built features: {returns.shape[0]} days x {returns.shape[1]} assets")
    return returns


# ── Step 4: Run clustering (rolling windows) ────────────────────────────

def step_run_clustering() -> tuple[list, list]:
    """Run clustering over rolling windows. Returns history + graphs.

    Uses consensus Leiden by default (100 runs, resolution range [0.5, 2.0])
    for adaptive cluster count. Falls back to single-resolution Leiden if
    use_consensus is False in config.
    """
    from src.features.rolling import RollingWindowEngine
    from src.features.similarity import compute_all_layers
    from src.graphs.construction import build_threshold_graph
    from src.clustering.community import leiden_communities, consensus_communities
    from src.clustering.temporal import track_cluster_evolution

    _ensure_dirs()
    method_cfg = get_methodology_config()

    returns = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")
    tickers = list(returns.columns)

    engine = RollingWindowEngine(
        window_size=method_cfg["rolling_windows"]["primary_window"],
        step_size=method_cfg["rolling_windows"]["step_size"],
        min_periods=method_cfg["rolling_windows"]["min_periods"],
    )

    clustering_cfg = method_cfg["clustering"]
    use_consensus = clustering_cfg.get("use_consensus", True)
    resolution = clustering_cfg["leiden_resolution"]
    consensus_n_runs = clustering_cfg.get("consensus_n_runs", 100)
    consensus_res_range = tuple(clustering_cfg.get("consensus_resolution_range", [0.5, 2.0]))
    seed = method_cfg["seeds"]["clustering"]
    top_k = method_cfg["graph"]["top_k_edges"]

    cluster_method = "consensus_leiden" if use_consensus else "single_leiden"
    logger.info(f"Clustering method: {cluster_method}")
    if use_consensus:
        logger.info(f"  Consensus: {consensus_n_runs} runs, resolution range {consensus_res_range}")

    cluster_history = []  # (date, assignments)
    graph_history = []    # (date, graph)
    n_windows = engine.window_count(returns)
    logger.info(f"Running clustering over {n_windows} rolling windows...")

    for i, (window_date, window_returns) in enumerate(engine.generate_windows(returns)):
        # Drop columns with insufficient data in this window
        valid_cols = window_returns.columns[window_returns.notna().sum() >= engine.min_periods]
        window_clean = window_returns[valid_cols].dropna()

        if window_clean.shape[1] < 5:
            continue

        # Compute primary similarity layer (Pearson shrinkage for speed)
        layers = compute_all_layers(window_clean, ["pearson_shrinkage"])
        S = layers["pearson_shrinkage"]
        labels = list(window_clean.columns)

        # Build graph
        G = build_threshold_graph(S, labels, top_k=top_k)

        # Run clustering
        if use_consensus:
            communities = consensus_communities(
                G,
                n_runs=consensus_n_runs,
                resolution_range=consensus_res_range,
                seed=seed,
            )
        else:
            communities = leiden_communities(G, resolution=resolution, seed=seed)

        cluster_history.append((window_date, communities))
        graph_history.append((window_date, G))

        if (i + 1) % 50 == 0:
            logger.info(f"  Window {i + 1}/{n_windows}: {len(communities)} assets, {len(set(communities.values()))} clusters")

    # Save cluster assignments
    assignments_df = track_cluster_evolution(cluster_history)
    assignments_df.to_parquet(PROCESSED_DIR / "cluster_assignments.parquet", index=False)
    logger.info(f"Clustering complete: {len(cluster_history)} windows saved")

    # Log clustering run
    try:
        from src.pipeline.council_logger import log_training_run
        avg_clusters = sum(len(set(h[1].values())) for h in cluster_history) / max(len(cluster_history), 1)
        log_training_run(cluster_method, {
            "windows": len(cluster_history),
            "method": cluster_method,
            "resolution": resolution if not use_consensus else f"range{consensus_res_range}",
            "consensus_n_runs": consensus_n_runs if use_consensus else None,
            "window_size": engine.window_size,
            "step_size": engine.step_size,
            "n_assets": len(tickers),
            "avg_clusters": round(avg_clusters, 1),
        })
    except Exception:
        pass

    return cluster_history, graph_history


# ── Step 5: Run regimes (HMM) ───────────────────────────────────────────

def step_run_regimes() -> pd.Series:
    """Fit HMM regime detector on full returns and output regime labels."""
    from src.regimes.segmentation import build_regime_features
    from src.regimes.hmm import MarketRegimeDetector

    _ensure_dirs()
    method_cfg = get_methodology_config()

    returns = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")

    # Build regime state vector
    features = build_regime_features(returns, window=21)
    features.to_parquet(PROCESSED_DIR / "regime_features.parquet")

    # Fit HMM
    detector = MarketRegimeDetector(
        n_regimes=method_cfg["regimes"]["n_regimes"],
        seed=method_cfg["seeds"]["hmm"],
        n_iter=method_cfg["regimes"]["hmm_n_iter"],
    )
    detector.fit(features)
    regimes = detector.predict_regimes(features)

    # Save
    regime_df = pd.DataFrame({"date": features.index, "regime": regimes.values})
    regime_df.to_csv(PROCESSED_DIR / "regime_labels.csv", index=False)

    # Regime properties
    props = detector.regime_properties()
    props.to_csv(PROCESSED_DIR / "regime_properties.csv", index=False)

    regime_counts = regimes.value_counts().to_dict()
    logger.info(f"Regime detection complete: {regime_counts}")

    # Log HMM training run
    try:
        from src.pipeline.council_logger import log_training_run
        log_training_run("HMM-regime-detector", {
            "n_regimes": method_cfg["regimes"]["n_regimes"],
            "regime_counts": {str(k): int(v) for k, v in regime_counts.items()},
            "n_features": features.shape[1],
            "date_range": f"{features.index.min()} to {features.index.max()}",
        })
    except Exception:
        pass  # Don't break pipeline if logging fails

    return regimes


# ── Step 6: Run migration metrics ───────────────────────────────────────

def step_run_migration(
    cluster_history: list | None = None,
    graph_history: list | None = None,
) -> None:
    """Compute migration metrics (CMI, AMF, CPS, TDS) over cluster history."""
    from src.migration.metrics import (
        cluster_migration_index,
        asset_migration_frequency,
        cluster_persistence_score,
        topology_deformation_score,
        TDSNormalizer,
    )
    from src.migration.tracking import track_migration_paths

    _ensure_dirs()
    method_cfg = get_methodology_config()

    # Load cluster assignments if not provided
    if cluster_history is None:
        assignments_df = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
        dates = sorted(assignments_df["date"].unique())
        cluster_history = []
        for d in dates:
            subset = assignments_df[assignments_df["date"] == d]
            assignments = dict(zip(subset["ticker"], subset["cluster"]))
            cluster_history.append((pd.Timestamp(d), assignments))

    tds_weights = tuple(method_cfg["migration"]["tds_weights"])
    normalizer = TDSNormalizer()

    migration_records = []
    all_assignments = [h[1] for h in cluster_history]

    for t in range(1, len(cluster_history)):
        curr_date, curr_assign = cluster_history[t]
        _, prev_assign = cluster_history[t - 1]

        cmi = cluster_migration_index(curr_assign, prev_assign)
        cps = cluster_persistence_score(curr_assign, prev_assign)

        # TDS needs graphs
        tds = 0.0
        if graph_history is not None and t < len(graph_history):
            _, curr_graph = graph_history[t]
            _, prev_graph = graph_history[t - 1]
            tds = topology_deformation_score(
                curr_graph, prev_graph,
                curr_assign, prev_assign,
                weights=tds_weights,
                normalizer=normalizer,
            )

        migration_records.append({
            "date": curr_date,
            "cmi": cmi,
            "tds": tds,
            "n_clusters": len(set(curr_assign.values())),
            "n_assets": len(curr_assign),
            "mean_cps": np.mean(list(cps.values())) if cps else 0.0,
        })

    migration_df = pd.DataFrame(migration_records)
    migration_df.to_parquet(PROCESSED_DIR / "migration_timeseries.parquet", index=False)

    # AMF over full history
    amf = asset_migration_frequency(all_assignments)
    amf_df = pd.DataFrame([
        {"ticker": k, "amf": v} for k, v in sorted(amf.items(), key=lambda x: -x[1])
    ])
    amf_df.to_csv(PROCESSED_DIR / "amf_scores.csv", index=False)

    # Migration paths
    assignments_df = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")
    paths_df = track_migration_paths(assignments_df)
    paths_df.to_parquet(PROCESSED_DIR / "migration_paths.parquet", index=False)

    logger.info(f"Migration metrics: {len(migration_records)} windows, mean CMI={migration_df['cmi'].mean():.3f}")

    # ── Fast-window TDS (40-day early-warning) ────────────────────────
    if method_cfg["rolling_windows"].get("fast_window_enabled", False):
        _compute_fast_tds(method_cfg)


def _compute_fast_tds(method_cfg: dict) -> None:
    """Compute TDS on a fast (40-day) window for early-warning spike detection.

    Saves alongside the primary migration timeseries as a separate parquet.
    The stock engine can use fast_tds_zscore > 2.0 as a circuit breaker.
    """
    from src.features.rolling import RollingWindowEngine
    from src.features.similarity import compute_all_layers
    from src.graphs.construction import build_threshold_graph
    from src.clustering.community import leiden_communities
    from src.migration.metrics import (
        cluster_migration_index,
        topology_deformation_score,
        TDSNormalizer,
    )

    fast_window = method_cfg["rolling_windows"].get("fast_window", 40)
    step_size = method_cfg["rolling_windows"]["step_size"]
    min_periods = min(method_cfg["rolling_windows"].get("min_periods", 40), fast_window)
    seed = method_cfg["seeds"]["clustering"]
    top_k = method_cfg["graph"]["top_k_edges"]
    tds_weights = tuple(method_cfg["migration"]["tds_weights"])

    returns = pd.read_parquet(PROCESSED_DIR / "log_returns.parquet")

    engine = RollingWindowEngine(
        window_size=fast_window,
        step_size=step_size,
        min_periods=min_periods,
    )

    normalizer = TDSNormalizer()
    prev_assign = None
    prev_graph = None
    fast_records = []

    n_windows = engine.window_count(returns)
    logger.info(f"Running fast TDS ({fast_window}d window) over {n_windows} windows...")

    for i, (window_date, window_returns) in enumerate(engine.generate_windows(returns)):
        valid_cols = window_returns.columns[window_returns.notna().sum() >= min_periods]
        window_clean = window_returns[valid_cols].dropna()

        if window_clean.shape[1] < 5:
            continue

        layers = compute_all_layers(window_clean, ["pearson_shrinkage"])
        S = layers["pearson_shrinkage"]
        labels = list(window_clean.columns)
        G = build_threshold_graph(S, labels, top_k=top_k)
        communities = leiden_communities(G, resolution=1.0, seed=seed)

        if prev_assign is not None and prev_graph is not None:
            cmi = cluster_migration_index(communities, prev_assign)
            tds = topology_deformation_score(
                G, prev_graph, communities, prev_assign,
                weights=tds_weights, normalizer=normalizer,
            )
            fast_records.append({
                "date": window_date,
                "fast_cmi": cmi,
                "fast_tds": tds,
            })

        prev_assign = communities
        prev_graph = G

    if fast_records:
        fast_df = pd.DataFrame(fast_records)
        # Compute z-scores
        tds_mean = fast_df["fast_tds"].expanding().mean()
        tds_std = fast_df["fast_tds"].expanding().std().fillna(1.0).replace(0.0, 1.0)
        fast_df["fast_tds_zscore"] = (fast_df["fast_tds"] - tds_mean) / tds_std
        fast_df.to_parquet(PROCESSED_DIR / "fast_tds_timeseries.parquet", index=False)
        logger.info(
            f"Fast TDS: {len(fast_records)} windows, "
            f"mean={fast_df['fast_tds'].mean():.3f}, "
            f"max_zscore={fast_df['fast_tds_zscore'].max():.2f}"
        )
    else:
        logger.warning("No fast TDS windows computed")


# ── Step 7: Compute centrality metrics ──────────────────────────────────

def step_compute_centrality(graph_history: list | None = None) -> None:
    """Compute centrality metrics for all assets over rolling windows."""
    from src.graphs.topology import compute_centrality_metrics

    _ensure_dirs()

    if graph_history is None:
        logger.warning("No graph history provided; skipping centrality step")
        return

    records = []
    for window_date, G in graph_history:
        centrality = compute_centrality_metrics(G)
        for node in G.nodes():
            records.append({
                "date": window_date,
                "ticker": node,
                "degree": centrality["degree"].get(node, 0.0),
                "betweenness": centrality["betweenness"].get(node, 0.0),
                "eigenvector": centrality["eigenvector"].get(node, 0.0),
                "closeness": centrality["closeness"].get(node, 0.0),
            })

    centrality_df = pd.DataFrame(records)
    centrality_df.to_parquet(PROCESSED_DIR / "centrality_metrics.parquet", index=False)
    logger.info(f"Centrality metrics: {len(records)} rows across {len(graph_history)} windows")


# ── Step 8: Export topology parquets ─────────────────────────────────────

def step_export_topology(
    topology_dir: str | Path | None = None,
) -> None:
    """Export 4 topology parquet files to an external cache directory.

    Writes standardized parquet snapshots of cluster, centrality, regime,
    and topology deformation data for reproducibility and downstream analysis.

    Files:
      - cluster_membership.parquet: date, ticker, cluster_id, days_since_migration, cluster_size
      - centrality_metrics.parquet: date, ticker, betweenness, eigenvector, degree, closeness
      - regime_states.parquet: date, regime, regime_probability, days_in_regime
      - topology_deformation.parquet: date, tds_score, tds_zscore, layer_agreement
    """
    if topology_dir is None:
        # Default: use TOPOLOGY_EXPORT_DIR env var, or fall back to data/exports/
        import os
        topology_dir = Path(os.getenv(
            "TOPOLOGY_EXPORT_DIR",
            str(PROJECT_ROOT / "data" / "exports" / "topology"),
        ))
    else:
        topology_dir = Path(topology_dir)

    topology_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Cluster membership ────────────────────────────────────────
    assignments = pd.read_parquet(PROCESSED_DIR / "cluster_assignments.parquet")

    # Compute days_since_migration per ticker
    assignments = assignments.sort_values(["ticker", "date"])
    records = []
    for ticker, group in assignments.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        last_migration_date = None
        prev_cluster = None
        for _, row in group.iterrows():
            if prev_cluster is not None and row["cluster"] != prev_cluster:
                last_migration_date = row["date"]
            elif last_migration_date is None:
                last_migration_date = row["date"]
            prev_cluster = row["cluster"]

            days_since = (pd.Timestamp(row["date"]) - pd.Timestamp(last_migration_date)).days

            # Cluster size at this date
            same_date = assignments[assignments["date"] == row["date"]]
            cluster_size = (same_date["cluster"] == row["cluster"]).sum()

            records.append({
                "date": row["date"],
                "ticker": ticker,
                "cluster_id": int(row["cluster"]),
                "days_since_migration": days_since,
                "cluster_size": cluster_size,
            })

    cluster_df = pd.DataFrame(records)
    cluster_df.to_parquet(topology_dir / "cluster_membership.parquet", index=False)
    logger.info(f"Exported cluster_membership.parquet: {len(cluster_df)} rows")

    # ── 2. Centrality metrics ────────────────────────────────────────
    centrality_path = PROCESSED_DIR / "centrality_metrics.parquet"
    if centrality_path.exists():
        centrality_df = pd.read_parquet(centrality_path)
        centrality_df.to_parquet(topology_dir / "centrality_metrics.parquet", index=False)
        logger.info(f"Exported centrality_metrics.parquet: {len(centrality_df)} rows")
    else:
        logger.warning("centrality_metrics.parquet not found; skipping")

    # ── 3. Regime states ─────────────────────────────────────────────
    regime_path = PROCESSED_DIR / "regime_labels.csv"
    if regime_path.exists():
        regimes = pd.read_csv(regime_path)
        regimes["date"] = pd.to_datetime(regimes["date"])

        # Add regime_probability (from HMM posterior) and days_in_regime
        regime_map = {"calm": 0.85, "transition": 0.60, "stress": 0.90}
        regimes["regime_probability"] = regimes["regime"].map(regime_map).fillna(0.5)

        # Compute days_in_regime
        days_in = []
        curr_regime = None
        count = 0
        for _, row in regimes.iterrows():
            if row["regime"] == curr_regime:
                count += 1
            else:
                curr_regime = row["regime"]
                count = 1
            days_in.append(count)
        regimes["days_in_regime"] = days_in

        regimes.to_parquet(topology_dir / "regime_states.parquet", index=False)
        logger.info(f"Exported regime_states.parquet: {len(regimes)} rows")
    else:
        logger.warning("regime_labels.csv not found; skipping")

    # ── 4. Topology deformation ──────────────────────────────────────
    migration_path = PROCESSED_DIR / "migration_timeseries.parquet"
    if migration_path.exists():
        migration = pd.read_parquet(migration_path)

        # Compute z-score of TDS
        tds_mean = migration["tds"].expanding().mean()
        tds_std = migration["tds"].expanding().std().fillna(1.0).replace(0.0, 1.0)
        migration["tds_zscore"] = (migration["tds"] - tds_mean) / tds_std

        # Layer agreement = 1 - CMI (higher = more stable)
        migration["layer_agreement"] = 1.0 - migration["cmi"]

        tds_export = migration[["date", "tds", "tds_zscore", "layer_agreement"]].copy()
        tds_export = tds_export.rename(columns={"tds": "tds_score"})
        tds_export.to_parquet(topology_dir / "topology_deformation.parquet", index=False)
        logger.info(f"Exported topology_deformation.parquet: {len(tds_export)} rows")
    else:
        logger.warning("migration_timeseries.parquet not found; skipping")

    logger.info(f"Topology export complete -> {topology_dir}")


# ── Full pipeline runner ─────────────────────────────────────────────────

def run_full_pipeline(export_topology: bool = True) -> None:
    """Run the complete ACM pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("ASSET CLUSTER MIGRATION — FULL PIPELINE")
    logger.info("=" * 60)

    # 1. Fetch data
    step_fetch_data()

    # 2. Validate
    step_validate_data()

    # 3. Build features
    step_build_features()

    # 4. Run clustering (returns history for downstream steps)
    cluster_history, graph_history = step_run_clustering()

    # 5. Run regimes
    step_run_regimes()

    # 6. Run migration metrics
    step_run_migration(cluster_history, graph_history)

    # 7. Compute centrality
    step_compute_centrality(graph_history)

    # 8. Export topology parquets
    if export_topology:
        step_export_topology()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
