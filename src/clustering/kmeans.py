"""K-Means baseline for comparison against topology-aware clustering."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.migration.metrics import cluster_migration_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core baseline
# ---------------------------------------------------------------------------

def kmeans_communities(
    returns: pd.DataFrame,
    n_clusters: int = 4,
    seed: int = 42,
) -> dict[str, int]:
    """Cluster assets via K-Means on a returns window.

    Each asset becomes a point in T-dimensional return space.  K-Means
    partitions these points using Euclidean distance — a purely linear,
    variance-based method with no notion of network topology.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows = dates, columns = tickers.  One rolling window of data.
    n_clusters : int
        Number of clusters (matched to Leiden for fair comparison).
    seed : int
        Random state for reproducibility.

    Returns
    -------
    dict[str, int]
        Mapping of ticker → cluster_id (same format as Leiden output).
    """
    clean = returns.dropna(axis=1)
    if clean.shape[1] < n_clusters:
        logger.warning(
            "Fewer valid assets (%d) than clusters (%d) — returning trivial partition.",
            clean.shape[1],
            n_clusters,
        )
        return {t: 0 for t in clean.columns}

    # Transpose: rows = assets, columns = daily returns (features)
    features = clean.T.values

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = model.fit_predict(features)

    return dict(zip(clean.columns, labels.tolist()))


# ---------------------------------------------------------------------------
# Rolling baseline comparison
# ---------------------------------------------------------------------------

def rolling_kmeans_baseline(
    returns: pd.DataFrame,
    windows: list[tuple[pd.Timestamp, pd.DataFrame]],
    leiden_history: list[tuple[pd.Timestamp, dict[str, int]]],
    n_clusters: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Run K-Means across the same rolling windows used by Leiden.

    Produces a date-indexed DataFrame that, for every window, records:
      • the K-Means CMI (cluster migration index)
      • the Leiden CMI (for comparison)
      • ARI between the two partitions
      • NMI between the two partitions
      • K-Means silhouette score

    This is the core "same input → different algorithm → compare" baseline.

    Parameters
    ----------
    returns : pd.DataFrame
        Full returns matrix (superset of every window).
    windows : list[tuple[Timestamp, DataFrame]]
        The *exact* rolling windows consumed by the Leiden pipeline.
    leiden_history : list[tuple[Timestamp, dict[str, int]]]
        Dated Leiden partitions in chronological order.
    n_clusters : int
        Passed through to K-Means (should match Leiden cluster count).
    seed : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: kmeans_cmi, leiden_cmi, ari, nmi, kmeans_silhouette.
    """
    leiden_lookup = dict(leiden_history)
    prev_km: dict[str, int] | None = None
    prev_ld: dict[str, int] | None = None
    records: list[dict] = []

    for date, window_returns in windows:
        km = kmeans_communities(window_returns, n_clusters=n_clusters, seed=seed)
        ld = leiden_lookup.get(date)

        # --- CMI for both methods ----------------------------------------
        km_cmi = cluster_migration_index(km, prev_km) if prev_km else 0.0
        ld_cmi = (
            cluster_migration_index(ld, prev_ld)
            if ld and prev_ld
            else 0.0
        )

        # --- Cross-method agreement (ARI / NMI) --------------------------
        ari, nmi = _partition_agreement(km, ld) if ld else (np.nan, np.nan)

        # --- Silhouette (internal quality of K-Means partition) -----------
        sil = _silhouette(window_returns, km)

        records.append(
            {
                "date": date,
                "kmeans_cmi": km_cmi,
                "leiden_cmi": ld_cmi,
                "ari": ari,
                "nmi": nmi,
                "kmeans_silhouette": sil,
            }
        )
        prev_km = km
        prev_ld = ld

    df = pd.DataFrame(records).set_index("date").sort_index()
    logger.info(
        "Baseline comparison: %d windows, mean K-Means CMI=%.4f, mean Leiden CMI=%.4f",
        len(df),
        df["kmeans_cmi"].mean(),
        df["leiden_cmi"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Event-window summary
# ---------------------------------------------------------------------------

def baseline_event_summary(
    comparison: pd.DataFrame,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    post_start: pd.Timestamp,
    post_end: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate baseline comparison into pre / event / post windows.

    Returns a DataFrame with one row per phase showing mean CMI for both
    methods, mean ARI, and NMI — the table you'd put in the paper.
    """
    phases = {
        "pre": (pre_start, pre_end),
        "event": (event_start, event_end),
        "post": (post_start, post_end),
    }
    rows: list[dict] = []
    for phase, (start, end) in phases.items():
        mask = (comparison.index >= start) & (comparison.index <= end)
        subset = comparison.loc[mask]
        rows.append(
            {
                "phase": phase,
                "n_windows": len(subset),
                "kmeans_cmi_mean": subset["kmeans_cmi"].mean() if len(subset) else np.nan,
                "leiden_cmi_mean": subset["leiden_cmi"].mean() if len(subset) else np.nan,
                "ari_mean": subset["ari"].mean() if len(subset) else np.nan,
                "nmi_mean": subset["nmi"].mean() if len(subset) else np.nan,
                "kmeans_sil_mean": subset["kmeans_silhouette"].mean() if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("phase")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partition_agreement(
    a: dict[str, int], b: dict[str, int]
) -> tuple[float, float]:
    """ARI and NMI between two partitions on their common assets."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return np.nan, np.nan
    la = [a[t] for t in common]
    lb = [b[t] for t in common]
    return adjusted_rand_score(la, lb), normalized_mutual_info_score(la, lb)


def _silhouette(returns: pd.DataFrame, assignments: dict[str, int]) -> float:
    """Silhouette score for a partition (higher = better separation)."""
    tickers = [t for t in returns.columns if t in assignments]
    if len(tickers) < 3:
        return np.nan
    X = returns[tickers].dropna(axis=0).T.values
    labels = [assignments[t] for t in tickers]
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= len(tickers):
        return np.nan
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except ValueError:
        return np.nan
