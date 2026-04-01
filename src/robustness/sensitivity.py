"""Sensitivity analysis for hyperparameter choices.

Phase 4.3 — Systematic sweep of key parameters to test finding stability:
1. Rolling window size: 60, 90, 120, 150, 180, 252 days
2. Top-K threshold: 3, 5, 7, 10 edges per node
3. Leiden resolution: 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0
4. Tail dependence quantile: 0.01, 0.03, 0.05, 0.10

For each configuration, recomputes CMI, TDS, and cluster structure,
then measures deviation from baseline (primary_window=120, top_k=5, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)


@dataclass
class SensitivityPoint:
    """Result for a single parameter configuration."""
    param_name: str
    param_value: float
    mean_cmi: float
    std_cmi: float
    mean_tds: float
    std_tds: float
    n_clusters_mode: int  # most common cluster count
    ari_vs_baseline: float  # agreement with baseline clustering
    nmi_vs_baseline: float
    mean_silhouette: float


@dataclass
class SensitivityResult:
    """Full sensitivity analysis result for one parameter."""
    param_name: str
    points: list[SensitivityPoint] = field(default_factory=list)
    baseline_value: float = 0.0
    conclusion: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy visualization."""
        records = []
        for p in self.points:
            records.append({
                "param_name": p.param_name,
                "param_value": p.param_value,
                "mean_cmi": p.mean_cmi,
                "std_cmi": p.std_cmi,
                "mean_tds": p.mean_tds,
                "std_tds": p.std_tds,
                "n_clusters_mode": p.n_clusters_mode,
                "ari_vs_baseline": p.ari_vs_baseline,
                "nmi_vs_baseline": p.nmi_vs_baseline,
                "mean_silhouette": p.mean_silhouette,
            })
        return pd.DataFrame(records)


def sweep_window_size(
    returns: pd.DataFrame,
    window_sizes: list[int] | None = None,
    step_size: int = 5,
    baseline_window: int = 120,
    leiden_resolution: float = 1.0,
    top_k: int = 5,
    seed: int = 42,
) -> SensitivityResult:
    """Sweep rolling window sizes and measure impact on CMI, TDS, clusters.

    Tests whether findings survive different temporal granularities.
    Shorter windows = more responsive but noisier.
    Longer windows = smoother but lagging.
    """
    from src.features.similarity import shrinkage_correlation
    from src.graphs.construction import build_threshold_graph
    from src.clustering.community import leiden_communities
    from src.migration.metrics import cluster_migration_index

    if window_sizes is None:
        window_sizes = [60, 90, 120, 150, 180, 252]

    result = SensitivityResult(
        param_name="window_size",
        baseline_value=baseline_window,
    )

    # Compute baseline clustering for ARI/NMI comparison
    baseline_assignments = None
    labels = returns.columns.tolist()

    for ws in window_sizes:
        logger.info(f"Sensitivity: window_size={ws}")
        cmis = []
        cluster_counts = []
        assignments_list = []

        n_windows = 0
        for start in range(0, len(returns) - ws, step_size):
            window = returns.iloc[start:start + ws]
            if len(window) < ws:
                continue

            try:
                corr = shrinkage_correlation(window)
                G = build_threshold_graph(corr, labels, top_k=top_k)
                communities = leiden_communities(G, resolution=leiden_resolution, seed=seed)
                assignments_list.append(communities)
                cluster_counts.append(len(set(communities.values())))
                n_windows += 1
            except Exception as e:
                logger.debug(f"Window {start} failed: {e}")
                continue

        # Compute CMI between consecutive windows
        for t in range(1, len(assignments_list)):
            cmis.append(
                cluster_migration_index(assignments_list[t], assignments_list[t - 1])
            )

        # Compare last assignment to baseline
        ari = 0.0
        nmi = 0.0
        if ws == baseline_window and assignments_list:
            baseline_assignments = assignments_list[-1]
        elif baseline_assignments and assignments_list:
            common = sorted(
                set(assignments_list[-1].keys()) & set(baseline_assignments.keys())
            )
            if len(common) > 1:
                a1 = [assignments_list[-1][a] for a in common]
                a2 = [baseline_assignments[a] for a in common]
                ari = adjusted_rand_score(a1, a2)
                nmi = normalized_mutual_info_score(a1, a2)

        # Silhouette score
        mean_sil = _compute_silhouette(returns.iloc[-ws:], assignments_list[-1] if assignments_list else {}, labels)

        from statistics import mode as stat_mode
        try:
            n_clusters = stat_mode(cluster_counts) if cluster_counts else 0
        except Exception:
            n_clusters = int(np.median(cluster_counts)) if cluster_counts else 0

        result.points.append(SensitivityPoint(
            param_name="window_size",
            param_value=ws,
            mean_cmi=np.mean(cmis) if cmis else 0.0,
            std_cmi=np.std(cmis) if cmis else 0.0,
            mean_tds=0.0,  # TDS requires graph comparison, computed separately
            std_tds=0.0,
            n_clusters_mode=n_clusters,
            ari_vs_baseline=ari,
            nmi_vs_baseline=nmi,
            mean_silhouette=mean_sil,
        ))

    # Assess stability
    cmi_range = max(p.mean_cmi for p in result.points) - min(p.mean_cmi for p in result.points)
    if cmi_range < 0.05:
        result.conclusion = "ROBUST: CMI stable across window sizes (range < 0.05)"
    elif cmi_range < 0.15:
        result.conclusion = f"MODERATE: CMI varies moderately (range = {cmi_range:.3f})"
    else:
        result.conclusion = f"SENSITIVE: CMI varies substantially (range = {cmi_range:.3f})"

    return result


def sweep_top_k(
    returns: pd.DataFrame,
    top_k_values: list[int] | None = None,
    window_size: int = 120,
    leiden_resolution: float = 1.0,
    seed: int = 42,
) -> SensitivityResult:
    """Sweep graph sparsification threshold (top-K edges per node)."""
    from src.features.similarity import shrinkage_correlation
    from src.graphs.construction import build_threshold_graph
    from src.clustering.community import leiden_communities
    from src.migration.metrics import cluster_migration_index

    if top_k_values is None:
        top_k_values = [3, 5, 7, 10]

    result = SensitivityResult(
        param_name="top_k",
        baseline_value=5,
    )

    labels = returns.columns.tolist()
    baseline_assignments = None

    # Use last window for comparison
    window = returns.iloc[-window_size:]
    corr = shrinkage_correlation(window)

    for k in top_k_values:
        logger.info(f"Sensitivity: top_k={k}")

        try:
            G = build_threshold_graph(corr, labels, top_k=k)
            communities = leiden_communities(G, resolution=leiden_resolution, seed=seed)
            n_clusters = len(set(communities.values()))

            if k == 5:
                baseline_assignments = communities

            ari = 0.0
            nmi = 0.0
            if baseline_assignments and k != 5:
                common = sorted(set(communities.keys()) & set(baseline_assignments.keys()))
                if len(common) > 1:
                    a1 = [communities[a] for a in common]
                    a2 = [baseline_assignments[a] for a in common]
                    ari = adjusted_rand_score(a1, a2)
                    nmi = normalized_mutual_info_score(a1, a2)
            elif k == 5:
                ari = 1.0
                nmi = 1.0

            mean_sil = _compute_silhouette(window, communities, labels)

            result.points.append(SensitivityPoint(
                param_name="top_k",
                param_value=k,
                mean_cmi=0.0,
                std_cmi=0.0,
                mean_tds=0.0,
                std_tds=0.0,
                n_clusters_mode=n_clusters,
                ari_vs_baseline=ari,
                nmi_vs_baseline=nmi,
                mean_silhouette=mean_sil,
            ))
        except Exception as e:
            logger.error(f"top_k={k} failed: {e}")

    # Assess
    ari_values = [p.ari_vs_baseline for p in result.points if p.param_value != 5]
    min_ari = min(ari_values) if ari_values else 1.0
    if min_ari > 0.7:
        result.conclusion = f"ROBUST: cluster structure stable (min ARI vs baseline = {min_ari:.3f})"
    elif min_ari > 0.4:
        result.conclusion = f"MODERATE: some sensitivity to top_k (min ARI = {min_ari:.3f})"
    else:
        result.conclusion = f"SENSITIVE: cluster structure changes substantially (min ARI = {min_ari:.3f})"

    return result


def sweep_leiden_resolution(
    returns: pd.DataFrame,
    resolutions: list[float] | None = None,
    window_size: int = 120,
    top_k: int = 5,
    seed: int = 42,
) -> SensitivityResult:
    """Sweep Leiden resolution parameter."""
    from src.features.similarity import shrinkage_correlation
    from src.graphs.construction import build_threshold_graph
    from src.clustering.community import leiden_communities

    if resolutions is None:
        resolutions = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]

    result = SensitivityResult(
        param_name="leiden_resolution",
        baseline_value=1.0,
    )

    labels = returns.columns.tolist()
    window = returns.iloc[-window_size:]
    corr = shrinkage_correlation(window)
    G = build_threshold_graph(corr, labels, top_k=top_k)

    baseline_assignments = None

    for res in resolutions:
        logger.info(f"Sensitivity: leiden_resolution={res}")

        try:
            communities = leiden_communities(G, resolution=res, seed=seed)
            n_clusters = len(set(communities.values()))

            if res == 1.0:
                baseline_assignments = communities

            ari = 0.0
            nmi = 0.0
            if baseline_assignments and res != 1.0:
                common = sorted(set(communities.keys()) & set(baseline_assignments.keys()))
                if len(common) > 1:
                    a1 = [communities[a] for a in common]
                    a2 = [baseline_assignments[a] for a in common]
                    ari = adjusted_rand_score(a1, a2)
                    nmi = normalized_mutual_info_score(a1, a2)
            elif res == 1.0:
                ari = 1.0
                nmi = 1.0

            mean_sil = _compute_silhouette(window, communities, labels)

            result.points.append(SensitivityPoint(
                param_name="leiden_resolution",
                param_value=res,
                mean_cmi=0.0,
                std_cmi=0.0,
                mean_tds=0.0,
                std_tds=0.0,
                n_clusters_mode=n_clusters,
                ari_vs_baseline=ari,
                nmi_vs_baseline=nmi,
                mean_silhouette=mean_sil,
            ))
        except Exception as e:
            logger.error(f"resolution={res} failed: {e}")

    # Find optimal by silhouette
    if result.points:
        best = max(result.points, key=lambda p: p.mean_silhouette)
        result.conclusion = (
            f"Optimal resolution by silhouette: {best.param_value} "
            f"(silhouette={best.mean_silhouette:.3f}, "
            f"{best.n_clusters_mode} clusters)"
        )

    return result


def sweep_tail_quantile(
    returns: pd.DataFrame,
    quantiles: list[float] | None = None,
    window_size: int = 120,
) -> SensitivityResult:
    """Sweep tail dependence quantile threshold."""
    from src.features.similarity import tail_dependence_matrix

    if quantiles is None:
        quantiles = [0.01, 0.03, 0.05, 0.10]

    result = SensitivityResult(
        param_name="tail_quantile",
        baseline_value=0.05,
    )

    window = returns.iloc[-window_size:]
    baseline_td = None

    for q in quantiles:
        logger.info(f"Sensitivity: tail_quantile={q}")

        try:
            td = tail_dependence_matrix(window, quantile=q)

            if q == 0.05:
                baseline_td = td

            # Measure deviation from baseline
            if baseline_td is not None:
                frobenius = np.linalg.norm(td - baseline_td, "fro")
                max_norm = np.linalg.norm(baseline_td, "fro")
                relative_diff = frobenius / max(max_norm, 1e-10)
            else:
                relative_diff = 0.0

            # Mean off-diagonal tail dependence
            mask = ~np.eye(td.shape[0], dtype=bool)
            mean_td = td[mask].mean()
            std_td = td[mask].std()

            result.points.append(SensitivityPoint(
                param_name="tail_quantile",
                param_value=q,
                mean_cmi=mean_td,  # repurpose: mean tail dependence
                std_cmi=std_td,
                mean_tds=relative_diff,  # repurpose: relative diff from baseline
                std_tds=0.0,
                n_clusters_mode=0,
                ari_vs_baseline=1.0 - relative_diff,
                nmi_vs_baseline=0.0,
                mean_silhouette=0.0,
            ))
        except Exception as e:
            logger.error(f"quantile={q} failed: {e}")

    if result.points:
        max_diff = max(p.mean_tds for p in result.points)
        if max_diff < 0.1:
            result.conclusion = f"ROBUST: tail dependence stable (max relative diff = {max_diff:.3f})"
        else:
            result.conclusion = f"SENSITIVE: tail dependence varies (max relative diff = {max_diff:.3f})"

    return result


def _compute_silhouette(
    returns: pd.DataFrame,
    assignments: dict[str, int],
    labels: list[str],
) -> float:
    """Compute silhouette score for a clustering assignment."""
    from sklearn.metrics import silhouette_score
    from src.features.similarity import shrinkage_correlation

    if not assignments or len(set(assignments.values())) < 2:
        return 0.0

    # Align labels
    common = [l for l in labels if l in assignments]
    if len(common) < 4:
        return 0.0

    cluster_labels = [assignments[l] for l in common]
    if len(set(cluster_labels)) < 2:
        return 0.0

    # Use correlation distance
    idx = [labels.index(l) for l in common]
    sub_returns = returns.iloc[:, idx]

    try:
        corr = shrinkage_correlation(sub_returns)
        dist = np.sqrt(2 * np.clip(1 - corr, 0, 2))  # angular distance
        np.fill_diagonal(dist, 0)
        return float(silhouette_score(dist, cluster_labels, metric="precomputed"))
    except Exception:
        return 0.0


def run_full_sensitivity(
    returns: pd.DataFrame,
    seed: int = 42,
) -> dict[str, SensitivityResult]:
    """Run all sensitivity sweeps and return results."""
    results = {}

    logger.info("=== Sensitivity Analysis: Window Size ===")
    results["window_size"] = sweep_window_size(returns, seed=seed)

    logger.info("=== Sensitivity Analysis: Top-K ===")
    results["top_k"] = sweep_top_k(returns, seed=seed)

    logger.info("=== Sensitivity Analysis: Leiden Resolution ===")
    results["leiden_resolution"] = sweep_leiden_resolution(returns, seed=seed)

    logger.info("=== Sensitivity Analysis: Tail Quantile ===")
    results["tail_quantile"] = sweep_tail_quantile(returns)

    for name, res in results.items():
        logger.info(f"{name}: {res.conclusion}")

    return results
