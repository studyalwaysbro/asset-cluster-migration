"""Multi-layer similarity/dependency computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


def shrinkage_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimator for correlation matrix."""
    lw = LedoitWolf()
    lw.fit(returns.values)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0] = 1.0  # avoid division by zero
    corr = cov / outer
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def rank_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Spearman rank correlation matrix."""
    corr, _ = stats.spearmanr(returns.values)
    if corr.ndim == 0:
        return np.array([[1.0]])
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def distance_correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
    """Pairwise distance correlation (dCor). Captures nonlinear dependence."""
    import dcor

    n = returns.shape[1]
    result = np.ones((n, n))
    values = returns.values

    for i in range(n):
        for j in range(i + 1, n):
            dc = dcor.distance_correlation(values[:, i], values[:, j])
            result[i, j] = dc
            result[j, i] = dc

    return result


def mutual_information_matrix(
    returns: pd.DataFrame, k: int = 5
) -> np.ndarray:
    """KSG k-NN mutual information estimator (pairwise)."""
    from sklearn.feature_selection import mutual_info_regression

    n = returns.shape[1]
    result = np.zeros((n, n))
    values = returns.values

    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_info_regression(
                values[:, i].reshape(-1, 1),
                values[:, j],
                n_neighbors=k,
                random_state=42,
            )[0]
            result[i, j] = mi
            result[j, i] = mi

    # Normalize to [0, 1] range
    max_mi = result.max()
    if max_mi > 0:
        result = result / max_mi

    np.fill_diagonal(result, 1.0)
    return result


def tail_dependence_matrix(
    returns: pd.DataFrame, quantile: float = 0.05
) -> np.ndarray:
    """Empirical lower-tail dependence coefficients."""
    n = returns.shape[1]
    result = np.zeros((n, n))
    values = returns.values
    threshold_idx = int(quantile * len(values))

    for i in range(n):
        for j in range(i + 1, n):
            # Empirical co-exceedance below quantile
            q_i = np.sort(values[:, i])[threshold_idx]
            q_j = np.sort(values[:, j])[threshold_idx]
            joint_exceedance = np.mean(
                (values[:, i] <= q_i) & (values[:, j] <= q_j)
            )
            marginal = quantile
            if marginal > 0:
                td = joint_exceedance / marginal
            else:
                td = 0.0
            result[i, j] = td
            result[j, i] = td

    np.fill_diagonal(result, 1.0)
    return np.clip(result, 0.0, 1.0)


def compute_all_layers(
    returns: pd.DataFrame,
    layers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute all requested similarity layers for a single window."""
    available = {
        "pearson_shrinkage": shrinkage_correlation,
        "spearman": rank_correlation,
        "distance_correlation": distance_correlation_matrix,
        "mutual_information": mutual_information_matrix,
        "tail_dependence": tail_dependence_matrix,
    }

    if layers is None:
        layers = list(available.keys())

    result = {}
    for layer_name in layers:
        if layer_name in available:
            result[layer_name] = available[layer_name](returns)

    return result
