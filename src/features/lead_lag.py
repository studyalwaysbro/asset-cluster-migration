"""Lead-lag, transfer entropy, and Granger causality computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import KDTree


def cross_correlation_matrix(
    returns: pd.DataFrame, max_lag: int = 5
) -> dict[int, np.ndarray]:
    """Compute cross-correlation matrices at multiple lags."""
    n = returns.shape[1]
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x = returns.iloc[:, i].values
                y = returns.iloc[:, j].values
                if lag > 0:
                    corr[i, j] = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                else:
                    corr[i, j] = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        results[lag] = corr
    return results


def lead_lag_score(returns: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Compute net lead-lag score for each asset pair.

    Positive score(i,j) means asset i leads asset j.
    """
    n = returns.shape[1]
    labels = returns.columns.tolist()
    cc = cross_correlation_matrix(returns, max_lag)

    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lead_sum = sum(abs(cc[lag][i, j]) for lag in range(1, max_lag + 1))
            follow_sum = sum(abs(cc[-lag][i, j]) for lag in range(1, max_lag + 1))
            scores[i, j] = lead_sum - follow_sum

    return pd.DataFrame(scores, index=labels, columns=labels)


def transfer_entropy_pair(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
) -> float:
    """Estimate transfer entropy from source to target using KSG estimator.

    TE(X->Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    Estimated via k-nearest-neighbor conditional MI.
    """
    n = len(target) - lag
    if n < k + 5:
        return 0.0

    # Build joint embedding: [Y_t, Y_{t-lag}, X_{t-lag}]
    y_t = target[lag:].reshape(-1, 1)
    y_past = target[:-lag].reshape(-1, 1)
    x_past = source[:-lag].reshape(-1, 1)

    # Normalize
    for arr in [y_t, y_past, x_past]:
        std = arr.std()
        if std > 0:
            arr[:] = (arr - arr.mean()) / std

    # Joint space [Y_t, Y_past, X_past]
    joint = np.hstack([y_t, y_past, x_past])
    # Marginal spaces
    yz = np.hstack([y_t, y_past])
    xz = np.hstack([x_past, y_past])
    z = y_past

    # KSG estimator for CMI: I(Y_t; X_past | Y_past)
    try:
        tree_joint = KDTree(joint, metric="chebyshev")
        dists, _ = tree_joint.query(joint, k=k + 1)
        eps = dists[:, -1]  # distance to k-th neighbor in joint space
        eps = np.maximum(eps, 1e-10)

        # Count neighbors within eps in marginal spaces
        tree_yz = KDTree(yz, metric="chebyshev")
        tree_xz = KDTree(xz, metric="chebyshev")
        tree_z = KDTree(z, metric="chebyshev")

        n_yz = np.array([tree_yz.query_radius(yz[i:i+1], r=eps[i], count_only=True)[0] - 1 for i in range(n)])
        n_xz = np.array([tree_xz.query_radius(xz[i:i+1], r=eps[i], count_only=True)[0] - 1 for i in range(n)])
        n_z = np.array([tree_z.query_radius(z[i:i+1], r=eps[i], count_only=True)[0] - 1 for i in range(n)])

        # Avoid log(0)
        n_yz = np.maximum(n_yz, 1)
        n_xz = np.maximum(n_xz, 1)
        n_z = np.maximum(n_z, 1)

        te = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(n_z))
        return max(te, 0.0)  # TE is non-negative
    except Exception:
        return 0.0


def transfer_entropy_matrix(
    returns: pd.DataFrame,
    lag: int = 1,
    k: int = 5,
) -> pd.DataFrame:
    """Compute pairwise transfer entropy matrix.

    TE[i,j] = transfer entropy from asset i to asset j (i leads j).
    """
    labels = returns.columns.tolist()
    n = len(labels)
    values = returns.values
    te = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            te[i, j] = transfer_entropy_pair(values[:, i], values[:, j], lag=lag, k=k)

    return pd.DataFrame(te, index=labels, columns=labels)


def net_transfer_entropy(te_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute net transfer entropy: NTE(i,j) = TE(i->j) - TE(j->i).

    Positive NTE means i is a net information sender to j.
    """
    return te_matrix - te_matrix.T


def granger_causality_matrix(
    returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Granger causality test.

    Returns:
        f_stat_matrix: F-statistics for each pair
        pvalue_matrix: p-values for each pair
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    labels = returns.columns.tolist()
    n = len(labels)
    f_stats = np.zeros((n, n))
    p_values = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                data = pd.DataFrame({
                    "target": returns.iloc[:, j].values,
                    "source": returns.iloc[:, i].values,
                })
                data = data.dropna()
                if len(data) < max_lag + 10:
                    continue
                result = grangercausalitytests(data[["target", "source"]], maxlag=max_lag, verbose=False)
                # Take minimum p-value across lags
                best_lag = min(result.keys(), key=lambda l: result[l][0]["ssr_ftest"][1])
                f_stats[i, j] = result[best_lag][0]["ssr_ftest"][0]
                p_values[i, j] = result[best_lag][0]["ssr_ftest"][1]
            except Exception:
                pass

    return (
        pd.DataFrame(f_stats, index=labels, columns=labels),
        pd.DataFrame(p_values, index=labels, columns=labels),
    )


def information_flow_ranking(
    te_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Rank assets by net information outflow.

    Assets with high outflow are market leaders; high inflow are followers.
    """
    labels = te_matrix.index.tolist()
    outflow = te_matrix.sum(axis=1)  # total TE sent
    inflow = te_matrix.sum(axis=0)   # total TE received
    net = outflow - inflow

    return pd.DataFrame({
        "outflow": outflow,
        "inflow": inflow,
        "net_flow": net,
        "leader_score": net / (outflow + inflow + 1e-10),
    }).sort_values("net_flow", ascending=False)
