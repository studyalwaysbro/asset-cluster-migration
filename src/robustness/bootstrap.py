"""Bootstrap confidence intervals for migration and causality metrics.

Phase 4.2 — Provides statistical significance testing for:
1. CMI / TDS / AMF — are observed values significantly different from noise?
2. Transfer entropy rankings — are leaders/followers stable across resamples?
3. Cross-layer Granger F-statistic — is p=0.041 robust to resampling?

Uses block bootstrap to preserve temporal autocorrelation structure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    observed: float
    mean: float
    std: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    p_value: float  # fraction of bootstrap samples <= 0
    n_resamples: int = 1000
    significant: bool = False  # CI excludes 0


@dataclass
class TEStabilityResult:
    """Transfer entropy leadership stability across bootstrap resamples."""
    asset: str
    mean_rank: float
    std_rank: float
    rank_ci_lower: float
    rank_ci_upper: float
    pct_top5: float  # fraction of resamples where asset is top-5 leader
    pct_top10: float  # fraction of resamples where asset is top-10 leader
    stable: bool  # rank CI width < n_assets / 3


def block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate block bootstrap sample indices.
    
    Preserves temporal autocorrelation by resampling contiguous blocks.
    Uses the circular block bootstrap (Politis & Romano 1992).
    """
    n_blocks = int(np.ceil(n / block_size))
    # Random block starting positions (circular)
    starts = rng.integers(0, n, size=n_blocks)
    
    indices = []
    for s in starts:
        block = np.arange(s, s + block_size) % n  # circular wrap
        indices.extend(block)
    
    return np.array(indices[:n])


def bootstrap_metric(
    compute_fn,
    data: pd.DataFrame,
    n_resamples: int = 1000,
    block_size: int = 20,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Bootstrap confidence interval for any scalar metric.
    
    Parameters
    ----------
    compute_fn : callable
        Function that takes a DataFrame and returns a float.
    data : DataFrame
        Input data (rows = time, columns = assets).
    n_resamples : int
        Number of bootstrap resamples.
    block_size : int
        Block size for block bootstrap (preserves autocorrelation).
    seed : int
        Random seed for reproducibility.
    confidence : float
        Confidence level (default 95%).
    """
    rng = np.random.default_rng(seed)
    observed = compute_fn(data)
    
    boot_values = np.zeros(n_resamples)
    n = len(data)
    
    for b in range(n_resamples):
        idx = block_bootstrap_indices(n, block_size, rng)
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            boot_values[b] = compute_fn(boot_data)
        except Exception:
            boot_values[b] = np.nan
    
    boot_values = boot_values[~np.isnan(boot_values)]
    if len(boot_values) < n_resamples * 0.5:
        logger.warning(
            f"Only {len(boot_values)}/{n_resamples} bootstrap samples succeeded"
        )
    
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot_values, alpha * 100)
    ci_upper = np.percentile(boot_values, (1 - alpha) * 100)
    
    # P-value: fraction of bootstrap distribution <= 0
    p_value = np.mean(boot_values <= 0)
    
    return BootstrapCI(
        observed=observed,
        mean=np.mean(boot_values),
        std=np.std(boot_values),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_resamples=len(boot_values),
        significant=(ci_lower > 0 or ci_upper < 0),
    )


def bootstrap_te_rankings(
    returns: pd.DataFrame,
    n_resamples: int = 1000,
    block_size: int = 20,
    seed: int = 42,
    te_lag: int = 1,
    te_k: int = 5,
) -> list[TEStabilityResult]:
    """Bootstrap stability of transfer entropy leadership rankings.
    
    For each resample:
    1. Block-bootstrap the return series
    2. Recompute the full TE matrix
    3. Record each asset's net-flow rank
    
    Returns stability metrics: mean rank, rank CI, fraction in top-5/top-10.
    """
    from src.features.lead_lag import transfer_entropy_matrix, information_flow_ranking
    
    rng = np.random.default_rng(seed)
    labels = returns.columns.tolist()
    n_assets = len(labels)
    n_obs = len(returns)
    
    # Storage: ranks[asset_idx, resample]
    ranks = np.full((n_assets, n_resamples), np.nan)
    
    for b in range(n_resamples):
        if (b + 1) % 100 == 0:
            logger.info(f"TE bootstrap: {b + 1}/{n_resamples}")
        
        idx = block_bootstrap_indices(n_obs, block_size, rng)
        boot_returns = returns.iloc[idx].reset_index(drop=True)
        
        try:
            te = transfer_entropy_matrix(boot_returns, lag=te_lag, k=te_k)
            ranking = information_flow_ranking(te)
            
            for asset_idx, asset in enumerate(labels):
                if asset in ranking.index:
                    # Rank by net_flow descending (rank 1 = biggest leader)
                    sorted_assets = ranking.sort_values(
                        "net_flow", ascending=False
                    ).index.tolist()
                    ranks[asset_idx, b] = sorted_assets.index(asset) + 1
        except Exception:
            continue
    
    # Compute stability metrics
    results = []
    for asset_idx, asset in enumerate(labels):
        r = ranks[asset_idx]
        valid = r[~np.isnan(r)]
        if len(valid) < 10:
            continue
        
        results.append(TEStabilityResult(
            asset=asset,
            mean_rank=np.mean(valid),
            std_rank=np.std(valid),
            rank_ci_lower=np.percentile(valid, 2.5),
            rank_ci_upper=np.percentile(valid, 97.5),
            pct_top5=np.mean(valid <= 5),
            pct_top10=np.mean(valid <= 10),
            stable=(np.percentile(valid, 97.5) - np.percentile(valid, 2.5))
                   < n_assets / 3,
        ))
    
    results.sort(key=lambda x: x.mean_rank)
    return results


def bootstrap_granger_f_stat(
    cmi_tail: pd.Series,
    cmi_pearson: pd.Series,
    n_resamples: int = 1000,
    block_size: int = 20,
    max_lag: int = 5,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap the cross-layer Granger F-statistic.
    
    Tests robustness of the key finding that tail-dependence CMI
    Granger-causes Pearson CMI (in-sample p=0.041).
    """
    from src.robustness.walk_forward import cross_layer_granger_test
    
    combined = pd.DataFrame({
        "cmi_tail": cmi_tail,
        "cmi_pearson": cmi_pearson,
    }).dropna()
    
    def compute_f_stat(df):
        f, _ = cross_layer_granger_test(
            df["cmi_tail"], df["cmi_pearson"], max_lag=max_lag
        )
        return f
    
    return bootstrap_metric(
        compute_fn=compute_f_stat,
        data=combined,
        n_resamples=n_resamples,
        block_size=block_size,
        seed=seed,
    )
