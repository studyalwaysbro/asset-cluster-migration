"""Multiple testing correction for pairwise causality tests.

Phase 4.4 — With 91 assets there are 91*90 = 8,190 directed Granger pairs.
Testing all at alpha=0.05 without correction yields ~410 false positives.

Implements:
1. Bonferroni correction (family-wise error rate control)
2. Benjamini-Hochberg FDR (false discovery rate control)
3. Storey's q-value (adaptive FDR with pi_0 estimation)

Also provides aggregate significance testing: is the NUMBER of significant
pairs itself significant under the null of no causality?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MultipleTestingResult:
    """Results from multiple testing correction on a p-value matrix."""
    method: str
    n_tests: int
    n_significant_raw: int  # before correction
    n_significant_corrected: int  # after correction
    alpha: float
    # Corrected p-value matrix
    corrected_pvalues: pd.DataFrame | None = None
    # Binary significance matrix
    significant_mask: pd.DataFrame | None = None
    # Summary stats
    fdr_estimate: float = 0.0  # estimated false discovery rate
    pi_0_estimate: float = 1.0  # estimated proportion of true nulls
    # Aggregate test
    aggregate_p_value: float = 1.0  # p-value for "more significant than chance"
    aggregate_significant: bool = False


def bonferroni_correction(
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Bonferroni correction: multiply each p-value by number of tests.

    Controls family-wise error rate (FWER): probability of ANY false positive.
    Conservative but guaranteed: if Bonferroni says significant, it's real.
    """
    labels = pvalue_matrix.index.tolist()
    n = len(labels)
    n_tests = n * (n - 1)  # exclude diagonal

    # Extract off-diagonal p-values
    mask = ~np.eye(n, dtype=bool)
    raw_p = pvalue_matrix.values[mask]

    n_sig_raw = int(np.sum(raw_p < alpha))

    # Bonferroni: multiply all p-values by number of tests
    corrected = np.minimum(pvalue_matrix.values * n_tests, 1.0)
    np.fill_diagonal(corrected, 1.0)

    corrected_df = pd.DataFrame(corrected, index=labels, columns=labels)
    sig_mask = corrected_df < alpha

    n_sig_corrected = int(sig_mask.values[mask].sum())

    # Aggregate test: under H0 of no causality, number of significant pairs
    # follows Binomial(n_tests, alpha). Is observed count extreme?
    from scipy.stats import binom
    aggregate_p = 1.0 - binom.cdf(n_sig_raw - 1, n_tests, alpha)

    logger.info(
        f"Bonferroni: {n_sig_raw}/{n_tests} raw significant -> "
        f"{n_sig_corrected}/{n_tests} after correction"
    )

    return MultipleTestingResult(
        method="bonferroni",
        n_tests=n_tests,
        n_significant_raw=n_sig_raw,
        n_significant_corrected=n_sig_corrected,
        alpha=alpha,
        corrected_pvalues=corrected_df,
        significant_mask=sig_mask,
        aggregate_p_value=aggregate_p,
        aggregate_significant=aggregate_p < alpha,
    )


def benjamini_hochberg(
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Benjamini-Hochberg procedure for false discovery rate (FDR) control.

    Controls expected proportion of false discoveries among rejections.
    Less conservative than Bonferroni — more power to detect real effects.
    """
    labels = pvalue_matrix.index.tolist()
    n = len(labels)
    n_tests = n * (n - 1)

    # Extract off-diagonal p-values with indices
    mask = ~np.eye(n, dtype=bool)
    raw_p = pvalue_matrix.values[mask]
    n_sig_raw = int(np.sum(raw_p < alpha))

    # BH procedure
    sorted_idx = np.argsort(raw_p)
    sorted_p = raw_p[sorted_idx]
    ranks = np.arange(1, len(sorted_p) + 1)

    # BH critical values: (rank / n_tests) * alpha
    bh_critical = (ranks / n_tests) * alpha

    # Find largest k such that p_(k) <= bh_critical_(k)
    rejections = sorted_p <= bh_critical
    if rejections.any():
        max_rejection_idx = np.max(np.where(rejections))
        threshold = sorted_p[max_rejection_idx]
    else:
        threshold = 0.0

    # Adjusted p-values (Yekutieli & Benjamini 1999)
    adjusted_p = np.minimum(
        sorted_p * n_tests / ranks, 1.0
    )
    # Enforce monotonicity (adjusted p-values must be non-decreasing)
    for i in range(len(adjusted_p) - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

    # Map back to original order
    corrected_flat = np.ones(len(raw_p))
    corrected_flat[sorted_idx] = adjusted_p

    # Rebuild matrix
    corrected_matrix = np.ones((n, n))
    corrected_matrix[mask] = corrected_flat
    np.fill_diagonal(corrected_matrix, 1.0)

    corrected_df = pd.DataFrame(corrected_matrix, index=labels, columns=labels)
    sig_mask = corrected_df < alpha
    n_sig_corrected = int(sig_mask.values[mask].sum())

    # Estimate FDR
    if n_sig_corrected > 0:
        fdr_est = min(alpha * n_tests / n_sig_corrected, 1.0)
    else:
        fdr_est = 0.0

    # Aggregate test
    from scipy.stats import binom
    aggregate_p = 1.0 - binom.cdf(n_sig_raw - 1, n_tests, alpha)

    logger.info(
        f"BH-FDR: {n_sig_raw}/{n_tests} raw significant -> "
        f"{n_sig_corrected}/{n_tests} after correction "
        f"(est FDR = {fdr_est:.3f})"
    )

    return MultipleTestingResult(
        method="benjamini_hochberg",
        n_tests=n_tests,
        n_significant_raw=n_sig_raw,
        n_significant_corrected=n_sig_corrected,
        alpha=alpha,
        corrected_pvalues=corrected_df,
        significant_mask=sig_mask,
        fdr_estimate=fdr_est,
        aggregate_p_value=aggregate_p,
        aggregate_significant=aggregate_p < alpha,
    )


def storey_qvalue(
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
    lambda_range: np.ndarray | None = None,
) -> MultipleTestingResult:
    """Storey's q-value: adaptive FDR with pi_0 estimation.

    Estimates the proportion of true null hypotheses (pi_0) from the
    p-value distribution, then adjusts BH accordingly. More powerful
    than BH when many tests are truly non-null.
    """
    labels = pvalue_matrix.index.tolist()
    n = len(labels)
    n_tests = n * (n - 1)

    mask = ~np.eye(n, dtype=bool)
    raw_p = pvalue_matrix.values[mask]
    n_sig_raw = int(np.sum(raw_p < alpha))

    # Estimate pi_0 using Storey's bootstrap method
    if lambda_range is None:
        lambda_range = np.arange(0.05, 0.95, 0.05)

    pi_0_estimates = []
    for lam in lambda_range:
        pi_0_lam = np.mean(raw_p > lam) / (1 - lam)
        pi_0_estimates.append(min(pi_0_lam, 1.0))

    # Use the minimum of the smoothed estimates (conservative)
    pi_0 = min(pi_0_estimates) if pi_0_estimates else 1.0
    pi_0 = max(pi_0, 1 / n_tests)  # floor at 1/m

    # Compute q-values
    sorted_idx = np.argsort(raw_p)
    sorted_p = raw_p[sorted_idx]
    ranks = np.arange(1, len(sorted_p) + 1)

    # q-value = pi_0 * m * p / rank (with monotonicity enforcement)
    q_values = pi_0 * n_tests * sorted_p / ranks
    q_values = np.minimum(q_values, 1.0)

    # Enforce monotonicity
    for i in range(len(q_values) - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])

    # Map back
    corrected_flat = np.ones(len(raw_p))
    corrected_flat[sorted_idx] = q_values

    corrected_matrix = np.ones((n, n))
    corrected_matrix[mask] = corrected_flat
    np.fill_diagonal(corrected_matrix, 1.0)

    corrected_df = pd.DataFrame(corrected_matrix, index=labels, columns=labels)
    sig_mask = corrected_df < alpha
    n_sig_corrected = int(sig_mask.values[mask].sum())

    from scipy.stats import binom
    aggregate_p = 1.0 - binom.cdf(n_sig_raw - 1, n_tests, alpha)

    logger.info(
        f"Storey q-value: pi_0={pi_0:.3f}, "
        f"{n_sig_raw}/{n_tests} raw -> {n_sig_corrected}/{n_tests} corrected"
    )

    return MultipleTestingResult(
        method="storey_qvalue",
        n_tests=n_tests,
        n_significant_raw=n_sig_raw,
        n_significant_corrected=n_sig_corrected,
        alpha=alpha,
        corrected_pvalues=corrected_df,
        significant_mask=sig_mask,
        pi_0_estimate=pi_0,
        aggregate_p_value=aggregate_p,
        aggregate_significant=aggregate_p < alpha,
    )


def run_all_corrections(
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, MultipleTestingResult]:
    """Run Bonferroni, BH, and Storey corrections on a p-value matrix.

    Returns dict mapping method name to result.
    """
    results = {
        "bonferroni": bonferroni_correction(pvalue_matrix, alpha),
        "benjamini_hochberg": benjamini_hochberg(pvalue_matrix, alpha),
        "storey_qvalue": storey_qvalue(pvalue_matrix, alpha),
    }

    # Log comparison
    logger.info("=== Multiple Testing Correction Summary ===")
    for name, res in results.items():
        logger.info(
            f"  {name}: {res.n_significant_corrected}/{res.n_tests} "
            f"significant (aggregate p={res.aggregate_p_value:.4f})"
        )

    return results


def summarize_corrections(
    results: dict[str, MultipleTestingResult],
) -> pd.DataFrame:
    """Create publication-ready summary table."""
    records = []
    for name, res in results.items():
        records.append({
            "Method": name,
            "Total Tests": res.n_tests,
            "Significant (raw)": res.n_significant_raw,
            "Significant (corrected)": res.n_significant_corrected,
            "Survival Rate": (
                res.n_significant_corrected / max(res.n_significant_raw, 1)
            ),
            "Est. FDR": res.fdr_estimate,
            "Est. pi_0": res.pi_0_estimate,
            "Aggregate p-value": res.aggregate_p_value,
            "Aggregate Significant": res.aggregate_significant,
        })
    return pd.DataFrame(records)
