"""Small-sample robustness: surrogate data testing and block bootstrap.

Phase 4.5 — Addresses the concern that regime-conditional metrics
(e.g., TE during the Twelve-Day War ~8 trading days) use very small
sub-samples that may not support reliable inference.

Methods:
1. Surrogate data testing (phase-randomized surrogates)
   - Shuffles time series while preserving power spectrum
   - Recomputes TE on surrogates to build null distribution
   - Tests if observed TE is significantly above null

2. Stationary block bootstrap (Politis & Romano 1994)
   - Random block length (geometric distribution)
   - Preserves higher-order temporal structure better than fixed-block

3. Minimum sample size estimation
   - Monte Carlo power analysis for TE and Granger
   - Reports minimum n for reliable inference at given effect size
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SurrogateTestResult:
    """Result of surrogate data significance test."""
    metric_name: str
    observed_value: float
    surrogate_mean: float
    surrogate_std: float
    surrogate_p_value: float  # fraction of surrogates >= observed
    n_surrogates: int
    ci_lower_95: float
    ci_upper_95: float
    significant: bool  # p < alpha
    effect_size: float  # (observed - mean) / std


@dataclass
class MinSampleResult:
    """Minimum sample size for reliable inference."""
    metric_name: str
    min_n_80_power: int  # n for 80% power
    min_n_90_power: int  # n for 90% power
    current_n: int
    adequate_80: bool
    adequate_90: bool


def phase_randomize_surrogate(
    series: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a phase-randomized surrogate of a time series.

    Preserves the power spectrum (autocorrelation structure) but
    destroys any causal/nonlinear relationships. This is the standard
    null hypothesis for testing TE significance: H0 = "observed TE
    is explainable by linear autocorrelation alone."

    Method: Theiler et al. (1992) — randomize Fourier phases.
    """
    n = len(series)
    # FFT
    fft_vals = np.fft.rfft(series)
    magnitudes = np.abs(fft_vals)
    # Random phases (preserve DC and Nyquist)
    phases = rng.uniform(0, 2 * np.pi, size=len(fft_vals))
    phases[0] = 0  # DC component: no phase shift
    if n % 2 == 0:
        phases[-1] = 0  # Nyquist: must be real
    # Reconstruct with randomized phases
    surrogate_fft = magnitudes * np.exp(1j * phases)
    surrogate = np.fft.irfft(surrogate_fft, n=n)
    return surrogate


def iaaft_surrogate(
    series: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.

    Preserves BOTH the power spectrum AND the marginal distribution
    of the original series. Stronger null than simple phase randomization.

    Method: Schreiber & Schmitz (1996).
    """
    n = len(series)
    sorted_values = np.sort(series)
    target_spectrum = np.abs(np.fft.rfft(series))

    # Initialize with random shuffle
    surrogate = rng.permutation(series).copy()

    for iteration in range(max_iter):
        # Step 1: Match power spectrum
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        adjusted_fft = target_spectrum * np.exp(1j * surr_phases)
        surrogate_new = np.fft.irfft(adjusted_fft, n=n)

        # Step 2: Match marginal distribution (rank-order matching)
        ranks = np.argsort(np.argsort(surrogate_new))
        surrogate_ranked = sorted_values[ranks]

        # Check convergence
        if np.max(np.abs(surrogate_ranked - surrogate)) < tol:
            return surrogate_ranked

        surrogate = surrogate_ranked

    return surrogate


def surrogate_te_test(
    source: np.ndarray,
    target: np.ndarray,
    n_surrogates: int = 199,
    lag: int = 1,
    k: int = 5,
    method: str = "phase",
    seed: int = 42,
    alpha: float = 0.05,
) -> SurrogateTestResult:
    """Test TE significance against surrogate null distribution.

    Generates surrogate versions of the SOURCE series (preserving its
    autocorrelation but destroying causal coupling to target), computes
    TE for each, and tests if observed TE exceeds the null distribution.

    Parameters
    ----------
    method : str
        "phase" for simple phase randomization, "iaaft" for IAAFT
        (preserves both spectrum and distribution).
    n_surrogates : int
        Number of surrogate series. Use 199 for p=0.005 resolution,
        999 for p=0.001 resolution.
    """
    from src.features.lead_lag import transfer_entropy_pair

    rng = np.random.default_rng(seed)

    # Observed TE
    observed_te = transfer_entropy_pair(source, target, lag=lag, k=k)

    # Generate surrogate distribution
    surrogate_tes = np.zeros(n_surrogates)
    surrogate_fn = iaaft_surrogate if method == "iaaft" else phase_randomize_surrogate

    for i in range(n_surrogates):
        surr_source = surrogate_fn(source, rng)
        surrogate_tes[i] = transfer_entropy_pair(surr_source, target, lag=lag, k=k)

    # p-value: fraction of surrogates >= observed (one-sided test)
    p_value = (np.sum(surrogate_tes >= observed_te) + 1) / (n_surrogates + 1)

    # Effect size (standardized)
    surr_std = np.std(surrogate_tes)
    effect_size = (
        (observed_te - np.mean(surrogate_tes)) / surr_std
        if surr_std > 1e-10
        else 0.0
    )

    return SurrogateTestResult(
        metric_name="transfer_entropy",
        observed_value=observed_te,
        surrogate_mean=np.mean(surrogate_tes),
        surrogate_std=surr_std,
        surrogate_p_value=p_value,
        n_surrogates=n_surrogates,
        ci_lower_95=np.percentile(surrogate_tes, 2.5),
        ci_upper_95=np.percentile(surrogate_tes, 97.5),
        significant=p_value < alpha,
        effect_size=effect_size,
    )


def surrogate_te_matrix(
    returns: pd.DataFrame,
    n_surrogates: int = 199,
    lag: int = 1,
    k: int = 5,
    method: str = "phase",
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Surrogate-corrected TE matrix with significance flags.

    Returns:
        te_pvalues: DataFrame of surrogate p-values for each pair
        te_significant: DataFrame of boolean significance flags
    """
    labels = returns.columns.tolist()
    n = len(labels)
    values = returns.values

    pvalues = np.ones((n, n))
    significant = np.zeros((n, n), dtype=bool)

    total_pairs = n * (n - 1)
    computed = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            computed += 1
            if computed % 50 == 0:
                logger.info(f"Surrogate TE: {computed}/{total_pairs} pairs")

            result = surrogate_te_test(
                source=values[:, i],
                target=values[:, j],
                n_surrogates=n_surrogates,
                lag=lag,
                k=k,
                method=method,
                seed=seed + i * n + j,
                alpha=alpha,
            )
            pvalues[i, j] = result.surrogate_p_value
            significant[i, j] = result.significant

    return (
        pd.DataFrame(pvalues, index=labels, columns=labels),
        pd.DataFrame(significant, index=labels, columns=labels),
    )


def stationary_block_bootstrap_indices(
    n: int,
    mean_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano 1994).

    Block lengths follow geometric distribution with given mean.
    This makes the resampled series strictly stationary, unlike
    fixed-block bootstrap which introduces edge effects.
    """
    p = 1.0 / mean_block_size  # geometric parameter
    indices = []
    pos = rng.integers(0, n)

    while len(indices) < n:
        indices.append(pos % n)
        # With probability p, jump to random new position
        if rng.random() < p:
            pos = rng.integers(0, n)
        else:
            pos += 1

    return np.array(indices[:n])


def minimum_sample_size_te(
    source: np.ndarray,
    target: np.ndarray,
    sample_sizes: list[int] | None = None,
    n_simulations: int = 200,
    n_surrogates: int = 99,
    lag: int = 1,
    k: int = 5,
    alpha: float = 0.05,
    seed: int = 42,
) -> MinSampleResult:
    """Monte Carlo power analysis for transfer entropy.

    For each candidate sample size n:
    1. Draw n consecutive observations from the data
    2. Compute TE and test against surrogates
    3. Record rejection rate (power)

    Returns minimum n for 80% and 90% power.
    """
    from src.features.lead_lag import transfer_entropy_pair

    rng = np.random.default_rng(seed)

    if sample_sizes is None:
        sample_sizes = [30, 50, 75, 100, 150, 200, 300, 500]

    full_n = len(source)
    power_at_n = {}

    for n_sample in sample_sizes:
        if n_sample > full_n:
            continue

        rejections = 0
        valid_sims = 0

        for sim in range(n_simulations):
            # Random contiguous subsample
            start = rng.integers(0, full_n - n_sample)
            sub_source = source[start:start + n_sample]
            sub_target = target[start:start + n_sample]

            observed_te = transfer_entropy_pair(sub_source, sub_target, lag=lag, k=k)

            # Quick surrogate test
            surr_tes = np.zeros(n_surrogates)
            for s in range(n_surrogates):
                surr = phase_randomize_surrogate(sub_source, rng)
                surr_tes[s] = transfer_entropy_pair(surr, sub_target, lag=lag, k=k)

            p_val = (np.sum(surr_tes >= observed_te) + 1) / (n_surrogates + 1)
            if p_val < alpha:
                rejections += 1
            valid_sims += 1

        power = rejections / max(valid_sims, 1)
        power_at_n[n_sample] = power
        logger.info(f"Power at n={n_sample}: {power:.2%} ({rejections}/{valid_sims})")

    # Find minimum n for target power levels
    min_80 = max(sample_sizes)
    min_90 = max(sample_sizes)
    for n_sample in sorted(power_at_n.keys()):
        if power_at_n[n_sample] >= 0.80 and n_sample < min_80:
            min_80 = n_sample
        if power_at_n[n_sample] >= 0.90 and n_sample < min_90:
            min_90 = n_sample

    return MinSampleResult(
        metric_name="transfer_entropy",
        min_n_80_power=min_80,
        min_n_90_power=min_90,
        current_n=full_n,
        adequate_80=full_n >= min_80,
        adequate_90=full_n >= min_90,
    )
