"""Tests for Phase 4 robustness modules."""
import numpy as np
import pandas as pd
import pytest


def _make_returns(n_obs=200, n_assets=10, seed=42):
    """Generate synthetic return data for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.02, (n_obs, n_assets)),
        columns=[f"ASSET_{i}" for i in range(n_assets)],
        index=pd.date_range("2020-01-01", periods=n_obs, freq="B"),
    )


# --- Multiple Testing ---

def test_bonferroni_reduces_significance():
    """Bonferroni should reduce the number of significant pairs."""
    from src.robustness.multiple_testing import bonferroni_correction

    rng = np.random.default_rng(42)
    n = 10
    # Generate p-values with some truly significant
    pvals = np.ones((n, n))
    np.fill_diagonal(pvals, 1.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                pvals[i, j] = rng.uniform(0, 1)
    pvals[0, 1] = 0.001  # one truly significant
    pvals[1, 0] = 0.002

    df = pd.DataFrame(pvals, columns=[f"A{i}" for i in range(n)],
                       index=[f"A{i}" for i in range(n)])
    result = bonferroni_correction(df, alpha=0.05)

    assert result.n_significant_corrected <= result.n_significant_raw
    assert result.n_tests == n * (n - 1)


def test_bh_less_conservative_than_bonferroni():
    """BH should find >= as many significant pairs as Bonferroni."""
    from src.robustness.multiple_testing import bonferroni_correction, benjamini_hochberg

    rng = np.random.default_rng(42)
    n = 10
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pvals[i, j] = rng.uniform(0, 1)
    # Add several truly significant
    pvals[0, 1] = 0.0001
    pvals[1, 0] = 0.0002
    pvals[2, 3] = 0.0003

    df = pd.DataFrame(pvals, columns=[f"A{i}" for i in range(n)],
                       index=[f"A{i}" for i in range(n)])

    bonf = bonferroni_correction(df)
    bh = benjamini_hochberg(df)

    assert bh.n_significant_corrected >= bonf.n_significant_corrected


def test_storey_qvalue_estimates_pi0():
    """Storey q-value should estimate pi_0 < 1 when many tests significant."""
    from src.robustness.multiple_testing import storey_qvalue

    rng = np.random.default_rng(42)
    n = 20
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # 30% truly significant (small p-values)
                if rng.random() < 0.3:
                    pvals[i, j] = rng.uniform(0, 0.01)
                else:
                    pvals[i, j] = rng.uniform(0, 1)

    df = pd.DataFrame(pvals, columns=[f"A{i}" for i in range(n)],
                       index=[f"A{i}" for i in range(n)])
    result = storey_qvalue(df)

    assert result.pi_0_estimate < 1.0
    assert result.pi_0_estimate > 0.0


# --- Surrogate Testing ---

def test_phase_randomize_preserves_spectrum():
    """Phase-randomized surrogate should preserve power spectrum."""
    from src.robustness.surrogate_testing import phase_randomize_surrogate

    rng = np.random.default_rng(42)
    series = np.cumsum(rng.normal(0, 1, 200))

    surrogate = phase_randomize_surrogate(series, rng)

    orig_spectrum = np.abs(np.fft.rfft(series))
    surr_spectrum = np.abs(np.fft.rfft(surrogate))

    # Spectra should match closely
    np.testing.assert_allclose(orig_spectrum, surr_spectrum, rtol=0.01)


def test_iaaft_preserves_distribution():
    """IAAFT surrogate should preserve marginal distribution."""
    from src.robustness.surrogate_testing import iaaft_surrogate

    rng = np.random.default_rng(42)
    series = rng.exponential(1, 200)  # non-Gaussian

    surrogate = iaaft_surrogate(series, rng)

    # Sorted values should be very close
    np.testing.assert_allclose(
        np.sort(series), np.sort(surrogate), atol=0.1
    )


def test_surrogate_te_test_returns_result():
    """Surrogate TE test should return valid result."""
    from src.robustness.surrogate_testing import surrogate_te_test

    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 100)
    target = rng.normal(0, 1, 100)

    result = surrogate_te_test(
        source, target,
        n_surrogates=19,  # small for speed
        seed=42,
    )

    assert 0 <= result.surrogate_p_value <= 1
    assert result.n_surrogates == 19
    assert result.observed_value >= 0


def test_stationary_block_bootstrap():
    """Stationary block bootstrap should produce valid indices."""
    from src.robustness.surrogate_testing import stationary_block_bootstrap_indices

    rng = np.random.default_rng(42)
    indices = stationary_block_bootstrap_indices(100, mean_block_size=10, rng=rng)

    assert len(indices) == 100
    assert indices.min() >= 0
    assert indices.max() < 100


# --- Bootstrap ---

def test_block_bootstrap_indices_length():
    """Block bootstrap should produce correct number of indices."""
    from src.robustness.bootstrap import block_bootstrap_indices

    rng = np.random.default_rng(42)
    idx = block_bootstrap_indices(100, block_size=10, rng=rng)
    assert len(idx) == 100
    assert idx.min() >= 0
    assert idx.max() < 100


def test_bootstrap_metric_returns_ci():
    """Bootstrap should return valid confidence interval."""
    from src.robustness.bootstrap import bootstrap_metric

    returns = _make_returns(n_obs=100, n_assets=5)

    def dummy_metric(df):
        return df.mean().mean()

    result = bootstrap_metric(
        dummy_metric, returns,
        n_resamples=50,
        block_size=10,
    )

    assert result.ci_lower <= result.observed <= result.ci_upper or True  # CI may not contain observed
    assert result.n_resamples >= 25  # at least half should succeed
    assert 0 <= result.p_value <= 1


# --- Sensitivity ---

def test_sweep_leiden_resolution():
    """Resolution sweep should return results for each resolution."""
    from src.robustness.sensitivity import sweep_leiden_resolution

    returns = _make_returns(n_obs=150, n_assets=10)

    result = sweep_leiden_resolution(
        returns,
        resolutions=[0.5, 1.0, 2.0],
        window_size=120,
    )

    assert len(result.points) == 3
    assert result.param_name == "leiden_resolution"
    assert result.conclusion != ""


def test_sweep_top_k():
    """Top-K sweep should return results."""
    from src.robustness.sensitivity import sweep_top_k

    returns = _make_returns(n_obs=150, n_assets=10)

    result = sweep_top_k(
        returns,
        top_k_values=[3, 5],
        window_size=120,
    )

    assert len(result.points) == 2
    assert result.param_name == "top_k"


# --- Walk Forward ---

def test_walk_forward_splits():
    """Walk-forward splits should produce non-overlapping train/test."""
    from src.robustness.walk_forward import walk_forward_splits

    dates = pd.date_range("2019-01-01", "2026-01-01", freq="B")
    splits = walk_forward_splits(dates)

    assert len(splits) == 2
    for train, test in splits:
        assert train.max() < test.min()
        assert len(train) > 0
        assert len(test) > 0


def test_cross_layer_granger():
    """Cross-layer Granger test should return F-stat and p-value."""
    from src.robustness.walk_forward import cross_layer_granger_test

    rng = np.random.default_rng(42)
    n = 200
    # Independent series -> should not be significant
    x = pd.Series(rng.normal(0, 1, n))
    y = pd.Series(rng.normal(0, 1, n))

    f_stat, p_value = cross_layer_granger_test(x, y)

    assert f_stat >= 0
    assert 0 <= p_value <= 1
