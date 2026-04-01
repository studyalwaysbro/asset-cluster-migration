"""Tests for similarity computation."""
import numpy as np


def test_shrinkage_correlation_shape(sample_returns):
    from src.features.similarity import shrinkage_correlation
    result = shrinkage_correlation(sample_returns)
    n = sample_returns.shape[1]
    assert result.shape == (n, n)
    assert np.allclose(np.diag(result), 1.0)


def test_shrinkage_correlation_symmetric(sample_returns):
    from src.features.similarity import shrinkage_correlation
    result = shrinkage_correlation(sample_returns)
    assert np.allclose(result, result.T, atol=1e-10)


def test_rank_correlation_bounds(sample_returns):
    from src.features.similarity import rank_correlation
    result = rank_correlation(sample_returns)
    assert result.min() >= -1.0
    assert result.max() <= 1.0
