"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    n_days, n_assets = 252, 10
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    cov = np.eye(n_assets) * 0.0004
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            cov[i, j] = cov[j, i] = 0.0002 * np.exp(-abs(i-j) * 0.3)
    returns = np.random.multivariate_normal(np.zeros(n_assets), cov, n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


@pytest.fixture
def sample_similarity_matrix():
    np.random.seed(42)
    n = 10
    A = np.random.randn(100, n)
    return np.corrcoef(A.T)
