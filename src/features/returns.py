"""Return computation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from adjusted close prices."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage returns."""
    return prices.pct_change().dropna()


def compute_excess_returns(
    returns: pd.DataFrame, risk_free: pd.Series
) -> pd.DataFrame:
    """Compute excess returns over risk-free rate."""
    aligned_rf = risk_free.reindex(returns.index).ffill()
    daily_rf = aligned_rf / 252
    return returns.sub(daily_rf, axis=0)


def winsorize_returns(
    returns: pd.DataFrame, limits: tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """Winsorize returns at specified quantiles."""
    lower = returns.quantile(limits[0])
    upper = returns.quantile(limits[1])
    return returns.clip(lower=lower, upper=upper, axis=1)
