"""Regime segmentation orchestrator."""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.regimes.hmm import MarketRegimeDetector
from src.regimes.changepoint import detect_changepoints


def build_regime_features(
    returns: pd.DataFrame, window: int = 21
) -> pd.DataFrame:
    """Build regime state vector from returns."""
    realized_vol = returns.std(axis=1).rolling(window).mean()
    mean_corr = returns.rolling(window).corr().groupby(level=0).mean().mean(axis=1)
    dispersion = returns.std(axis=0).mean()  # cross-sectional

    features = pd.DataFrame({
        "vol": realized_vol,
        "mean_corr": mean_corr,
        "dispersion": dispersion,
    }).dropna()

    return features
