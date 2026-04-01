"""Tests for regime detection."""
import numpy as np
import pandas as pd
from src.regimes.changepoint import detect_changepoints


def test_changepoint_detection():
    np.random.seed(42)
    s1 = np.random.normal(0, 1, 100)
    s2 = np.random.normal(3, 1, 100)
    values = np.concatenate([s1, s2])
    dates = pd.bdate_range("2023-01-01", periods=200)
    series = pd.Series(values, index=dates)
    cps = detect_changepoints(series, penalty=10.0)
    assert len(cps) >= 1
