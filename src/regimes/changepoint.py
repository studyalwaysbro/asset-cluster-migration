"""Change-point detection using PELT algorithm.

v0.4.0 — Fixed BIC penalty formula to use proper Schwarz criterion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures


def detect_changepoints(
    series: pd.Series,
    model: str = "rbf",
    penalty: float | None = None,
    min_size: int = 10,
) -> list[pd.Timestamp]:
    """Detect change-points in a time series using PELT.

    Parameters
    ----------
    model : str
        Cost model: "rbf" (radial basis), "l2" (quadratic), "linear".
    penalty : float | None
        Manual penalty. If None, uses proper BIC: pen = d * log(n)
        where d = number of features being estimated per segment
        (mean + variance = 2 for univariate).
    min_size : int
        Minimum segment length.
    """
    values = series.values.reshape(-1, 1)

    algo = ruptures.Pelt(model=model, min_size=min_size)
    algo.fit(values)

    if penalty is None:
        # Proper BIC penalty: d * log(n)
        # For univariate Gaussian segments, d=2 (mean + variance)
        n = len(values)
        d = 2  # parameters per segment
        penalty = d * np.log(n)

    breakpoints = algo.predict(pen=penalty)
    # Remove last breakpoint (always equals len)
    breakpoints = [bp for bp in breakpoints if bp < len(values)]

    dates = series.index
    return [dates[bp] for bp in breakpoints if bp < len(dates)]
