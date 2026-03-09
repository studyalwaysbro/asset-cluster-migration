"""Change-point detection using PELT algorithm."""
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
    """Detect change-points in a time series using PELT."""
    values = series.values.reshape(-1, 1)

    algo = ruptures.Pelt(model=model, min_size=min_size)
    algo.fit(values)

    if penalty is None:
        # BIC-based penalty
        n = len(values)
        penalty = np.log(n) * np.var(values)

    breakpoints = algo.predict(pen=penalty)
    # Remove last breakpoint (always equals len)
    breakpoints = [bp for bp in breakpoints if bp < len(values)]

    dates = series.index
    return [dates[bp] for bp in breakpoints if bp < len(dates)]
