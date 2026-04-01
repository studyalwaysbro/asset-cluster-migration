"""Rolling window engine for temporal analysis."""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


class RollingWindowEngine:
    """Generate rolling windows over return data."""

    def __init__(
        self,
        window_size: int = 120,
        step_size: int = 5,
        min_periods: int = 60,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.min_periods = min_periods

    def generate_windows(
        self, returns: pd.DataFrame
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
        """Yield (window_end_date, window_returns) tuples."""
        dates = returns.index
        n = len(dates)

        for end_idx in range(self.window_size - 1, n, self.step_size):
            start_idx = end_idx - self.window_size + 1
            window = returns.iloc[start_idx : end_idx + 1]

            # Check minimum valid observations per column
            valid_counts = window.notna().sum()
            if (valid_counts >= self.min_periods).all():
                yield dates[end_idx], window

    def window_count(self, returns: pd.DataFrame) -> int:
        """Count number of windows that will be generated."""
        n = len(returns)
        if n < self.window_size:
            return 0
        return (n - self.window_size) // self.step_size + 1
