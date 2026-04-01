"""Temporal cluster tracking."""
from __future__ import annotations

import pandas as pd


def track_cluster_evolution(
    history: list[tuple[pd.Timestamp, dict[str, int]]]
) -> pd.DataFrame:
    """Build a DataFrame of cluster assignments over time."""
    records = []
    for date, assignments in history:
        for ticker, cluster_id in assignments.items():
            records.append({"date": date, "ticker": ticker, "cluster": cluster_id})
    return pd.DataFrame(records)
