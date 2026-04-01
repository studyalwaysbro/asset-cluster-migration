"""Migration path tracking across time windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict


def track_migration_paths(
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Track which assets migrate between clusters over time.

    Parameters
    ----------
    cluster_assignments : DataFrame with columns [date, ticker, cluster]

    Returns
    -------
    DataFrame with columns [date, ticker, from_cluster, to_cluster, migrated]
    """
    dates = sorted(cluster_assignments["date"].unique())
    records = []

    for i in range(1, len(dates)):
        prev_date, curr_date = dates[i - 1], dates[i]
        prev = cluster_assignments[cluster_assignments["date"] == prev_date].set_index("ticker")["cluster"]
        curr = cluster_assignments[cluster_assignments["date"] == curr_date].set_index("ticker")["cluster"]

        common = prev.index.intersection(curr.index)
        for ticker in common:
            migrated = prev[ticker] != curr[ticker]
            records.append({
                "date": curr_date,
                "ticker": ticker,
                "from_cluster": int(prev[ticker]),
                "to_cluster": int(curr[ticker]),
                "migrated": bool(migrated),
            })

    return pd.DataFrame(records)


def migration_flow_matrix(
    cluster_assignments: pd.DataFrame,
    date_from: str,
    date_to: str,
) -> np.ndarray:
    """Compute migration flow matrix between two dates.

    Returns matrix where M[i,j] = count of assets moving from cluster i to j.
    """
    prev = cluster_assignments[
        cluster_assignments["date"] == date_from
    ].set_index("ticker")["cluster"]
    curr = cluster_assignments[
        cluster_assignments["date"] == date_to
    ].set_index("ticker")["cluster"]

    common = prev.index.intersection(curr.index)
    max_cluster = max(prev[common].max(), curr[common].max()) + 1

    flow = np.zeros((max_cluster, max_cluster), dtype=int)
    for ticker in common:
        flow[prev[ticker], curr[ticker]] += 1

    return flow


def dominant_migration_direction(
    tracking_df: pd.DataFrame,
    window_start: str,
    window_end: str,
) -> dict[str, dict]:
    """Identify dominant migration directions during a period.

    Returns per-asset summary: most common source/destination clusters.
    """
    subset = tracking_df[
        (tracking_df["date"] >= window_start)
        & (tracking_df["date"] <= window_end)
        & tracking_df["migrated"]
    ]

    result = {}
    for ticker, group in subset.groupby("ticker"):
        n_migrations = len(group)
        from_counts = group["from_cluster"].value_counts()
        to_counts = group["to_cluster"].value_counts()
        result[ticker] = {
            "n_migrations": n_migrations,
            "primary_source": int(from_counts.index[0]) if len(from_counts) > 0 else None,
            "primary_destination": int(to_counts.index[0]) if len(to_counts) > 0 else None,
            "migration_rate": n_migrations / len(
                tracking_df[
                    (tracking_df["date"] >= window_start)
                    & (tracking_df["date"] <= window_end)
                    & (tracking_df["ticker"] == ticker)
                ]
            ),
        }

    return result
