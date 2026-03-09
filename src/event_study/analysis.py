"""Event study topology analysis."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import networkx as nx

from src.event_study.windows import EventWindow
from src.migration.metrics import (
    cluster_migration_index,
    topology_deformation_score,
)
from src.graphs.topology import compute_modularity, compute_centrality_metrics


@dataclass
class EventStudyReport:
    """Results of an event study topology comparison."""
    event: EventWindow
    pre_modularity: float = 0.0
    event_modularity: float = 0.0
    post_modularity: float = 0.0
    pre_mean_corr: float = 0.0
    event_mean_corr: float = 0.0
    post_mean_corr: float = 0.0
    migration_during_event: float = 0.0
    tds_during_event: float = 0.0
    migrated_assets: list[str] = field(default_factory=list)
    bridge_assets: list[str] = field(default_factory=list)


class EventStudyAnalyzer:
    """Compare topology metrics across event windows."""

    def __init__(self, event: EventWindow):
        self.event = event

    def compare_topology(
        self,
        rolling_communities: dict[pd.Timestamp, dict[str, int]],
        rolling_graphs: dict[pd.Timestamp, nx.Graph],
    ) -> EventStudyReport:
        """Compare pre/during/post topology metrics."""
        report = EventStudyReport(event=self.event)

        # Find dates in each window
        pre_dates = [d for d in rolling_communities if self.event.pre_start <= d <= self.event.pre_end]
        event_dates = [d for d in rolling_communities if self.event.event_start <= d <= self.event.event_end]
        post_dates = [d for d in rolling_communities if self.event.post_start <= d <= self.event.post_end]

        # Average modularity per window
        for window_dates, attr in [(pre_dates, "pre_modularity"), (event_dates, "event_modularity"), (post_dates, "post_modularity")]:
            if window_dates:
                mods = []
                for d in window_dates:
                    if d in rolling_graphs and d in rolling_communities:
                        m = compute_modularity(rolling_graphs[d], rolling_communities[d])
                        mods.append(m)
                if mods:
                    setattr(report, attr, sum(mods) / len(mods))

        # Migration during event
        if len(event_dates) >= 2:
            first = rolling_communities.get(event_dates[0], {})
            last = rolling_communities.get(event_dates[-1], {})
            report.migration_during_event = cluster_migration_index(last, first)

            # Identify migrated assets
            common = set(first.keys()) & set(last.keys())
            report.migrated_assets = [a for a in common if first[a] != last[a]]

        # TDS relative to pre-event baseline
        if pre_dates and event_dates:
            baseline_date = pre_dates[-1]
            stress_date = event_dates[-1] if event_dates else None
            if baseline_date in rolling_graphs and stress_date and stress_date in rolling_graphs:
                report.tds_during_event = topology_deformation_score(
                    rolling_graphs[stress_date],
                    rolling_graphs[baseline_date],
                    rolling_communities.get(stress_date, {}),
                    rolling_communities.get(baseline_date, {}),
                )

        return report
