"""Event window definition and slicing."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml

from src.config import PROJECT_ROOT


@dataclass
class EventWindow:
    """Defines pre/event/post windows for an event study."""
    name: str
    description: str
    pre_start: pd.Timestamp
    pre_end: pd.Timestamp
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    post_start: pd.Timestamp
    post_end: pd.Timestamp
    peak_stress_date: pd.Timestamp | None = None

    @classmethod
    def from_config(cls, event_name: str) -> EventWindow:
        """Load event window from config/event_windows.yaml."""
        config_path = PROJECT_ROOT / "config" / "event_windows.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        event = config["events"][event_name]
        return cls(
            name=event["name"],
            description=event.get("description", ""),
            pre_start=pd.Timestamp(event["pre_start"]),
            pre_end=pd.Timestamp(event["pre_end"]),
            event_start=pd.Timestamp(event["event_start"]),
            event_end=pd.Timestamp(event["event_end"]),
            post_start=pd.Timestamp(event["post_start"]),
            post_end=pd.Timestamp(event["post_end"]),
            peak_stress_date=pd.Timestamp(event["peak_stress_date"]) if "peak_stress_date" in event else None,
        )

    def slice_returns(self, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Slice returns into pre/event/post windows."""
        return {
            "pre": returns.loc[self.pre_start:self.pre_end],
            "event": returns.loc[self.event_start:self.event_end],
            "post": returns.loc[self.post_start:self.post_end],
        }
