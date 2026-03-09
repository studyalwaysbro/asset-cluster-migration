"""Universe loading and validation."""
from __future__ import annotations

import pandas as pd

from src.config import get_universe_config


def load_universe() -> list[dict]:
    """Load and flatten asset universe from config."""
    config = get_universe_config()
    assets = []
    for group_name, group_assets in config["assets"].items():
        for asset in group_assets:
            asset["group"] = group_name
            assets.append(asset)
    return assets


def get_ticker_list() -> list[str]:
    """Get flat list of tickers."""
    return [a["ticker"] for a in load_universe()]


def get_ticker_categories() -> dict[str, str]:
    """Get mapping of ticker to category."""
    return {a["ticker"]: a["category"] for a in load_universe()}
