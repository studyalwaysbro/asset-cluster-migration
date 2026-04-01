"""Data fetch orchestrator."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd

from src.config import get_universe_config, get_settings, PROJECT_ROOT
from src.data.fmp_client import FMPClient

logger = logging.getLogger(__name__)


async def fetch_universe_data(
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch historical prices for all assets in the universe."""
    config = get_universe_config()
    from_date = from_date or config["start_date"]

    # Collect all tickers
    tickers = []
    for group in config["assets"].values():
        for asset in group:
            tickers.append(asset["ticker"])

    logger.info(f"Fetching {len(tickers)} tickers from {from_date}")

    client = FMPClient()
    try:
        data = await client.batch_fetch(tickers, from_date, to_date)
    finally:
        await client.close()

    logger.info(f"Fetched {sum(len(df) for df in data.values())} total rows")
    return data


def run_ingestion() -> dict[str, pd.DataFrame]:
    """Synchronous wrapper for fetch_universe_data."""
    return asyncio.run(fetch_universe_data())
