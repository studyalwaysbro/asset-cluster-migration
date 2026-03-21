"""Data cleaning, alignment, and validation."""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def align_and_clean(
    price_data: dict[str, pd.DataFrame],
    column: str = "adjClose",
    max_forward_fill: int = 5,
    max_missing_pct: float = 0.05,
) -> pd.DataFrame:
    """Align multiple price series and handle missing data."""
    # Extract adjusted close prices
    prices = {}
    for ticker, df in price_data.items():
        if column in df.columns and len(df) > 0:
            prices[ticker] = df[column]
        else:
            logger.warning(f"{ticker}: missing {column} column or empty")

    # Combine into single DataFrame
    combined = pd.DataFrame(prices)
    combined = combined.sort_index()

    # Forward fill up to max days
    combined = combined.ffill(limit=max_forward_fill)

    # Check missing data per ticker
    missing_pct = combined.isna().mean()
    excluded = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if excluded:
        logger.warning(f"Excluding {len(excluded)} assets with >{max_missing_pct*100}% missing: {excluded}")
        combined = combined.drop(columns=excluded)

    # Drop rows with any remaining NaN
    combined = combined.dropna()

    logger.info(f"Cleaned data: {combined.shape[0]} days x {combined.shape[1]} assets")
    return combined
