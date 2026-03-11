import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging

# Set up logging to match your orchestrator style
logger = logging.getLogger(__name__)

def compute_baseline_kmeans(returns_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> dict:
    """
    Computes a baseline K-Means clustering for a given rolling window of asset returns.
    
    Args:
        returns_df: DataFrame of asset returns (rows = dates, columns = tickers).
        n_clusters: The number of clusters to form. Default is 4 to match your typical regimes.
        random_state: Seed for reproducibility (matches your methodology config).
        
    Returns:
        A dictionary mapping each ticker to its assigned cluster ID.
    """
    logger.info(f"Running baseline K-Means with {n_clusters} clusters...")
    
    # Drop any assets that have missing data in this specific window to prevent KMeans from failing
    clean_returns = returns_df.dropna(axis=1)
    
    # We transpose the dataframe because we want to cluster the ASSETS (columns), not the DAYS (rows)
    features = clean_returns.T
    
    # Initialize and fit the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Map the tickers back to their new cluster labels
    asset_clusters = dict(zip(features.index, cluster_labels))
    
    return asset_clusters
