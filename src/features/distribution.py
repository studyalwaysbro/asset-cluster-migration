"""Wasserstein and distributional distance methods (Phase 2)."""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def pairwise_wasserstein(
    returns_a: np.ndarray, returns_b: np.ndarray
) -> float:
    """Wasserstein-1 distance between two return distributions."""
    return wasserstein_distance(returns_a, returns_b)
