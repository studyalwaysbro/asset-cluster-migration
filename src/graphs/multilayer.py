"""Multi-layer graph tensor construction."""
from __future__ import annotations

import numpy as np
import networkx as nx


def build_adjacency_tensor(
    layers: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack similarity layers into 3D tensor [layer, i, j]."""
    matrices = list(layers.values())
    return np.stack(matrices, axis=0)
