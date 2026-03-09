"""Project constants, type aliases, and enumerations."""
from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np
import pandas as pd


# Type aliases
SymbolList: TypeAlias = list[str]
SimilarityMatrix: TypeAlias = np.ndarray  # NxN float array
CommunityAssignment: TypeAlias = dict[str, int]  # {ticker: community_id}
LayerDict: TypeAlias = dict[str, SimilarityMatrix]


class SimilarityLayer(str, Enum):
    """Available similarity/dependency layers."""
    PEARSON_SHRINKAGE = "pearson_shrinkage"
    SPEARMAN = "spearman"
    DISTANCE_CORRELATION = "distance_correlation"
    MUTUAL_INFORMATION = "mutual_information"
    TAIL_DEPENDENCE = "tail_dependence"
    LEAD_LAG = "lead_lag"
    TRANSFER_ENTROPY = "transfer_entropy"


class ClusterMethod(str, Enum):
    """Available clustering methods."""
    LEIDEN = "leiden"
    SPECTRAL = "spectral"
    CONSENSUS = "consensus"
    MULTIPLEX = "multiplex"


class RegimeLabel(str, Enum):
    """Market regime labels (assigned post-hoc by volatility ordering)."""
    CALM = "calm"
    TRANSITION = "transition"
    STRESS = "stress"


class AssetCategory(str, Enum):
    """Asset classification categories."""
    EQUITY_US = "equity_us"
    SECTOR = "sector"
    EQUITY_INTL = "equity_intl"
    EQUITY_EM = "equity_em"
    BOND_GOVT = "bond_govt"
    BOND_CREDIT = "bond_credit"
    BOND_EM = "bond_em"
    COMMODITY = "commodity"
    REAL_ASSET = "real_asset"
    FX = "fx"
    VOLATILITY = "volatility"
    THEMATIC = "thematic"
    CRYPTO = "crypto"
    INFLATION = "inflation"
