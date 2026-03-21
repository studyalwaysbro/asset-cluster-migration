"""HMM-based market regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm


class MarketRegimeDetector:
    """Gaussian HMM for regime detection on topology state vector."""

    def __init__(self, n_regimes: int = 3, seed: int = 42, n_iter: int = 100):
        self.n_regimes = n_regimes
        self.seed = seed
        self.n_iter = n_iter
        self.model: hmm.GaussianHMM | None = None
        self._regime_order: list[int] | None = None

    def fit(self, features: pd.DataFrame) -> MarketRegimeDetector:
        """Fit HMM on state vector [realized_vol, mean_corr, dispersion]."""
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self.model.fit(features.values)

        # Order regimes by mean of first feature (volatility) ascending
        means = self.model.means_[:, 0]
        self._regime_order = list(np.argsort(means))

        return self

    def predict_regimes(self, features: pd.DataFrame) -> pd.Series:
        """Return regime labels for each date."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        raw_labels = self.model.predict(features.values)

        # Remap to ordered labels: 0=calm, 1=transition, 2=stress
        label_map = {old: new for new, old in enumerate(self._regime_order)}
        ordered = [label_map[r] for r in raw_labels]

        labels = ["calm", "transition", "stress"]
        named = [labels[min(o, len(labels) - 1)] for o in ordered]

        return pd.Series(named, index=features.index, name="regime")

    def regime_properties(self) -> pd.DataFrame:
        """Return mean/std of each feature per regime."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        means = self.model.means_
        covars = self.model.covars_

        records = []
        labels = ["calm", "transition", "stress"]
        for i, orig_idx in enumerate(self._regime_order):
            record = {"regime": labels[min(i, len(labels) - 1)]}
            for j, feat in enumerate(["vol", "mean_corr", "dispersion"]):
                record[f"{feat}_mean"] = means[orig_idx, j]
                record[f"{feat}_std"] = np.sqrt(covars[orig_idx, j, j])
            records.append(record)

        return pd.DataFrame(records)
