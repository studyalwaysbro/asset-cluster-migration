"""Out-of-sample validation for regime detection via topology metrics.

This module answers: *do our novel topology metrics (CMI, TDS, layer
agreement) carry genuine predictive information about market regimes, or
are they just noise that happens to correlate in-sample?*

Approach
--------
1. Build a feature matrix from rolling topology metrics.
2. Align with HMM-labelled regime targets (shifted forward by 1 step).
3. Evaluate with forward-chaining TimeSeriesSplit so that we never
   train on future data — the gold standard for financial time series.
4. Report per-fold accuracy, macro-F1, and overall classification report.

This is a *validation* exercise, not a forecasting engine.  The goal is
to demonstrate that the topology metrics capture meaningful regime
structure — not to ship a live prediction model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured output from time-series cross-validation."""

    n_splits: int
    fold_accuracies: list[float] = field(default_factory=list)
    fold_f1_scores: list[float] = field(default_factory=list)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_f1: float = 0.0
    classification_report_text: str = ""
    feature_importances: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"OOS Accuracy: {self.mean_accuracy:.4f} ± {self.std_accuracy:.4f}  |  "
            f"OOS Macro-F1: {self.mean_f1:.4f}  |  {self.n_splits} folds"
        )


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def align_features_and_target(
    topology_metrics: pd.DataFrame,
    regime_labels: pd.Series,
    forecast_horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align topology features with forward-shifted regime labels.

    Parameters
    ----------
    topology_metrics : pd.DataFrame
        Date-indexed frame of rolling metrics (e.g., CMI, TDS,
        layer_agreement, mean_corr, realized_vol).
    regime_labels : pd.Series
        Date-indexed HMM regime labels (e.g., "calm" / "transition" / "stress").
    forecast_horizon : int
        Steps ahead — default 1 means today's features predict tomorrow's regime.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix (no NaN rows).
    y : pd.Series
        Corresponding target labels.
    """
    target = regime_labels.shift(-forecast_horizon)
    combined = topology_metrics.join(target.rename("_target"), how="inner").dropna()

    X = combined.drop(columns=["_target"])
    y = combined["_target"]
    return X, y


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_regime_detection(
    topology_metrics: pd.DataFrame,
    regime_labels: pd.Series,
    n_splits: int = 5,
    forecast_horizon: int = 1,
    seed: int = 42,
) -> ValidationResult:
    """Forward-chaining OOS validation of topology → regime mapping.

    Uses TimeSeriesSplit (expanding window) so that every test fold is
    strictly *after* its corresponding training data.  This prevents
    look-ahead bias.

    Parameters
    ----------
    topology_metrics : pd.DataFrame
        Columns like CMI, TDS, layer_agreement, realized_vol, mean_corr.
    regime_labels : pd.Series
        HMM regime labels aligned to the same dates.
    n_splits : int
        Number of chronological folds (default 5).
    forecast_horizon : int
        Steps ahead for target shift.
    seed : int
        Random state for the classifier.

    Returns
    -------
    ValidationResult
        Structured results including per-fold metrics and feature importances.
    """
    X, y = align_features_and_target(topology_metrics, regime_labels, forecast_horizon)

    if len(X) < n_splits * 10:
        logger.warning(
            "Only %d aligned samples — reducing splits to %d.",
            len(X),
            max(2, len(X) // 10),
        )
        n_splits = max(2, len(X) // 10)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=seed,
    )

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    all_y_true: list = []
    all_y_pred: list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        fold_acc.append(acc)
        fold_f1.append(f1)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())

        logger.info(
            "Fold %d/%d | %s → %s | Accuracy: %.4f | F1: %.4f",
            fold,
            n_splits,
            X_test.index[0].date(),
            X_test.index[-1].date(),
            acc,
            f1,
        )

    # Final model on full data for feature importances
    model.fit(X, y)
    importances = dict(
        sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    result = ValidationResult(
        n_splits=n_splits,
        fold_accuracies=fold_acc,
        fold_f1_scores=fold_f1,
        mean_accuracy=float(np.mean(fold_acc)),
        std_accuracy=float(np.std(fold_acc)),
        mean_f1=float(np.mean(fold_f1)),
        classification_report_text=classification_report(
            all_y_true, all_y_pred, zero_division=0,
        ),
        feature_importances=importances,
    )
    logger.info(result.summary)
    return result
