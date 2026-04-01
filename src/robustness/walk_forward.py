"""Walk-forward validation for topology-based early warning signals.

Phase 4.1 \u2014 Tests whether findings discovered in-sample (2019-2022) replicate
out-of-sample (2023-2024), and whether re-training on 2019-2024 replicates
on 2025-2026.

Key questions tested:
1. Does cross-layer Granger causality (tail-dep CMI -> Pearson CMI) hold OOS?
2. Does topology crystallization (restructuring before events) replicate?
3. What is the false positive rate of early warning signals?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    # Cross-layer Granger
    granger_f_stat_train: float = 0.0
    granger_p_value_train: float = 1.0
    granger_f_stat_test: float = 0.0
    granger_p_value_test: float = 1.0
    granger_replicates: bool = False
    # Topology crystallization
    mean_tds_pre_event_train: float = 0.0
    mean_tds_during_event_train: float = 0.0
    mean_tds_pre_event_test: float = 0.0
    mean_tds_during_event_test: float = 0.0
    crystallization_replicates: bool = False
    # Early warning
    n_warnings_test: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    false_positive_rate: float = 0.0
    # Regime prediction
    regime_accuracy_test: float = 0.0
    regime_f1_test: float = 0.0


@dataclass
class WalkForwardSummary:
    """Aggregate results across all walk-forward folds."""
    folds: list[WalkForwardResult] = field(default_factory=list)
    granger_replication_rate: float = 0.0
    crystallization_replication_rate: float = 0.0
    mean_false_positive_rate: float = 0.0
    mean_regime_accuracy: float = 0.0
    mean_regime_f1: float = 0.0


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    splits: list[dict[str, str]] | None = None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate walk-forward train/test splits.

    Default splits:
    - Fold 1: Train 2019-2022, Test 2023-2024
    - Fold 2: Train 2019-2024, Test 2025-2026
    """
    if splits is None:
        splits = [
            {"train_end": "2022-12-31", "test_end": "2024-12-31"},
            {"train_end": "2024-12-31", "test_end": "2026-12-31"},
        ]

    result = []
    for split in splits:
        train_end = pd.Timestamp(split["train_end"])
        test_end = pd.Timestamp(split["test_end"])
        train_mask = dates <= train_end
        test_mask = (dates > train_end) & (dates <= test_end)
        if train_mask.any() and test_mask.any():
            result.append((dates[train_mask], dates[test_mask]))

    return result


def cross_layer_granger_test(
    cmi_tail: pd.Series,
    cmi_pearson: pd.Series,
    max_lag: int = 5,
) -> tuple[float, float]:
    """Test whether tail-dependence CMI Granger-causes Pearson CMI.

    This is the key cross-layer causality finding (p=0.041 in-sample).

    Returns (F-statistic, Bonferroni-corrected p-value).
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Align and clean
    combined = pd.DataFrame({
        "target": cmi_pearson,
        "source": cmi_tail,
    }).dropna()

    if len(combined) < max_lag + 20:
        logger.warning(
            f"Insufficient data for cross-layer Granger: {len(combined)} obs"
        )
        return 0.0, 1.0

    try:
        result = grangercausalitytests(
            combined[["target", "source"]],
            maxlag=max_lag,
            verbose=False,
        )

        # Bonferroni correction across lags
        best_f = 0.0
        best_p = 1.0
        for lag in result:
            raw_p = result[lag][0]["ssr_ftest"][1]
            corrected_p = min(raw_p * max_lag, 1.0)
            if corrected_p < best_p:
                best_p = corrected_p
                best_f = result[lag][0]["ssr_ftest"][0]

        return best_f, best_p
    except Exception as e:
        logger.error(f"Cross-layer Granger test failed: {e}")
        return 0.0, 1.0


def detect_early_warnings(
    tds_series: pd.Series,
    cmi_tail: pd.Series | None = None,
    tds_threshold_percentile: float = 90,
    lookback_days: int = 252,
) -> pd.Series:
    """Detect early warning signals from topology metrics.

    A warning fires when TDS exceeds its rolling percentile threshold,
    indicating unusual topology deformation.

    Returns boolean Series: True = warning active.
    """
    if len(tds_series) < lookback_days:
        lookback_days = max(len(tds_series) // 2, 10)

    rolling_threshold = tds_series.rolling(
        lookback_days, min_periods=lookback_days // 2
    ).quantile(tds_threshold_percentile / 100)

    warnings = tds_series > rolling_threshold

    # Optionally combine with tail CMI spike
    if cmi_tail is not None:
        tail_threshold = cmi_tail.rolling(
            lookback_days, min_periods=lookback_days // 2
        ).quantile(tds_threshold_percentile / 100)
        tail_warning = cmi_tail > tail_threshold
        # Both must fire (reduces false positives)
        warnings = warnings & tail_warning

    return warnings.fillna(False)


def evaluate_warnings_against_events(
    warnings: pd.Series,
    event_windows: list[dict[str, str]],
    lead_days: int = 30,
) -> tuple[int, int, int]:
    """Evaluate early warning signals against known event dates.

    A warning is a true positive if it fires within lead_days before
    an event window. Otherwise it's a false positive.

    Returns (n_warnings, n_true_positives, n_false_positives).
    """
    warning_dates = warnings[warnings].index
    n_warnings = len(warning_dates)

    if n_warnings == 0:
        return 0, 0, 0

    true_positives = 0
    for event in event_windows:
        event_start = pd.Timestamp(event["start"])
        pre_window_start = event_start - pd.Timedelta(days=lead_days)

        # Check if any warning fired in the pre-event window
        hits = warning_dates[
            (warning_dates >= pre_window_start) & (warning_dates < event_start)
        ]
        if len(hits) > 0:
            true_positives += 1

    # False positives: warnings not near any event
    fp_count = 0
    for wd in warning_dates:
        near_event = False
        for event in event_windows:
            event_start = pd.Timestamp(event["start"])
            event_end = pd.Timestamp(event["end"])
            pre_start = event_start - pd.Timedelta(days=lead_days)
            if pre_start <= wd <= event_end:
                near_event = True
                break
        if not near_event:
            fp_count += 1

    return n_warnings, true_positives, fp_count


def run_walk_forward_validation(
    metrics_df: pd.DataFrame,
    event_windows: list[dict[str, str]],
    splits: list[dict[str, str]] | None = None,
    significance: float = 0.05,
) -> WalkForwardSummary:
    """Run full walk-forward validation.

    Parameters
    ----------
    metrics_df : DataFrame
        Must contain columns: 'tds', 'cmi_pearson', 'cmi_tail'
        Index must be DatetimeIndex.
    event_windows : list
        Each dict has 'start', 'end', 'name' keys.
    splits : list | None
        Custom train/test splits. Default: 2019-2022/2023-2024 then 2019-2024/2025-2026.
    significance : float
        Alpha level for Granger causality replication.
    """
    summary = WalkForwardSummary()
    fold_splits = walk_forward_splits(metrics_df.index, splits)

    for train_dates, test_dates in fold_splits:
        fold = WalkForwardResult(
            train_start=train_dates.min(),
            train_end=train_dates.max(),
            test_start=test_dates.min(),
            test_end=test_dates.max(),
        )

        train_df = metrics_df.loc[train_dates]
        test_df = metrics_df.loc[test_dates]

        # 1. Cross-layer Granger causality
        if "cmi_tail" in metrics_df.columns and "cmi_pearson" in metrics_df.columns:
            fold.granger_f_stat_train, fold.granger_p_value_train = (
                cross_layer_granger_test(
                    train_df["cmi_tail"], train_df["cmi_pearson"]
                )
            )
            fold.granger_f_stat_test, fold.granger_p_value_test = (
                cross_layer_granger_test(
                    test_df["cmi_tail"], test_df["cmi_pearson"]
                )
            )
            fold.granger_replicates = fold.granger_p_value_test < significance

        # 2. Topology crystallization
        if "tds" in metrics_df.columns:
            for event in event_windows:
                ev_start = pd.Timestamp(event["start"])
                ev_end = pd.Timestamp(event["end"])
                pre_start = ev_start - pd.Timedelta(days=30)

                # Check if event falls in train or test period
                if ev_start >= fold.train_start and ev_end <= fold.train_end:
                    pre = train_df.loc[pre_start:ev_start, "tds"]
                    during = train_df.loc[ev_start:ev_end, "tds"]
                    if len(pre) > 0:
                        fold.mean_tds_pre_event_train = pre.mean()
                    if len(during) > 0:
                        fold.mean_tds_during_event_train = during.mean()
                elif ev_start >= fold.test_start and ev_end <= fold.test_end:
                    pre = test_df.loc[pre_start:ev_start, "tds"]
                    during = test_df.loc[ev_start:ev_end, "tds"]
                    if len(pre) > 0:
                        fold.mean_tds_pre_event_test = pre.mean()
                    if len(during) > 0:
                        fold.mean_tds_during_event_test = during.mean()

            # Crystallization = TDS peaks BEFORE event, not during
            if fold.mean_tds_pre_event_test > 0 and fold.mean_tds_during_event_test > 0:
                fold.crystallization_replicates = (
                    fold.mean_tds_pre_event_test > fold.mean_tds_during_event_test
                )

        # 3. Early warning false positive rate
        if "tds" in metrics_df.columns:
            test_events = [
                e for e in event_windows
                if pd.Timestamp(e["start"]) >= fold.test_start
                and pd.Timestamp(e["end"]) <= fold.test_end
            ]

            cmi_tail_test = test_df.get("cmi_tail")
            warnings = detect_early_warnings(
                test_df["tds"],
                cmi_tail=cmi_tail_test,
            )

            n_warn, n_tp, n_fp = evaluate_warnings_against_events(
                warnings, test_events
            )
            fold.n_warnings_test = n_warn
            fold.n_true_positives = n_tp
            fold.n_false_positives = n_fp
            fold.false_positive_rate = (
                n_fp / max(n_warn, 1) if n_warn > 0 else 0.0
            )

        summary.folds.append(fold)
        logger.info(
            f"Fold {len(summary.folds)}: "
            f"train={fold.train_start.date()}->{fold.train_end.date()}, "
            f"test={fold.test_start.date()}->{fold.test_end.date()}, "
            f"Granger replicates={fold.granger_replicates}, "
            f"FPR={fold.false_positive_rate:.2%}"
        )

    # Aggregate
    if summary.folds:
        n = len(summary.folds)
        summary.granger_replication_rate = (
            sum(f.granger_replicates for f in summary.folds) / n
        )
        summary.crystallization_replication_rate = (
            sum(f.crystallization_replicates for f in summary.folds) / n
        )
        summary.mean_false_positive_rate = (
            np.mean([f.false_positive_rate for f in summary.folds])
        )

    return summary
