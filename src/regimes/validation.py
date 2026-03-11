import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

def run_timeseries_validation(X: pd.DataFrame, y: pd.Series, n_splits: int = 3):
    """
    Validates regime predictions out-of-sample using forward-chaining Time Series Split.
    
    Args:
        X: Feature dataframe (e.g., your CMI, TDS, and Layer Agreement metrics).
        y: Target series (the HMM regime labels shifted forward by 1 step).
        n_splits: Number of chronological splits for training/testing.
        
    Returns:
        The trained model and a list of accuracy scores for each fold.
    """
    logger.info(f"Starting Time Series Split validation with {n_splits} splits...")
    
    # Initialize the chronological splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize a standard supervised classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    fold_accuracies = []
    
    # Loop through the data chronologically
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train on the past, predict the future
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate accuracy for this specific time block
        acc = accuracy_score(y_test, predictions)
        fold_accuracies.append(acc)
        
        logger.info(f"Fold {fold} | Test Window: {X_test.index[0].date()} to {X_test.index[-1].date()} | Accuracy: {acc:.4f}")
        
    logger.info(f"Average Out-of-Sample Accuracy: {np.mean(fold_accuracies):.4f}")
    
    return model, fold_accuracies