import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

logger = logging.getLogger(__name__)

def train_regime_predictor(features_df: pd.DataFrame, regimes_series: pd.Series, forecast_horizon: int = 1):
    """
    Trains a Gradient Boosting classifier to predict future market regimes based on current topology metrics.
    
    Args:
        features_df: DataFrame of current metrics (e.g., CMI, TDS, Layer Agreement).
        regimes_series: Series of current HMM regime labels.
        forecast_horizon: How many steps ahead to predict (default is 1).
        
    Returns:
        The trained Gradient Boosting model and its accuracy on the training set.
    """
    logger.info(f"Training Boosting Classifier to predict {forecast_horizon} step(s) ahead...")
    
    # Shift the target variable backwards to align TODAY'S features with TOMORROW'S regime
    target = regimes_series.shift(-forecast_horizon)
    
    # Combine them temporarily to drop the NaN rows created by the shift at the very end of the dataset
    data = pd.concat([features_df, target.rename('target_regime')], axis=1).dropna()
    
    X = data.drop(columns=['target_regime'])
    y = data['target_regime']
    
    # Initialize the Boosting model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    # Calculate training accuracy
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    logger.info(f"Model trained successfully. Training Accuracy: {acc:.4f}")
    
    return model, acc