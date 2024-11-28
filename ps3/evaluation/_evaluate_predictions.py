import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(actuals, predictions, sample_weight=None):
    """
    Evaluate predictions using various metrics and return a DataFrame.
    
    Parameters:
        actuals (array-like): The true outcome values.
        predictions (array-like): The predicted values.
        sample_weight (array-like, optional): Weights for each sample (e.g., exposure).
    
    Returns:
        pd.DataFrame: A DataFrame with metrics as index and their values.
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate Mean Absolute Error (MAE)
    metrics['MAE'] = mean_absolute_error(actuals, predictions, sample_weight=sample_weight)
    
    # Calculate Mean Squared Error (MSE)
    metrics['MSE'] = mean_squared_error(actuals, predictions, sample_weight=sample_weight)
    
    # Calculate Root Mean Squared Error (RMSE)
    metrics['RMSE'] = metrics['MSE'] ** 0.5
    
    # Compute exposure-adjusted mean
    if sample_weight is not None:
        actual_mean = np.average(actuals, weights=sample_weight)
        predicted_mean = np.average(predictions, weights=sample_weight)
    else:
        actual_mean = np.mean(actuals)
        predicted_mean = np.mean(predictions)
    
    # Compute Bias (deviation of predicted mean from actual mean)
    metrics['Bias'] = predicted_mean - actual_mean
    
    # Compute Deviance
    if sample_weight is None:
        deviance = 2 * np.sum(
            actuals * np.log(np.where(actuals == 0, 1, actuals / predictions)) - (actuals - predictions)
        )
    else:
        deviance = 2 * np.sum(
            sample_weight * (actuals * np.log(np.where(actuals == 0, 1, actuals / predictions)) - (actuals - predictions))
        )
    metrics['Deviance'] = deviance
    
    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    
    return metrics_df