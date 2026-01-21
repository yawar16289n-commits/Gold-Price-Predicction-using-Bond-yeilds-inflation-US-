"""Evaluation Metrics for Gold Price Forecasting"""

import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mre(y_true, y_pred):
    """Mean Relative Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean((y_true[mask] - y_pred[mask]) / y_true[mask]) if mask.any() else np.nan


def mmre(y_true, y_pred):
    """Mean Magnitude of Relative Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) if mask.any() else np.nan


def rrse(y_true, y_pred):
    """Root Relative Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / denom) if denom != 0 else np.nan


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100 if mask.any() else np.nan


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100 if mask.any() else np.nan


def r_squared(y_true, y_pred):
    """R-squared (Coefficient of Determination)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def calculate_all_metrics(y_true, y_pred):
    """Calculate all metrics."""
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MRE': mre(y_true, y_pred),
        'MMRE': mmre(y_true, y_pred),
        'RRSE': rrse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
    }


def print_metrics_report(metrics, title="Performance Metrics"):
    """Print formatted metrics report."""
    print(f"\n{'='*50}\n {title}\n{'='*50}")
    for key, val in metrics.items():
        suffix = '%' if key in ['MAPE', 'SMAPE'] else ''
        print(f"  {key:10s}: {val:10.4f}{suffix}")
    print('='*50)
