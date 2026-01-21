"""NeuralProphet Model for Gold Price Forecasting"""

import pandas as pd
import pickle
import os
import warnings
from typing import Optional, Tuple

warnings.filterwarnings('ignore')

from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")

class GoldPriceForecaster:
    """NeuralProphet-based gold price forecasting model."""
    
    DEFAULT_CONFIG = {
        'growth': 'linear',
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'n_lags': 12,
        'n_forecasts': 6,
        'learning_rate': 0.1,
        'epochs': 100,
        'batch_size': 32,
        'loss_func': 'MSE',
        'normalize': 'auto'
    }
    
    def __init__(self, **kwargs):
        self.config = {**self.DEFAULT_CONFIG, **kwargs}
        self.model = None
        self.regressors = []
        self.is_fitted = False
    
    def add_regressor(self, name: str, regularization: float = 0.01):
        """Add a lagged regressor."""
        self.regressors.append({'name': name, 'regularization': regularization})
    
    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None, freq: str = "MS"):
        """Train the model."""
        self.model = NeuralProphet(**self.config)
        
        for reg in self.regressors:
            self.model.add_lagged_regressor(names=reg['name'], regularization=reg['regularization'])
        
        if validation_df is not None:
            metrics = self.model.fit(train_df, validation_df, freq=freq)
        else:
            metrics = self.model.fit(train_df, freq=freq)
        
        self.is_fitted = True
        print("Model training completed!")
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(df)
    
    def forecast_future(self, df: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
        """Generate future forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        future = self.model.make_future_dataframe(df, periods=periods, n_historic_predictions=len(df))
        return self.model.predict(future)
    
    def plot_components(self, forecast: pd.DataFrame):
        """Plot forecast components."""
        return self.model.plot_components(forecast) if self.is_fitted else None
    
    def plot_forecast(self, forecast: pd.DataFrame):
        """Plot the forecast."""
        return self.model.plot(forecast) if self.is_fitted else None
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'config': self.config, 'regressors': self.regressors}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model, self.config, self.regressors = data['model'], data['config'], data['regressors']
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


def create_default_model(with_regressors: bool = True) -> GoldPriceForecaster:
    """Create a default model for gold price forecasting."""
    model = GoldPriceForecaster()
    if with_regressors:
        model.add_regressor('bond_yield')
        model.add_regressor('inflation_rate')
    return model


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    with_regressors: bool = True,
    save_path: Optional[str] = None
) -> Tuple[GoldPriceForecaster, pd.DataFrame, pd.DataFrame]:
    """Train model and generate predictions."""
    model = create_default_model(with_regressors)
    
    print("Training NeuralProphet model...")
    model.fit(train_df, freq="MS")
    
    print("Generating predictions...")
    train_preds = model.predict(train_df)
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_preds = model.predict(full_df)
    test_preds = full_preds.iloc[-len(test_df):].copy()
    
    if save_path:
        model.save_model(save_path)
    
    return model, train_preds, test_preds


if __name__ == "__main__":
    model = create_default_model()
    print(f"Config: {model.config}")
    print(f"Regressors: {model.regressors}")
