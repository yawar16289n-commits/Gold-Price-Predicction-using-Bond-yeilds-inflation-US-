"""Data Preprocessing for Gold Price Forecasting"""

import pandas as pd
from typing import Tuple


class DataPreprocessor:
    """Preprocess data for NeuralProphet."""
    
    def resample_to_monthly(self, df: pd.DataFrame, date_col: str, value_col: str, agg: str = 'mean') -> pd.DataFrame:
        """Resample to monthly frequency."""
        df = df.set_index(pd.to_datetime(df[date_col]))
        agg_func = {'mean': 'mean', 'last': 'last', 'first': 'first'}.get(agg, 'mean')
        result = df[[value_col]].resample('MS').agg(agg_func).reset_index()
        result.columns = [date_col, value_col]
        return result.dropna()
    
    def merge_datasets(self, gold_df: pd.DataFrame, bond_df: pd.DataFrame, inflation_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on date."""
        # Resample
        gold = self.resample_to_monthly(gold_df, 'date', 'gold_price', 'last')
        bond = self.resample_to_monthly(bond_df, 'date', 'bond_yield', 'mean')
        
        # Calculate inflation rate (YoY % change)
        inflation = inflation_df.sort_values('date').copy()
        inflation['inflation_rate'] = inflation['cpi'].pct_change(12) * 100
        inflation = inflation[['date', 'inflation_rate']].dropna()
        
        # Merge
        merged = gold.merge(bond, on='date').merge(inflation, on='date')
        print(f"Merged: {len(merged)} samples ({merged['date'].min()} to {merged['date'].max()})")
        return merged.sort_values('date').reset_index(drop=True)
    
    def prepare_neuralprophet_data(self, df: pd.DataFrame, regressors: list = None) -> pd.DataFrame:
        """Convert to NeuralProphet format (ds, y, regressors)."""
        regressors = regressors or ['bond_yield', 'inflation_rate']
        result = pd.DataFrame({'ds': pd.to_datetime(df['date']), 'y': df['gold_price']})
        for col in regressors:
            if col in df.columns:
                result[col] = df[col].values
        print(f"NeuralProphet data: {result.shape}, columns: {list(result.columns)}")
        return result.dropna()
    
    def train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split chronologically into train/test."""
        df = df.sort_values('ds').reset_index(drop=True)
        split = int(len(df) * (1 - test_ratio))
        train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
        print(f"Train: {len(train)} ({train['ds'].min()} to {train['ds'].max()})")
        print(f"Test: {len(test)} ({test['ds'].min()} to {test['ds'].max()})")
        return train, test


def preprocess_pipeline(gold_df, bond_df, inflation_df, test_ratio=0.2, normalize=False):
    """Complete preprocessing pipeline."""
    prep = DataPreprocessor()
    merged = prep.merge_datasets(gold_df, bond_df, inflation_df)
    np_df = prep.prepare_neuralprophet_data(merged)
    train, test = prep.train_test_split(np_df, test_ratio)
    return train, test, prep


if __name__ == "__main__":
    from data_loader import DataLoader, create_sample_data
    create_sample_data()
    gold, bond, cpi = DataLoader().load_all_data()
    train, test, _ = preprocess_pipeline(gold, bond, cpi)
    print(train.head())
