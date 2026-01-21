"""Data Loader for Gold Price Forecasting"""

import os
import pandas as pd
import numpy as np


class DataLoader:
    """Load gold prices, bond yields, and inflation data."""
    
    def __init__(self, data_dir: str = "data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def _find_file(self, patterns: list) -> str:
        """Find file matching any pattern."""
        for fname in patterns:
            fpath = os.path.join(self.raw_dir, fname)
            if os.path.exists(fpath):
                return fpath
        return None
    
    def _detect_delimiter(self, filepath: str) -> str:
        """Detect CSV delimiter."""
        with open(filepath, 'r') as f:
            return ';' if ';' in f.readline() else ','
    
    def load_gold_prices(self, filepath: str = None) -> pd.DataFrame:
        """Load gold prices from CSV."""
        filepath = filepath or self._find_file(["gold_prices.csv", "XAU_1Month_data.csv", "xau_data.csv"])
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("Gold prices file not found")
        
        df = pd.read_csv(filepath, sep=self._detect_delimiter(filepath))
        df.columns = df.columns.str.strip().str.lower()
        
        # Parse date
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), df.columns[0])
        try:
            df['date'] = pd.to_datetime(df[date_col], format='%Y.%m.%d %H:%M')
        except:
            df['date'] = pd.to_datetime(df[date_col], format='mixed')
        
        # Get close price
        price_col = next((c for c in df.columns if 'close' in c), None)
        df['gold_price'] = df[price_col] if price_col else df.iloc[:, 1]
        
        df = df[['date', 'gold_price']].dropna().sort_values('date').reset_index(drop=True)
        print(f"Loaded gold prices: {len(df)} records")
        return df
    
    def load_bond_yields(self, filepath: str = None) -> pd.DataFrame:
        """Load bond yields from CSV."""
        filepath = filepath or self._find_file(["dgs10.csv", "yeilds.csv", "yields.csv", "bond_yields.csv"])
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("Bond yields file not found")
        
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        date_col = next((c for c in df.columns if 'date' in c), df.columns[0])
        df['date'] = pd.to_datetime(df[date_col])
        
        value_col = next((c for c in df.columns if c in ['dgs10', 'bond_yield', 'yield', 'value']), 
                        [c for c in df.columns if c != 'date' and c != date_col][0])
        df['bond_yield'] = pd.to_numeric(df[value_col], errors='coerce')
        
        df = df[['date', 'bond_yield']].dropna().sort_values('date').reset_index(drop=True)
        print(f"Loaded bond yields: {len(df)} records")
        return df
    
    def load_inflation_data(self, filepath: str = None) -> pd.DataFrame:
        """Load CPI/inflation data from CSV."""
        filepath = filepath or self._find_file(["cpiaucsl.csv", "inflaton.csv", "inflation.csv", "cpi.csv"])
        if not filepath or not os.path.exists(filepath):
            raise FileNotFoundError("Inflation data file not found")
        
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        date_col = next((c for c in df.columns if 'date' in c), df.columns[0])
        df['date'] = pd.to_datetime(df[date_col])
        
        value_col = next((c for c in df.columns if c in ['cpiaucsl', 'cpi', 'inflation', 'value']),
                        [c for c in df.columns if c != 'date' and c != date_col][0])
        df['cpi'] = pd.to_numeric(df[value_col], errors='coerce')
        
        df = df[['date', 'cpi']].dropna().sort_values('date').reset_index(drop=True)
        print(f"Loaded CPI data: {len(df)} records")
        return df
    
    def load_all_data(self):
        """Load all datasets."""
        return self.load_gold_prices(), self.load_bond_yields(), self.load_inflation_data()


def create_sample_data(output_dir: str = "data/raw"):
    """Create synthetic sample data for testing."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # Gold prices (daily)
    dates = pd.date_range('2004-01-01', '2024-12-31', freq='D')
    n = len(dates)
    trend = np.linspace(400, 2000, n)
    gold = trend + 50 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 20, n)
    pd.DataFrame({'Date': dates, 'Close': np.maximum(gold, 300)}).to_csv(
        os.path.join(output_dir, 'gold_prices.csv'), index=False)
    
    # Bond yields (monthly)
    monthly = pd.date_range('2004-01-01', '2024-12-31', freq='MS')
    n_m = len(monthly)
    yields = 3 + np.sin(2 * np.pi * np.arange(n_m) / 120) + np.random.normal(0, 0.3, n_m)
    pd.DataFrame({'DATE': monthly, 'DGS10': np.maximum(yields, 0.5)}).to_csv(
        os.path.join(output_dir, 'dgs10.csv'), index=False)
    
    # CPI (monthly)
    cpi = 180 * (1.002 ** np.arange(n_m)) + np.random.normal(0, 0.5, n_m)
    pd.DataFrame({'DATE': monthly, 'CPIAUCSL': cpi}).to_csv(
        os.path.join(output_dir, 'cpiaucsl.csv'), index=False)
    
    print(f"Sample data created in {output_dir}")


if __name__ == "__main__":
    create_sample_data()
    loader = DataLoader()
    gold, bond, cpi = loader.load_all_data()
