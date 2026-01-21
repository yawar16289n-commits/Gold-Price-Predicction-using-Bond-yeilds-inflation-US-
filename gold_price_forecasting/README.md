# Gold Price Forecasting with NeuralProphet

A time series forecasting model that predicts gold prices using NeuralProphet with lagged regressors (bond yields and inflation rates).

## 📊 Model Performance

- **MAPE:** 5.87% 
- **R²:** 0.8200 (82% variance explained)
- **MAE:** 123.42 USD
- **RMSE:** 155.90 USD

## 🎯 Features

- **Autoregression:** Uses 12 months of historical gold prices
- **External Regressors:** Bond yields (DGS10) and CPI inflation rates
- **Multi-step Forecasting:** Predicts 6 months ahead
- **Visualizations:** Prediction graphs and lagged correlation analysis

## 📁 Project Structure

```
gold_price_forecasting/
├── data/
│   └── raw/                    # Input data files (CSV)
├── models/                     # Saved model files
├── results/                    # Output visualizations and forecasts
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── evaluation.py          # Metrics calculation
│   └── visualization.py       # Plotting functions
├── main.py                    # Main pipeline script
├── plot_results.py            # Visualization-only script
├── requirements.txt           # Python dependencies
└── README.md
```

## 🚀 Installation

```bash
# Create virtual environment
python -m venv .venv311

# Activate environment
# On Windows:
.venv311\Scripts\activate
# On Linux/Mac:
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Train and Forecast

```bash
# Run complete pipeline (evaluate + visualize + forecast)
python main.py

# Train new model
python main.py --train

# Generate 12-month forecast
python main.py --forecast 12
```

### Generate Visualizations Only

```bash
python plot_results.py
```

## 📈 How It Works

The model uses a 3-step approach:

1. **Look Back:** Examines the last 12 months of gold prices (autoregression)
2. **Add Context:** Considers current inflation rates and bond yields (lagged regressors)
3. **Forecast:** Predicts the next 6 months based on learned patterns

### Model Architecture

- **NeuralProphet** with:
  - Linear growth trend
  - Multiplicative seasonality (yearly)
  - 12 autoregressive lags
  - 6-step ahead forecasting
  - Huber loss function
  - Lagged regressors: bond yield (12 lags), inflation rate (12 lags)

### Training Configuration

- **Learning Rate:** 0.005
- **Epochs:** 300
- **Batch Size:** 16
- **Regularization:** L2 (trend: 0.2, seasonality: 0.2, regressors: 0.25/0.1)

## 📊 Data Requirements

The model expects three CSV files in `data/raw/`:

1. **XAU_1Month_data.csv** - Gold prices (columns: Date, Close)
2. **yields.csv** - Bond yields/DGS10 (columns: Date, DGS10)
3. **inflation.csv** - CPI data (columns: Date, CPI)

Data frequency: Monthly (MS)

## 📂 Output Files

- `results/gold_price_prediction_graph.png` - Training/test predictions
- `results/lagged_effect_analysis.png` - Correlation analysis (2x2 grid)
- `results/future_forecast.csv` - 6-month ahead predictions
- `results/metrics.csv` - Model performance metrics

## 🛠️ Dependencies

- neuralprophet==0.9.0
- torch
- pytorch-lightning
- pandas
- numpy
- matplotlib
- scikit-learn

## 📝 License

MIT License

## 👤 Author

Gold Price Forecasting ML Project
