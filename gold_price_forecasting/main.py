"""Gold Price Forecasting with NeuralProphet - Main Pipeline"""

import argparse
import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.preprocessing import preprocess_pipeline
from src.evaluation import calculate_all_metrics, print_metrics_report
from src.visualization import plot_prediction_graph, plot_lagged_effect_analysis, setup_plot_style


def main():
    parser = argparse.ArgumentParser(description='Gold Price Forecasting')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--forecast', type=int, default=6, help='Forecast periods (default: 6 months)')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    args = parser.parse_args()
    
    # Create directories
    for d in [f'{args.data_dir}/raw', f'{args.data_dir}/processed', 'models', 'results']:
        os.makedirs(d, exist_ok=True)
    
    print("="*60)
    print(" GOLD PRICE FORECASTING WITH NEURALPROPHET")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    try:
        gold, bond, cpi = DataLoader(args.data_dir).load_all_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Preprocess
    print("\n[2] Preprocessing...")
    train_df, test_df, _ = preprocess_pipeline(gold, bond, cpi, test_ratio=0.2)
    
    # Train or load model
    model_path = 'models/gold_price_model.pkl'
    if args.train or not os.path.exists(model_path):
        print("\n[3] Training model...")
        from neuralprophet import NeuralProphet, set_log_level
        set_log_level("ERROR")
        
        model = NeuralProphet(
            growth='linear', seasonality_mode='multiplicative',
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
            n_lags=12, n_forecasts=6, learning_rate=0.005, epochs=300, batch_size=16, loss_func='Huber',
            trend_reg=0.2, seasonality_reg=0.2
        )
        model.add_lagged_regressor('bond_yield', n_lags=12, regularization=0.25)
        model.add_lagged_regressor('inflation_rate', n_lags=12, regularization=0.1)
        model.fit(train_df, freq='MS')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    else:
        print("\n[3] Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # Evaluate
    print("\n[4] Evaluating...")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    forecast = model.predict(full_df)
    
    test_forecast = forecast.iloc[-len(test_df):]
    y_true = test_forecast['y'].dropna().values
    y_pred = test_forecast['yhat1'].dropna().values
    n = min(len(y_true), len(y_pred))
    
    metrics = calculate_all_metrics(y_true[:n], y_pred[:n])
    print_metrics_report(metrics, "Test Set Metrics")
    pd.DataFrame([metrics]).to_csv('results/metrics.csv', index=False)
    
    # Generate visualizations
    print("\n[5] Generating visualizations...")
    setup_plot_style()
    
    # Prediction graph
    output_path = 'results/gold_price_prediction_graph.png'
    plot_prediction_graph(train_df, test_df, test_forecast, metrics, save_path=output_path)
    
    # Lagged effect analysis
    lagged_path = 'results/lagged_effect_analysis.png'
    plot_lagged_effect_analysis(full_df, target_col='y', max_lag=24, save_path=lagged_path)
    
    # Future forecast
    if args.forecast > 0:
        print(f"\n[6] Forecasting {args.forecast} months...")
        future = model.make_future_dataframe(full_df, periods=args.forecast, n_historic_predictions=len(full_df))
        future['bond_yield'] = future['bond_yield'].ffill()
        future['inflation_rate'] = future['inflation_rate'].ffill()
        
        forecast = model.predict(future)
        future_only = forecast[forecast['ds'] > full_df['ds'].max()].copy()
        
        # Extract forecasts - use yhat1 through yhat6 columns for 1-6 month ahead predictions
        clean_forecast = pd.DataFrame({
            'Date': future_only['ds'],
            'Predicted_Gold_Price_USD': future_only[['yhat1', 'yhat2', 'yhat3', 'yhat4', 'yhat5', 'yhat6']].bfill(axis=1).iloc[:, 0]
        })
        
        print("\n6-Month Gold Price Forecast:")
        print(clean_forecast.to_string(index=False))
        
        clean_forecast.to_csv('results/future_forecast.csv', index=False)
        print(f"\n✓ Forecast saved to: results/future_forecast.csv")
    
    print("\n" + "="*60)
    print(" Pipeline completed!")
    print("="*60)


if __name__ == "__main__":
    main()
