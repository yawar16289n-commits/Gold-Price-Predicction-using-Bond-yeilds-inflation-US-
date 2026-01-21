"""Visualization for Gold Price Forecasting"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def setup_plot_style():
    """Set up plot style."""
    plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 12, 'figure.dpi': 100})


def plot_prediction_graph(train_df, test_df, test_forecast, metrics, save_path=None):
    """Create comprehensive prediction visualization with metrics."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Training data (blue)
    ax.plot(train_df['ds'], train_df['y'], color='#2E86AB', linewidth=2, 
            label='Training Data', alpha=0.9)
    
    # Test actual (green)
    ax.plot(test_df['ds'], test_df['y'], color='#28A745', linewidth=2.5, 
            label='Test Data (Actual)', alpha=0.9)
    
    # Test predicted (red dashed)
    ax.plot(test_forecast['ds'], test_forecast['yhat1'], color='#E63946', linewidth=2, 
            linestyle='--', label='Forecast (Predicted)', alpha=0.9)
    
    # Train/test split line
    split_date = train_df['ds'].iloc[-1]
    ax.axvline(x=split_date, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    
    # Highlight regions
    ax.axvspan(train_df['ds'].iloc[0], split_date, alpha=0.05, color='blue')
    ax.axvspan(split_date, test_df['ds'].iloc[-1], alpha=0.05, color='green')
    
    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gold Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Gold Price Forecasting: Training, Test & Predictions', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    
    # Metrics box
    metrics_text = f"Test Metrics:\nR² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}\nMAPE = {metrics['MAPE']:.2f}%"
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Gold price prediction saved to: {save_path}")

    return fig


def plot_lagged_effect_analysis(df, target_col='y', max_lag=24, save_path=None):
    """
    Plot lagged effect analysis in 2x2 grid:
    - Top-left: Lagged Correlation Analysis (0-24 months for both regressors)
    - Top-right: Gold vs Lagged Bond Yield (12 months) scatter
    - Bottom-left: Gold vs Lagged Inflation (12 months) scatter
    - Bottom-right: Gold Price and Inflation over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top-left: Lagged Correlation Analysis
    ax1 = axes[0, 0]
    lags = range(0, max_lag + 1)
    
    # Calculate correlations for bond yield
    bond_corrs = []
    for lag in lags:
        if lag == 0:
            corr = df[target_col].corr(df['bond_yield'])
        else:
            corr = df[target_col].iloc[lag:].corr(df['bond_yield'].iloc[:-lag])
        bond_corrs.append(corr)
    
    # Calculate correlations for inflation
    inflation_corrs = []
    for lag in lags:
        if lag == 0:
            corr = df[target_col].corr(df['inflation_rate'])
        else:
            corr = df[target_col].iloc[lag:].corr(df['inflation_rate'].iloc[:-lag])
        inflation_corrs.append(corr)
    
    ax1.plot(lags, bond_corrs, color='#2E86AB', linewidth=2.5, marker='o', markersize=4, label='Bond Yield')
    ax1.plot(lags, inflation_corrs, color='#E63946', linewidth=2.5, marker='s', markersize=4, label='Inflation')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Lag (months)', fontsize=11)
    ax1.set_ylabel('Correlation with Gold Price', fontsize=11)
    ax1.set_title('Lagged Correlation Analysis', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_lag)
    
    # 2. Top-right: Gold vs Lagged Bond Yield (12 months)
    ax2 = axes[0, 1]
    lag = 12
    bond_lagged = df['bond_yield'].iloc[:-lag].values
    gold_future = df[target_col].iloc[lag:].values
    
    ax2.scatter(bond_lagged, gold_future, alpha=0.6, color='#2E86AB', edgecolor='white', s=50)
    # Trend line
    z = np.polyfit(bond_lagged, gold_future, 1)
    p = np.poly1d(z)
    x_line = np.linspace(bond_lagged.min(), bond_lagged.max(), 100)
    ax2.plot(x_line, p(x_line), color='#E63946', linestyle='--', linewidth=2)
    
    corr = np.corrcoef(bond_lagged, gold_future)[0, 1]
    ax2.set_xlabel('Bond Yield % (lagged 12 months)', fontsize=11)
    ax2.set_ylabel('Gold Price (USD)', fontsize=11)
    ax2.set_title(f'Gold Price vs Lagged Bond Yield', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Bottom-left: Gold vs Lagged Inflation (12 months)
    ax3 = axes[1, 0]
    inflation_lagged = df['inflation_rate'].iloc[:-lag].values
    
    ax3.scatter(inflation_lagged, gold_future, alpha=0.6, color='#28A745', edgecolor='white', s=50)
    # Trend line
    z2 = np.polyfit(inflation_lagged, gold_future, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(inflation_lagged.min(), inflation_lagged.max(), 100)
    ax3.plot(x_line2, p2(x_line2), color='#E63946', linestyle='--', linewidth=2)
    
    corr2 = np.corrcoef(inflation_lagged, gold_future)[0, 1]
    ax3.set_xlabel('Inflation Rate % (lagged 12 months)', fontsize=11)
    ax3.set_ylabel('Gold Price (USD)', fontsize=11)
    ax3.set_title(f'Gold Price vs Lagged Inflation', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Bottom-right: Gold Price and Inflation over time
    ax4 = axes[1, 1]
    ax4.plot(df['ds'], df[target_col], color='#28A745', linewidth=2, label='Gold Price (USD)')
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Gold Price (USD)', fontsize=11, color='#28A745')
    ax4.tick_params(axis='y', labelcolor='#28A745')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df['ds'], df['inflation_rate'], color='#E63946', linewidth=2, linestyle='--', label='Inflation Rate (%)')
    ax4_twin.set_ylabel('Inflation Rate (%)', fontsize=11, color='#E63946')
    ax4_twin.tick_params(axis='y', labelcolor='#E63946')
    
    ax4.set_title('Gold Price and Inflation Over Time', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.suptitle('Lagged Effect Analysis: Gold Price vs Economic Indicators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Lagged effect analysis saved to: {save_path}")
    
    return fig
