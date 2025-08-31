#!/usr/bin/env python3
"""
Rolling Regression Analysis: IEVR vs REVR
Walk-forward validation with 5-year training, 6-month validation, 6-month testing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load data and prepare for time series analysis.
    """
    print("📊 LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load merged dataset
    file_path = 'data_files/merged_revr_ievr_comprehensive.csv'
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {len(df):,} observations")
    
    # Convert dates and add time components
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    df['year'] = df['earnings_date'].dt.year
    df['quarter'] = df['earnings_date'].dt.quarter
    df['month'] = df['earnings_date'].dt.month
    
    # Create season identifier
    df['season'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
    
    # Calculate normative_iv_rv_ratio feature
    print("🔬 Creating normative_iv_rv_ratio feature...")
    df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
    
    # Handle infinite values and NaN
    df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Show feature statistics
    valid_ratio = df['normative_iv_rv_ratio'].notna().sum()
    print(f"  ✅ Created normative_iv_rv_ratio feature: {valid_ratio:,} valid values")
    if valid_ratio > 0:
        print(f"  📊 Mean: {df['normative_iv_rv_ratio'].mean():.4f}")
        print(f"  📊 Std: {df['normative_iv_rv_ratio'].std():.4f}")
        print(f"  📊 Range: {df['normative_iv_rv_ratio'].min():.4f} to {df['normative_iv_rv_ratio'].max():.4f}")
        
        # Check for reasonable values
        if df['normative_iv_rv_ratio'].mean() > 1.0:
            print(f"  ✅ IV > RV on average (typical volatility risk premium)")
        else:
            print(f"  ⚠️  RV > IV on average (unusual)")
    
    # Sort by date
    df = df.sort_values('earnings_date').reset_index(drop=True)
    
    # Remove extreme outliers (z-score > 3)
    revr_zscore = np.abs((df['revr'] - df['revr'].mean()) / df['revr'].std())
    ievr_zscore = np.abs((df['ievr'] - df['ievr'].mean()) / df['ievr'].std())
    
    df_clean = df[(revr_zscore <= 3) & (ievr_zscore <= 3)].copy()
    print(f"🧹 After outlier removal: {len(df_clean):,} observations")
    
    # Show date range
    print(f"📅 Date range: {df_clean['earnings_date'].min().strftime('%Y-%m')} to {df_clean['earnings_date'].max().strftime('%Y-%m')}")
    
    return df_clean

def create_time_windows(df, train_years=5, val_months=6, test_months=6):
    """
    Create rolling time windows for walk-forward validation.
    """
    print(f"\n🕒 CREATING TIME WINDOWS")
    print("="*60)
    print(f"Training window: {train_years} years")
    print(f"Validation window: {val_months} months")
    print(f"Testing window: {test_months} months")
    
    # Get unique dates and sort
    unique_dates = df['earnings_date'].dt.to_period('M').unique()
    unique_dates = np.sort(unique_dates)
    
    windows = []
    current_idx = 0
    
    while current_idx < len(unique_dates):
        # Training period
        train_end = current_idx + (train_years * 12) - 1
        if train_end >= len(unique_dates):
            break
            
        # Validation period
        val_start = train_end + 1
        val_end = val_start + val_months - 1
        if val_end >= len(unique_dates):
            break
            
        # Testing period
        test_start = val_end + 1
        test_end = test_start + test_months - 1
        if test_end >= len(unique_dates):
            break
            
        # Create window
        window = {
            'train_start': unique_dates[current_idx],
            'train_end': unique_dates[train_end],
            'val_start': unique_dates[val_start],
            'val_end': unique_dates[val_end],
            'test_start': unique_dates[test_start],
            'test_end': unique_dates[test_end],
            'window_id': len(windows) + 1
        }
        
        windows.append(window)
        current_idx += test_months  # Move forward by test window size
    
    print(f"📊 Created {len(windows)} rolling windows")
    
    # Show first few windows
    for i, window in enumerate(windows[:3]):
        print(f"  Window {i+1}: Train {window['train_start']}-{window['train_end']}, "
              f"Val {window['val_start']}-{window['val_end']}, "
              f"Test {window['test_start']}-{window['test_end']}")
    
    return windows

def get_data_for_window(df, window):
    """
    Extract data for a specific time window.
    """
    # Convert periods to datetime for filtering
    train_start = window['train_start'].to_timestamp()
    train_end = window['train_end'].to_timestamp()
    val_start = window['val_start'].to_timestamp()
    val_end = window['val_end'].to_timestamp()
    test_start = window['test_start'].to_timestamp()
    test_end = window['test_end'].to_timestamp()
    
    # Filter data
    train_data = df[(df['earnings_date'] >= train_start) & (df['earnings_date'] <= train_end)]
    val_data = df[(df['earnings_date'] >= val_start) & (df['earnings_date'] <= val_end)]
    test_data = df[(df['earnings_date'] >= test_start) & (df['earnings_date'] <= test_end)]
    
    return train_data, val_data, test_data

def run_rolling_regression(df, windows):
    """
    Run rolling regression for each time window.
    """
    print(f"\n🔬 RUNNING ROLLING REGRESSION")
    print("="*60)
    
    results = []
    all_predictions = []  # Store all predicted vs actual values
    
    for i, window in enumerate(windows):
        print(f"\n📊 Processing Window {i+1}/{len(windows)}")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Val:   {window['val_start']} to {window['val_end']}")
        print(f"  Test:  {window['test_start']} to {window['test_end']}")
        
        # Get data for this window
        train_data, val_data, test_data = get_data_for_window(df, window)
        
        # Check if we have enough data
        if len(train_data) < 50 or len(val_data) < 10 or len(test_data) < 10:
            print(f"  ⚠️  Insufficient data - skipping window")
            continue
        
        # Train model - now using both IEVR and normative_iv_rv_ratio
        X_train = train_data[['ievr', 'normative_iv_rv_ratio']].values
        y_train = train_data['revr'].values
        
        # Remove rows with NaN values in features
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]
        
        if len(X_train_clean) < 30:  # Need minimum observations
            print(f"  ⚠️  Insufficient clean data after NaN removal - skipping window")
            continue
        
        model = LinearRegression()
        model.fit(X_train_clean, y_train_clean)
        
        # Get coefficients
        intercept = model.intercept_
        ievr_coef = model.coef_[0]
        ratio_coef = model.coef_[1]
        
        # Validation performance
        X_val = val_data[['ievr', 'normative_iv_rv_ratio']].values
        y_val = val_data['revr'].values
        
        # Remove NaN values for validation
        val_valid_mask = ~np.isnan(X_val).any(axis=1)
        X_val_clean = X_val[val_valid_mask]
        y_val_clean = y_val[val_valid_mask]
        
        if len(X_val_clean) < 5:  # Need minimum validation observations
            print(f"  ⚠️  Insufficient clean validation data - skipping window")
            continue
        
        y_val_pred = model.predict(X_val_clean)
        
        val_r2 = r2_score(y_val_clean, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_clean, y_val_pred))
        val_mae = mean_absolute_error(y_val_clean, y_val_pred)
        
        # Test performance
        X_test = test_data[['ievr', 'normative_iv_rv_ratio']].values
        y_test = test_data['revr'].values
        
        # Remove NaN values for testing
        test_valid_mask = ~np.isnan(X_test).any(axis=1)
        X_test_clean = X_test[test_valid_mask]
        y_test_clean = y_test[test_valid_mask]
        
        if len(X_test_clean) < 5:  # Need minimum test observations
            print(f"  ⚠️  Insufficient clean test data - skipping window")
            continue
        
        y_test_pred = model.predict(X_test_clean)
        
        test_r2 = r2_score(y_test_clean, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_test_pred))
        test_mae = mean_absolute_error(y_test_clean, y_test_pred)
        
        # Store validation predictions
        val_data_clean = val_data[val_valid_mask].copy()
        val_data_clean['predicted_revr'] = y_val_pred
        val_data_clean['actual_revr'] = y_val_clean
        val_data_clean['window_id'] = window['window_id']
        val_data_clean['period'] = 'validation'
        val_data_clean['train_start'] = window['train_start'].strftime('%Y-%m')
        val_data_clean['train_end'] = window['train_end'].strftime('%Y-%m')
        val_data_clean['val_start'] = window['val_start'].strftime('%Y-%m')
        val_data_clean['val_end'] = window['val_end'].strftime('%Y-%m')
        val_data_clean['test_start'] = window['test_start'].strftime('%Y-%m')
        val_data_clean['test_end'] = window['test_end'].strftime('%Y-%m')
        
        # Store test predictions
        test_data_clean = test_data[test_valid_mask].copy()
        test_data_clean['predicted_revr'] = y_test_pred
        test_data_clean['actual_revr'] = y_test_clean
        test_data_clean['window_id'] = window['window_id']
        test_data_clean['period'] = 'test'
        test_data_clean['train_start'] = window['train_start'].strftime('%Y-%m')
        test_data_clean['train_end'] = window['train_end'].strftime('%Y-%m')
        test_data_clean['val_start'] = window['val_start'].strftime('%Y-%m')
        test_data_clean['val_end'] = window['val_end'].strftime('%Y-%m')
        test_data_clean['test_start'] = window['test_start'].strftime('%Y-%m')
        test_data_clean['test_end'] = window['test_end'].strftime('%Y-%m')
        
        # Combine predictions for this window
        window_predictions = pd.concat([val_data_clean, test_data_clean], ignore_index=True)
        all_predictions.append(window_predictions)
        
        # Store results
        window_result = {
            'window_id': window['window_id'],
            'train_start': window['train_start'].strftime('%Y-%m'),
            'train_end': window['train_end'].strftime('%Y-%m'),
            'val_start': window['val_start'].strftime('%Y-%m'),
            'val_end': window['val_end'].strftime('%Y-%m'),
            'test_start': window['test_start'].strftime('%Y-%m'),
            'test_end': window['test_end'].strftime('%Y-%m'),
            'train_obs': len(train_data),
            'val_obs': len(val_data),
            'test_obs': len(test_data),
            'intercept': intercept,
            'ievr_coef': ievr_coef,
            'ratio_coef': ratio_coef,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        results.append(window_result)
        
        print(f"  ✅ Train: {len(train_data)} obs, Val: {len(val_data)} obs, Test: {len(test_data)} obs")
        print(f"  📊 Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return results, all_predictions

def analyze_results(results):
    """
    Analyze and summarize the rolling regression results.
    """
    print(f"\n📊 ROLLING REGRESSION SUMMARY")
    print("="*60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Overall statistics
    print(f"📈 OVERALL PERFORMANCE:")
    print(f"  Total windows: {len(results_df)}")
    print(f"  Average validation R²: {results_df['val_r2'].mean():.4f}")
    print(f"  Average test R²: {results_df['test_r2'].mean():.4f}")
    print(f"  Average validation RMSE: {results_df['val_rmse'].mean():.4f}")
    print(f"  Average test RMSE: {results_df['test_rmse'].mean():.4f}")
    
    # Coefficient stability
    print(f"\n🔢 COEFFICIENT STABILITY:")
    print(f"  Intercept - Mean: {results_df['intercept'].mean():.4f}, Std: {results_df['intercept'].std():.4f}")
    print(f"  IEVR Coefficient - Mean: {results_df['ievr_coef'].mean():.4f}, Std: {results_df['ievr_coef'].std():.4f}")
    print(f"  Normative IV/RV Ratio Coefficient - Mean: {results_df['ratio_coef'].mean():.4f}, Std: {results_df['ratio_coef'].std():.4f}")
    
    # Performance over time
    print(f"\n📅 PERFORMANCE OVER TIME:")
    results_df['train_end_year'] = pd.to_numeric(results_df['train_end'].str[:4])
    yearly_perf = results_df.groupby('train_end_year').agg({
        'test_r2': ['mean', 'std', 'count'],
        'test_rmse': ['mean', 'std'],
        'ievr_coef': ['mean', 'std'],
        'ratio_coef': ['mean', 'std']
    }).round(4)
    
    print("Yearly Performance Summary:")
    print(yearly_perf)
    
    # Save results
    output_file = 'data_files/rolling_regression_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results_df

def save_predictions_to_csv(all_predictions, output_file):
    """
    Save all predicted vs actual REVR values to a CSV file.
    """
    if not all_predictions:
        print("No predictions to save.")
        return

    # Concatenate all predictions into a single DataFrame
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Select and reorder columns for better readability
    prediction_columns = [
        'earnings_date', 'ticker', 'company_name', 'year', 'quarter',
        'window_id', 'period', 'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end',
        'predicted_revr', 'actual_revr', 'ievr', 'normative_iv_rv_ratio',
        'revr', 'normative_realized_vol', 'avg_pre'
    ]
    
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in prediction_columns if col in all_predictions_df.columns]
    all_predictions_df = all_predictions_df[available_columns]
    
    # Sort by window_id, period, and earnings_date
    all_predictions_df = all_predictions_df.sort_values(['window_id', 'period', 'earnings_date'])
    
    all_predictions_df.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")
    print(f"📊 Total prediction records: {len(all_predictions_df):,}")
    print(f"📊 Windows: {all_predictions_df['window_id'].nunique()}")
    print(f"📊 Periods: {all_predictions_df['period'].value_counts().to_dict()}")
    
    return all_predictions_df

def analyze_predictions(all_predictions_df):
    """
    Analyze the quality of predictions across all windows.
    """
    if all_predictions_df is None or len(all_predictions_df) == 0:
        print("No predictions to analyze.")
        return
    
    print(f"\n🔍 PREDICTION QUALITY ANALYSIS")
    print("="*60)
    
    # Overall prediction accuracy
    overall_r2 = r2_score(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'])
    overall_rmse = np.sqrt(mean_squared_error(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr']))
    overall_mae = mean_absolute_error(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'])
    
    print(f"📊 OVERALL PREDICTION ACCURACY:")
    print(f"  R²: {overall_r2:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  MAE: {overall_mae:.4f}")
    
    # Performance by period (validation vs test)
    period_performance = all_predictions_df.groupby('period').apply(
        lambda x: pd.Series({
            'r2': r2_score(x['actual_revr'], x['predicted_revr']),
            'rmse': np.sqrt(mean_squared_error(x['actual_revr'], x['predicted_revr'])),
            'mae': mean_absolute_error(x['actual_revr'], x['predicted_revr']),
            'count': len(x)
        })
    ).round(4)
    
    print(f"\n📊 PERFORMANCE BY PERIOD:")
    print(period_performance)
    
    # Performance by window
    window_performance = all_predictions_df.groupby('window_id').apply(
        lambda x: pd.Series({
            'r2': r2_score(x['actual_revr'], x['predicted_revr']),
            'rmse': np.sqrt(mean_squared_error(x['actual_revr'], x['predicted_revr'])),
            'mae': mean_absolute_error(x['actual_revr'], x['predicted_revr']),
            'count': len(x)
        })
    ).round(4)
    
    print(f"\n📊 PERFORMANCE BY WINDOW:")
    print(f"  Best R²: {window_performance['r2'].max():.4f} (Window {window_performance['r2'].idxmax()})")
    print(f"  Worst R²: {window_performance['r2'].min():.4f} (Window {window_performance['r2'].idxmin()})")
    print(f"  Average R²: {window_performance['r2'].mean():.4f}")
    
    # Prediction bias analysis
    all_predictions_df['prediction_error'] = all_predictions_df['predicted_revr'] - all_predictions_df['actual_revr']
    all_predictions_df['abs_error'] = np.abs(all_predictions_df['prediction_error'])
    
    print(f"\n📊 PREDICTION BIAS ANALYSIS:")
    print(f"  Mean error: {all_predictions_df['prediction_error'].mean():.6f}")
    print(f"  Error std: {all_predictions_df['prediction_error'].std():.6f}")
    print(f"  Mean absolute error: {all_predictions_df['abs_error'].mean():.6f}")
    
    # Save detailed performance metrics
    performance_file = 'data_files/rolling_regression_predictions_performance.csv'
    window_performance.to_csv(performance_file)
    print(f"\n💾 Performance metrics saved to: {performance_file}")
    
    return all_predictions_df

def create_prediction_visualizations(all_predictions_df):
    """
    Create visualizations for predicted vs actual REVR values.
    """
    if all_predictions_df is None or len(all_predictions_df) == 0:
        print("No predictions to visualize.")
        return
    
    print(f"\n📊 CREATING PREDICTION VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rolling Regression: Predicted vs Actual REVR Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall predicted vs actual scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'], 
                alpha=0.6, s=20)
    
    # Add perfect prediction line
    min_val = min(all_predictions_df['actual_revr'].min(), all_predictions_df['predicted_revr'].min())
    max_val = max(all_predictions_df['actual_revr'].max(), all_predictions_df['predicted_revr'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual REVR')
    ax1.set_ylabel('Predicted REVR')
    ax1.set_title('Predicted vs Actual REVR (All Windows)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² text
    overall_r2 = r2_score(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'])
    ax1.text(0.05, 0.95, f'R² = {overall_r2:.4f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Predicted vs actual by period (validation vs test)
    ax2 = axes[0, 1]
    for period in ['validation', 'test']:
        period_data = all_predictions_df[all_predictions_df['period'] == period]
        ax2.scatter(period_data['actual_revr'], period_data['predicted_revr'], 
                   alpha=0.6, s=20, label=period.capitalize())
    
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual REVR')
    ax2.set_ylabel('Predicted REVR')
    ax2.set_title('Predicted vs Actual REVR by Period')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction errors over time
    ax3 = axes[1, 0]
    all_predictions_df['earnings_date'] = pd.to_datetime(all_predictions_df['earnings_date'])
    all_predictions_df = all_predictions_df.sort_values('earnings_date')
    
    ax3.scatter(all_predictions_df['earnings_date'], all_predictions_df['prediction_error'], 
                alpha=0.6, s=20, c=all_predictions_df['window_id'], cmap='tab10')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Earnings Date')
    ax3.set_ylabel('Prediction Error (Predicted - Actual)')
    ax3.set_title('Prediction Errors Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by window
    ax4 = axes[1, 1]
    window_performance = all_predictions_df.groupby('window_id').apply(
        lambda x: r2_score(x['actual_revr'], x['predicted_revr'])
    )
    
    ax4.bar(range(len(window_performance)), window_performance.values, alpha=0.7)
    ax4.set_xlabel('Window ID')
    ax4.set_ylabel('R² Score')
    ax4.set_title('Model Performance by Window')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(len(window_performance)))
    ax4.set_xticklabels(window_performance.index)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/rolling_regression_predictions_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def export_detailed_predictions(all_predictions_df):
    """
    Export detailed predictions with additional metadata for further analysis.
    """
    if all_predictions_df is None or len(all_predictions_df) == 0:
        print("No predictions to export.")
        return
    
    print(f"\n📋 EXPORTING DETAILED PREDICTIONS")
    print("="*60)
    
    # Create a detailed export with additional calculated fields
    detailed_df = all_predictions_df.copy()
    
    # Add additional calculated fields
    detailed_df['prediction_error'] = detailed_df['predicted_revr'] - detailed_df['actual_revr']
    detailed_df['abs_error'] = np.abs(detailed_df['prediction_error'])
    detailed_df['error_pct'] = (detailed_df['prediction_error'] / detailed_df['actual_revr']) * 100
    
    # Add rolling statistics for each window
    detailed_df['window_error_mean'] = detailed_df.groupby('window_id')['prediction_error'].transform('mean')
    detailed_df['window_error_std'] = detailed_df.groupby('window_id')['prediction_error'].transform('std')
    
    # Add time-based features
    detailed_df['earnings_date'] = pd.to_datetime(detailed_df['earnings_date'])
    detailed_df['year'] = detailed_df['earnings_date'].dt.year
    detailed_df['month'] = detailed_df['earnings_date'].dt.month
    detailed_df['quarter'] = detailed_df['earnings_date'].dt.quarter
    
    # Create a summary by window and period
    summary_stats = detailed_df.groupby(['window_id', 'period']).agg({
        'predicted_revr': ['count', 'mean', 'std'],
        'actual_revr': ['mean', 'std'],
        'prediction_error': ['mean', 'std'],
        'abs_error': 'mean',
        'error_pct': 'mean'
    }).round(6)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    # Save detailed predictions
    detailed_file = 'data_files/rolling_regression_detailed_predictions.csv'
    detailed_df.to_csv(detailed_file, index=False)
    print(f"💾 Detailed predictions saved to: {detailed_file}")
    
    # Save summary statistics
    summary_file = 'data_files/rolling_regression_predictions_summary.csv'
    summary_stats.to_csv(summary_file, index=False)
    print(f"💾 Summary statistics saved to: {summary_file}")
    
    # Print some key statistics
    print(f"\n📊 DETAILED STATISTICS:")
    print(f"  Total predictions: {len(detailed_df):,}")
    print(f"  Windows: {detailed_df['window_id'].nunique()}")
    print(f"  Date range: {detailed_df['earnings_date'].min().strftime('%Y-%m-%d')} to {detailed_df['earnings_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average absolute error: {detailed_df['abs_error'].mean():.6f}")
    print(f"  Average percentage error: {detailed_df['error_pct'].mean():.2f}%")
    
    return detailed_df, summary_stats

def create_prediction_comparison_table(all_predictions_df):
    """
    Create a comparison table showing best and worst predictions.
    """
    if all_predictions_df is None or len(all_predictions_df) == 0:
        print("No predictions to compare.")
        return
    
    print(f"\n📊 PREDICTION COMPARISON TABLE")
    print("="*60)
    
    # Calculate prediction errors
    all_predictions_df['prediction_error'] = all_predictions_df['predicted_revr'] - all_predictions_df['actual_revr']
    all_predictions_df['abs_error'] = np.abs(all_predictions_df['prediction_error'])
    
    # Best predictions (closest to actual)
    best_predictions = all_predictions_df.nsmallest(10, 'abs_error')[['earnings_date', 'ticker', 'company_name', 'actual_revr', 'predicted_revr', 'prediction_error', 'window_id', 'period']]
    
    # Worst predictions (furthest from actual)
    worst_predictions = all_predictions_df.nlargest(10, 'abs_error')[['earnings_date', 'ticker', 'company_name', 'actual_revr', 'predicted_revr', 'prediction_error', 'window_id', 'period']]
    
    # Largest overpredictions (predicted > actual)
    largest_overpredictions = all_predictions_df.nlargest(10, 'prediction_error')[['earnings_date', 'ticker', 'company_name', 'actual_revr', 'predicted_revr', 'prediction_error', 'window_id', 'period']]
    
    # Largest underpredictions (predicted < actual)
    largest_underpredictions = all_predictions_df.nsmallest(10, 'prediction_error')[['earnings_date', 'ticker', 'company_name', 'actual_revr', 'predicted_revr', 'prediction_error', 'window_id', 'period']]
    
    print("🏆 TOP 10 BEST PREDICTIONS (Smallest Absolute Error):")
    print(best_predictions.round(6).to_string(index=False))
    
    print(f"\n💥 TOP 10 WORST PREDICTIONS (Largest Absolute Error):")
    print(worst_predictions.round(6).to_string(index=False))
    
    print(f"\n📈 TOP 10 LARGEST OVERPREDICTIONS (Predicted > Actual):")
    print(largest_overpredictions.round(6).to_string(index=False))
    
    print(f"\n📉 TOP 10 LARGEST UNDERPREDICTIONS (Predicted < Actual):")
    print(largest_underpredictions.round(6).to_string(index=False))
    
    # Save comparison tables
    comparison_file = 'data_files/rolling_regression_predictions_comparison.csv'
    
    # Combine all comparison data
    comparison_data = []
    
    # Add best predictions
    best_predictions['category'] = 'Best Predictions'
    comparison_data.append(best_predictions)
    
    # Add worst predictions
    worst_predictions['category'] = 'Worst Predictions'
    comparison_data.append(worst_predictions)
    
    # Add largest overpredictions
    largest_overpredictions['category'] = 'Largest Overpredictions'
    comparison_data.append(largest_overpredictions)
    
    # Add largest underpredictions
    largest_underpredictions['category'] = 'Largest Underpredictions'
    comparison_data.append(largest_underpredictions)
    
    # Combine and save
    comparison_df = pd.concat(comparison_data, ignore_index=True)
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison table saved to: {comparison_file}")
    
    return comparison_df

def create_summary_report(results_df, all_predictions_df, detailed_df, summary_stats, comparison_df):
    """
    Create a comprehensive summary report of all findings.
    """
    if all_predictions_df is None or len(all_predictions_df) == 0:
        print("No data to summarize.")
        return
    
    print(f"\n📋 CREATING COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    # Calculate overall statistics
    overall_r2 = r2_score(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'])
    overall_rmse = np.sqrt(mean_squared_error(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr']))
    overall_mae = mean_absolute_error(all_predictions_df['actual_revr'], all_predictions_df['predicted_revr'])
    
    # Performance by period
    val_performance = all_predictions_df[all_predictions_df['period'] == 'validation']
    test_performance = all_predictions_df[all_predictions_df['period'] == 'test']
    
    val_r2 = r2_score(val_performance['actual_revr'], val_performance['predicted_revr']) if len(val_performance) > 0 else 0
    test_r2 = r2_score(test_performance['actual_revr'], test_performance['predicted_revr']) if len(test_performance) > 0 else 0
    
    # Create summary report
    report_lines = []
    report_lines.append("ROLLING REGRESSION ANALYSIS: COMPREHENSIVE SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overview
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Windows: {len(results_df) if results_df is not None else 0}")
    report_lines.append(f"Total Predictions: {len(all_predictions_df):,}")
    report_lines.append(f"Date Range: {all_predictions_df['earnings_date'].min().strftime('%Y-%m-%d')} to {all_predictions_df['earnings_date'].max().strftime('%Y-%m-%d')}")
    report_lines.append("")
    
    # Model Performance
    report_lines.append("MODEL PERFORMANCE")
    report_lines.append("-" * 40)
    report_lines.append(f"Overall R²: {overall_r2:.4f}")
    report_lines.append(f"Overall RMSE: {overall_rmse:.6f}")
    report_lines.append(f"Overall MAE: {overall_mae:.6f}")
    report_lines.append("")
    
    # Performance by Period
    report_lines.append("PERFORMANCE BY PERIOD")
    report_lines.append("-" * 40)
    report_lines.append(f"Validation R²: {val_r2:.4f} ({len(val_performance):,} predictions)")
    report_lines.append(f"Test R²: {test_r2:.4f} ({len(test_performance):,} predictions)")
    report_lines.append("")
    
    # Window Performance Summary
    if results_df is not None:
        report_lines.append("WINDOW PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Average Validation R²: {results_df['val_r2'].mean():.4f}")
        report_lines.append(f"Average Test R²: {results_df['test_r2'].mean():.4f}")
        report_lines.append(f"Best Test R²: {results_df['test_r2'].max():.4f}")
        report_lines.append(f"Worst Test R²: {results_df['test_r2'].min():.4f}")
        report_lines.append("")
    
    # Prediction Quality
    report_lines.append("PREDICTION QUALITY")
    report_lines.append("-" * 40)
    report_lines.append(f"Mean Absolute Error: {detailed_df['abs_error'].mean():.6f}")
    report_lines.append(f"Mean Percentage Error: {detailed_df['error_pct'].mean():.2f}%")
    report_lines.append(f"Error Standard Deviation: {detailed_df['prediction_error'].std():.6f}")
    report_lines.append("")
    
    # Best and Worst Predictions
    report_lines.append("PREDICTION EXTREMES")
    report_lines.append("-" * 40)
    best_pred = comparison_df[comparison_df['category'] == 'Best Predictions'].iloc[0] if len(comparison_df) > 0 else None
    worst_pred = comparison_df[comparison_df['category'] == 'Worst Predictions'].iloc[0] if len(comparison_df) > 0 else None
    
    if best_pred is not None:
        report_lines.append(f"Best Prediction: {best_pred['ticker']} on {best_pred['earnings_date']}")
        report_lines.append(f"  Actual: {best_pred['actual_revr']:.6f}, Predicted: {best_pred['predicted_revr']:.6f}")
        report_lines.append(f"  Error: {best_pred['prediction_error']:.6f}")
    
    if worst_pred is not None:
        report_lines.append(f"Worst Prediction: {worst_pred['ticker']} on {worst_pred['earnings_date']}")
        report_lines.append(f"  Actual: {worst_pred['actual_revr']:.6f}, Predicted: {worst_pred['predicted_revr']:.6f}")
        report_lines.append(f"  Error: {worst_pred['prediction_error']:.6f}")
    
    report_lines.append("")
    
    # Files Generated
    report_lines.append("FILES GENERATED")
    report_lines.append("-" * 40)
    report_lines.append("• rolling_regression_results.csv - Window-level performance metrics")
    report_lines.append("• rolling_regression_predictions.csv - All predicted vs actual values")
    report_lines.append("• rolling_regression_detailed_predictions.csv - Detailed predictions with errors")
    report_lines.append("• rolling_regression_predictions_summary.csv - Summary statistics by window/period")
    report_lines.append("• rolling_regression_predictions_comparison.csv - Best/worst predictions")
    report_lines.append("• rolling_regression_predictions_performance.csv - Performance metrics by window")
    report_lines.append("• rolling_regression_predictions_analysis.png - Visualization plots")
    report_lines.append("")
    
    # Write report to file
    report_file = 'data_files/rolling_regression_summary_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📋 Summary report saved to: {report_file}")
    
    # Print report to console
    print('\n'.join(report_lines))
    
    return report_file

def main():
    """
    Main function to run rolling regression analysis.
    Now uses both IEVR and normative_iv_rv_ratio as features.
    """
    print("📊 ROLLING REGRESSION ANALYSIS: IEVR + Normative IV/RV Ratio vs REVR")
    print("="*80)
    print("Features:")
    print("  • IEVR (Implied Earnings Volatility Ratio)")
    print("  • Normative IV/RV Ratio (Volatility Risk Premium Indicator)")
    print("Target: REVR (Realized Earnings Volatility Ratio)")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run rolling regression
    results, all_predictions = run_rolling_regression(df, windows)
    
    # Analyze results
    if results:
        results_df = analyze_results(results)
        print(f"\n🎉 Rolling regression analysis completed!")
        print(f"📊 Generated {len(results_df)} rolling windows")
        print(f"🔬 Model: REVR = α + β₁×IEVR + β₂×Normative_IV_RV_Ratio")
    else:
        print("❌ No results generated")

    # Save predictions to CSV
    predictions_output_file = 'data_files/rolling_regression_predictions.csv'
    all_predictions_df = save_predictions_to_csv(all_predictions, predictions_output_file)

    # Analyze predictions
    analyze_predictions(all_predictions_df)

    # Create visualizations
    create_prediction_visualizations(all_predictions_df)

    # Export detailed predictions
    detailed_df, summary_stats = export_detailed_predictions(all_predictions_df)

    # Create prediction comparison table
    comparison_df = create_prediction_comparison_table(all_predictions_df)

    # Create summary report
    create_summary_report(results_df, all_predictions_df, detailed_df, summary_stats, comparison_df)

if __name__ == "__main__":
    main()
