#!/usr/bin/env python3
"""
Rolling Regression Analysis: IEVR vs REVR
Walk-forward validation with 5-year training, 6-month validation, 6-month testing
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
        
        # Train model
        X_train = train_data['ievr'].values.reshape(-1, 1)
        y_train = train_data['revr'].values
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Get coefficients
        intercept = model.intercept_
        slope = model.coef_[0]
        
        # Validation performance
        X_val = val_data['ievr'].values.reshape(-1, 1)
        y_val = val_data['revr'].values
        y_val_pred = model.predict(X_val)
        
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Test performance
        X_test = test_data['ievr'].values.reshape(-1, 1)
        y_test = test_data['revr'].values
        y_test_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
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
            'slope': slope,
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
    
    return results

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
    print(f"  Slope - Mean: {results_df['slope'].mean():.4f}, Std: {results_df['slope'].std():.4f}")
    
    # Performance over time
    print(f"\n📅 PERFORMANCE OVER TIME:")
    results_df['train_end_year'] = pd.to_numeric(results_df['train_end'].str[:4])
    yearly_perf = results_df.groupby('train_end_year').agg({
        'val_r2': 'mean',
        'test_r2': 'mean',
        'val_rmse': 'mean',
        'test_rmse': 'mean'
    }).round(4)
    
    print(yearly_perf)
    
    # Best and worst performing windows
    print(f"\n🏆 BEST PERFORMING WINDOWS:")
    best_val = results_df.loc[results_df['val_r2'].idxmax()]
    best_test = results_df.loc[results_df['test_r2'].idxmax()]
    
    print(f"  Best Validation R²: {best_val['val_r2']:.4f} (Window {best_val['window_id']})")
    print(f"  Best Test R²: {best_test['test_r2']:.4f} (Window {best_test['window_id']})")
    
    # Save results
    results_file = 'data_files/rolling_regression_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    return results_df

def main():
    """
    Main function to run rolling regression analysis.
    """
    print("🔄 ROLLING REGRESSION ANALYSIS: IEVR vs REVR")
    print("="*60)
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data()
        
        # 2. Create time windows
        windows = create_time_windows(df, train_years=5, val_months=6, test_months=6)
        
        # 3. Run rolling regression
        results = run_rolling_regression(df, windows)
        
        # 4. Analyze results
        results_df = analyze_results(results)
        
        print(f"\n🎉 Rolling regression analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
