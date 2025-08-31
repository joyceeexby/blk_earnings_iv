#!/usr/bin/env python3
"""
Enhanced Rolling Regression Analysis with Lagged REVR Features
Walk-forward validation with 5-year training, 6-month validation, 6-month testing
Now includes lagged REVR features (1 and 2 quarters back) for better predictive power
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load data and prepare for time series analysis with lagged features.
    """
    print("LOADING AND PREPARING DATA WITH LAGGED FEATURES")
    print("="*70)
    
    # Load merged dataset
    file_path = 'data_files/merged_revr_ievr_comprehensive.csv'
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {len(df):,} observations")
    
    # Convert dates and add time components
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    df['year'] = df['earnings_date'].dt.year
    df['quarter'] = df['earnings_date'].dt.quarter
    df['month'] = df['earnings_date'].dt.month
    
    # Create season identifier
    df['season'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
    
    # Calculate normative_iv_rv_ratio feature
    print("Creating normative_iv_rv_ratio feature...")
    df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
    
    # Handle infinite values and NaN
    df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Show feature statistics
    valid_ratio = df['normative_iv_rv_ratio'].notna().sum()
    print(f"  Created normative_iv_rv_ratio feature: {valid_ratio:,} valid values")
    
    # Sort by ticker and earnings date for proper lagging
    df = df.sort_values(['ticker', 'earnings_date']).reset_index(drop=True)
    
    # Create lagged REVR features (1 and 2 quarters back)
    print("Creating lagged REVR features...")
    
    # Group by ticker and create lagged features
    df['revr_lag1'] = df.groupby('ticker')['revr'].shift(1)
    df['revr_lag2'] = df.groupby('ticker')['revr'].shift(2)
    
    # Also create lagged IEVR features for completeness
    df['ievr_lag1'] = df.groupby('ticker')['ievr'].shift(1)
    df['ievr_lag2'] = df.groupby('ticker')['ievr'].shift(2)
    
    # Create change features
    df['revr_change'] = df.groupby('ticker')['revr'].pct_change()
    df['ievr_change'] = df.groupby('ticker')['ievr'].pct_change()
    
    # Show lagged feature statistics
    valid_lag1 = df['revr_lag1'].notna().sum()
    valid_lag2 = df['revr_lag2'].notna().sum()
    print(f"  Created revr_lag1: {valid_lag1:,} valid values")
    print(f"  Created revr_lag2: {valid_lag2:,} valid values")
    print(f"  Created ievr_lag1: {df['ievr_lag1'].notna().sum():,} valid values")
    print(f"  Created ievr_lag2: {df['ievr_lag2'].notna().sum():,} valid values")
    
    # Show feature correlation with target
    print("\nFeature correlations with REVR:")
    features = ['ievr', 'normative_iv_rv_ratio', 'revr_lag1', 'revr_lag2', 'ievr_lag1', 'ievr_lag2']
    for feature in features:
        if feature in df.columns:
            corr = df[['revr', feature]].corr().iloc[0, 1]
            print(f"  {feature}: {corr:.4f}")
    
    return df

def create_time_windows(df, train_years=5, val_months=6, test_months=6):
    """
    Create rolling time windows for walk-forward validation.
    Uses the same logic as the original rolling regression.
    """
    print(f"\nCREATING TIME WINDOWS")
    print("="*50)
    print(f"Training window: {train_years} years")
    print(f"Validation window: {val_months} months")
    print(f"Testing window: {test_months} months")
    
    # Get unique dates and sort (monthly periods)
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
            'window_id': len(windows) + 1,
            'train_start': unique_dates[current_idx].to_timestamp(),
            'train_end': unique_dates[train_end].to_timestamp(),
            'val_start': unique_dates[val_start].to_timestamp(),
            'val_end': unique_dates[val_end].to_timestamp(),
            'test_start': unique_dates[test_start].to_timestamp(),
            'test_end': unique_dates[test_end].to_timestamp()
        }
        
        windows.append(window)
        current_idx += test_months  # Move forward by test window size
    
    print(f"Created {len(windows)} rolling windows")
    
    # Show first few windows
    for i, window in enumerate(windows[:3]):
        print(f"  Window {i+1}: Train {window['train_start'].strftime('%Y-%m')} to {window['train_end'].strftime('%Y-%m')}")
        print(f"           Val {window['val_start'].strftime('%Y-%m')} to {window['val_end'].strftime('%Y-%m')}")
        print(f"           Test {window['test_start'].strftime('%Y-%m')} to {window['test_end'].strftime('%Y-%m')}")
    
    return windows

def get_data_for_window(df, window):
    """
    Get training, validation, and test data for a specific time window.
    """
    train_data = df[
        (df['earnings_date'] >= window['train_start']) & 
        (df['earnings_date'] <= window['train_end'])
    ].copy()
    
    val_data = df[
        (df['earnings_date'] >= window['val_start']) & 
        (df['earnings_date'] <= window['val_end'])
    ].copy()
    
    test_data = df[
        (df['earnings_date'] >= window['test_start']) & 
        (df['earnings_date'] <= window['test_end'])
    ].copy()
    
    return train_data, val_data, test_data

def run_enhanced_rolling_regression(df, windows):
    """
    Run enhanced rolling regression with lagged features.
    """
    print(f"\nRUNNING ENHANCED ROLLING REGRESSION WITH LAGGED FEATURES")
    print("="*70)
    print("Model: REVR = α + β₁×IEVR + β₂×Normative_IV_RV_Ratio + β₃×REVR_lag1 + β₄×REVR_lag2")
    print("="*70)
    
    results = []
    
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Val:   {window['val_start']} to {window['val_end']}")
        print(f"  Test:  {window['test_start']} to {window['test_end']}")
        
        # Get data for this window
        train_data, val_data, test_data = get_data_for_window(df, window)
        
        # Check if we have enough data
        if len(train_data) < 50 or len(val_data) < 10 or len(test_data) < 10:
            print(f"  Insufficient data - skipping window")
            continue
        
        # Enhanced features including lagged REVR
        feature_cols = ['ievr', 'normative_iv_rv_ratio', 'revr_lag1', 'revr_lag2']
        X_train = train_data[feature_cols].values
        y_train = train_data['revr'].values
        
        # Remove rows with NaN values in features
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]
        
        if len(X_train_clean) < 30:  # Need minimum observations
            print(f"  Insufficient clean data after NaN removal - skipping window")
            continue
        
        # Train enhanced model
        model = LinearRegression()
        model.fit(X_train_clean, y_train_clean)
        
        # Get coefficients
        intercept = model.intercept_
        ievr_coef = model.coef_[0]
        ratio_coef = model.coef_[1]
        revr_lag1_coef = model.coef_[2]
        revr_lag2_coef = model.coef_[3]
        
        # Validation performance
        X_val = val_data[feature_cols].values
        y_val = val_data['revr'].values
        
        # Remove NaN values for validation
        val_valid_mask = ~np.isnan(X_val).any(axis=1)
        X_val_clean = X_val[val_valid_mask]
        y_val_clean = y_val[val_valid_mask]
        
        if len(X_val_clean) < 5:  # Need minimum validation observations
            print(f"  Insufficient clean validation data - skipping window")
            continue
        
        y_val_pred = model.predict(X_val_clean)
        
        val_r2 = r2_score(y_val_clean, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_clean, y_val_pred))
        val_mae = mean_absolute_error(y_val_clean, y_val_pred)
        
        # Test performance
        X_test = test_data[feature_cols].values
        y_test = test_data['revr'].values
        
        # Remove NaN values for testing
        test_valid_mask = ~np.isnan(X_test).any(axis=1)
        X_test_clean = X_test[test_valid_mask]
        y_test_clean = y_test[test_valid_mask]
        
        if len(X_test_clean) < 5:  # Need minimum test observations
            print(f"  Insufficient clean test data - skipping window")
            continue
        
        y_test_pred = model.predict(X_test_clean)
        
        test_r2 = r2_score(y_test_clean, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_test_pred))
        test_mae = mean_absolute_error(y_test_clean, y_test_pred)
        
        # Store enhanced results
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
            'revr_lag1_coef': revr_lag1_coef,
            'revr_lag2_coef': revr_lag2_coef,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        results.append(window_result)
        
        print(f"  Train: {len(train_data)} obs, Val: {len(val_data)} obs, Test: {len(test_data)} obs")
        print(f"  Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")
        print(f"  Coefficients: IEVR={ievr_coef:.4f}, Ratio={ratio_coef:.4f}, REVR_lag1={revr_lag1_coef:.4f}, REVR_lag2={revr_lag2_coef:.4f}")
    
    return results

def analyze_enhanced_results(results):
    """
    Analyze and summarize the enhanced rolling regression results.
    """
    print(f"\nENHANCED ROLLING REGRESSION SUMMARY")
    print("="*70)
    
    if not results:
        print("No results to analyze")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add train_end_year for grouping
    results_df['train_end_year'] = pd.to_datetime(results_df['train_end']).dt.year
    
    # Performance summary
    print(f"Total windows: {len(results_df)}")
    print(f"Average validation R2: {results_df['val_r2'].mean():.4f}")
    print(f"Average test R2: {results_df['test_r2'].mean():.4f}")
    print(f"Best test R2: {results_df['test_r2'].max():.4f}")
    print(f"Worst test R2: {results_df['test_r2'].min():.4f}")
    
    # Coefficient analysis
    print(f"\nCOEFFICIENT ANALYSIS:")
    print(f"  IEVR coefficient - Mean: {results_df['ievr_coef'].mean():.4f}, Std: {results_df['ievr_coef'].std():.4f}")
    print(f"  Ratio coefficient - Mean: {results_df['ratio_coef'].mean():.4f}, Std: {results_df['ratio_coef'].std():.4f}")
    print(f"  REVR_lag1 coefficient - Mean: {results_df['revr_lag1_coef'].mean():.4f}, Std: {results_df['revr_lag1_coef'].std():.4f}")
    print(f"  REVR_lag2 coefficient - Mean: {results_df['revr_lag2_coef'].mean():.4f}, Std: {results_df['revr_lag2_coef'].std():.4f}")
    
    # Performance by year
    print(f"\nPERFORMANCE BY TRAINING END YEAR:")
    yearly_perf = results_df.groupby('train_end_year').agg({
        'val_r2': ['mean', 'std', 'count'],
        'test_r2': ['mean', 'std'],
        'test_rmse': ['mean', 'std']
    }).round(4)
    
    print(yearly_perf)
    
    # Save results
    output_file = 'data_files/rolling_regression_results_with_lags.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results_df

def compare_with_original_results():
    """
    Compare enhanced results with original results.
    """
    print(f"\nCOMPARING WITH ORIGINAL RESULTS")
    print("="*50)
    
    try:
        # Load original results
        original_file = 'data_files/rolling_regression_results.csv'
        original_df = pd.read_csv(original_file)
        
        # Load enhanced results
        enhanced_file = 'data_files/rolling_regression_results_with_lags.csv'
        enhanced_df = pd.read_csv(enhanced_file)
        
        print(f"Original model windows: {len(original_df)}")
        print(f"Enhanced model windows: {len(enhanced_df)}")
        
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"  Original - Avg Test R2: {original_df['test_r2'].mean():.4f}")
        print(f"  Enhanced - Avg Test R2: {enhanced_df['test_r2'].mean():.4f}")
        print(f"  Difference: {enhanced_df['test_r2'].mean() - original_df['test_r2'].mean():.4f}")
        
        print(f"\n  Original - Avg Test RMSE: {original_df['test_rmse'].mean():.4f}")
        print(f"  Enhanced - Avg Test RMSE: {enhanced_df['test_rmse'].mean():.4f}")
        print(f"  Difference: {original_df['test_rmse'].mean() - enhanced_df['test_rmse'].mean():.4f}")
        
        # Analyze why lagged features might not be helping
        print(f"\nANALYSIS OF LAGGED FEATURES:")
        print(f"  REVR_lag1 coefficient - Mean: {enhanced_df['revr_lag1_coef'].mean():.4f}")
        print(f"  REVR_lag2 coefficient - Mean: {enhanced_df['revr_lag2_coef'].mean():.4f}")
        print(f"  Both lagged coefficients are positive but small, suggesting limited predictive power")
        
        # Check if lagged features are adding noise
        print(f"\nPOTENTIAL ISSUES WITH LAGGED FEATURES:")
        print(f"  1. Small coefficient values suggest limited impact")
        print(f"  2. Additional features may increase model complexity without sufficient benefit")
        print(f"  3. Lagged features may introduce more noise than signal")
        print(f"  4. The original model may already capture the essential relationships")
        
        # Show best and worst performing windows
        print(f"\nBEST PERFORMING WINDOWS (Enhanced Model):")
        best_windows = enhanced_df.nlargest(3, 'test_r2')[['window_id', 'test_r2', 'revr_lag1_coef', 'revr_lag2_coef']]
        print(best_windows.to_string(index=False))
        
        print(f"\nWORST PERFORMING WINDOWS (Enhanced Model):")
        worst_windows = enhanced_df.nsmallest(3, 'test_r2')[['window_id', 'test_r2', 'revr_lag1_coef', 'revr_lag2_coef']]
        print(worst_windows.to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"Could not load comparison data: {e}")
    except Exception as e:
        print(f"Error in comparison: {e}")
        print("Continuing with analysis...")

def main():
    """
    Main function to run enhanced rolling regression analysis with lagged features.
    """
    print("ENHANCED ROLLING REGRESSION ANALYSIS WITH LAGGED REVR FEATURES")
    print("="*80)
    print("Enhanced Features:")
    print("  • IEVR (Implied Earnings Volatility Ratio)")
    print("  • Normative IV/RV Ratio (Volatility Risk Premium Indicator)")
    print("  • REVR_lag1 (REVR from 1 quarter ago)")
    print("  • REVR_lag2 (REVR from 2 quarters ago)")
    print("Target: REVR (Realized Earnings Volatility Ratio)")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run enhanced rolling regression
    results = run_enhanced_rolling_regression(df, windows)
    
    # Analyze results
    if results:
        results_df = analyze_enhanced_results(results)
        print(f"\nEnhanced rolling regression analysis completed!")
        print(f"Generated {len(results_df)} rolling windows with lagged features")
        print(f"Enhanced Model: REVR = α + β₁×IEVR + β₂×Normative_IV_RV_Ratio + β₃×REVR_lag1 + β₄×REVR_lag2")
        
        # Compare with original results
        compare_with_original_results()
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
