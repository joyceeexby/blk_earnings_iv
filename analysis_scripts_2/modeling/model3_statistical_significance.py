#!/usr/bin/env python3
"""
Model 3 Statistical Significance Analysis
Analyze the statistical significance of features in Linear Regression Model 3
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
    Load the final merged dataset and prepare it for analysis.
    """
    print("📊 LOADING AND PREPARING FINAL MERGED DATASET")
    print("="*60)
    
    # Load the final merged dataset
    file_path = 'data_files/final_merged_dataset.csv'
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

def calculate_statistical_significance(X, y, feature_names):
    """
    Calculate statistical significance for linear regression coefficients.
    """
    try:
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate residuals
        residuals = y - y_pred
        
        
        # Calculate degrees of freedom
        n = len(y)
        p = X.shape[1] + 1  # +1 for intercept
        df = n - p
        
        # Calculate residual standard error
        mse = np.sum(residuals**2) / df
        rmse = np.sqrt(mse)
        
        # Calculate standard errors for coefficients
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        
        # Standard errors for coefficients (excluding intercept)
        se_coef = np.sqrt(np.diag(XtX_inv)[1:] * mse)
        
        # Calculate t-statistics
        t_stats = model.coef_ / se_coef
        
        # Calculate p-values (two-tailed t-test)
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
        
        # Calculate confidence intervals (95%)
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = model.coef_ - t_critical * se_coef
        ci_upper = model.coef_ + t_critical * se_coef
        
        # Create results dictionary
        significance_results = {}
        for i, feature in enumerate(feature_names):
            significance_results[feature] = {
                'coefficient': model.coef_[i],
                'standard_error': se_coef[i],
                't_statistic': t_stats[i],
                'p_value': p_values[i],
                'ci_lower_95': ci_lower[i],
                'ci_upper_95': ci_upper[i],
                'is_significant_5pct': p_values[i] < 0.05,
                'is_significant_1pct': p_values[i] < 0.01
            }
        
        return significance_results, model.intercept_, rmse, r2_score(y, y_pred)
        
    except Exception as e:
        print(f"    ⚠️  Error calculating statistical significance: {e}")
        return None, None, None, None

def run_model3_with_statistical_significance(df, windows):
    """
    Run Model 3 with statistical significance analysis.
    """
    print(f"\n🔬 RUNNING MODEL 3 WITH STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    # Define Model 3 features
    model3_features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
    
    # Check which features are available
    available_features = [f for f in model3_features if f in df.columns]
    print(f"📋 Available features: {available_features}")
    print(f"📊 Target variable: revr")
    
    # Store results
    all_results = []
    all_significance = []
    
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
        
        # Prepare training data
        X_train = train_data[available_features].values
        y_train = train_data['revr'].values
        
        # Remove rows with NaN values in features
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train_clean = X_train[valid_mask]
        y_train_clean = y_train[valid_mask]
        
        if len(X_train_clean) < 30:
            print(f"  ⚠️  Insufficient clean training data - skipping window")
            continue
        
        # Calculate statistical significance
        significance_results, intercept, rmse, r2 = calculate_statistical_significance(
            X_train_clean, y_train_clean, available_features
        )
        
        if significance_results is None:
            print(f"  ⚠️  Could not calculate significance - skipping window")
            continue
        
        # Validation performance
        X_val = val_data[available_features].values
        y_val = val_data['revr'].values
        
        val_valid_mask = ~np.isnan(X_val).any(axis=1)
        X_val_clean = X_val[val_valid_mask]
        y_val_clean = y_val[val_valid_mask]
        
        if len(X_val_clean) < 5:
            print(f"  ⚠️  Insufficient clean validation data - skipping window")
            continue
        
        # Fit model for validation
        model = LinearRegression()
        model.fit(X_train_clean, y_train_clean)
        y_val_pred = model.predict(X_val_clean)
        
        val_r2 = r2_score(y_val_clean, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val_clean, y_val_pred))
        val_mae = mean_absolute_error(y_val_clean, y_val_pred)
        
        # Test performance
        X_test = test_data[available_features].values
        y_test = test_data['revr'].values
        
        test_valid_mask = ~np.isnan(X_test).any(axis=1)
        X_test_clean = X_test[test_valid_mask]
        y_test_clean = y_test[test_valid_mask]
        
        if len(X_test_clean) < 5:
            print(f"  ⚠️  Insufficient clean test data - skipping window")
            continue
        
        y_test_pred = model.predict(X_test_clean)
        
        test_r2 = r2_score(y_test_clean, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_clean, y_test_pred))
        test_mae = mean_absolute_error(y_test_clean, y_test_pred)
        
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
            'features_used': available_features,
            'intercept': intercept,
            'train_rmse': rmse,
            'train_r2': r2,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        all_results.append(window_result)
        
        # Store significance results
        for feature, sig_data in significance_results.items():
            feature_result = {
                'window_id': window['window_id'],
                'test_start': window['test_start'].strftime('%Y-%m'),
                'test_end': window['test_end'].strftime('%Y-%m'),
                'feature': feature,
                **sig_data
            }
            all_significance.append(feature_result)
        
        print(f"  ✅ Window completed: Val R²={val_r2:.4f}, Test R²={test_r2:.4f}")
    
    return all_results, all_significance

def analyze_statistical_significance(all_significance):
    """
    Analyze the statistical significance across all windows.
    """
    print(f"\n🔍 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    if not all_significance:
        print("❌ No significance data available")
        return None
    
    # Convert to DataFrame
    sig_df = pd.DataFrame(all_significance)
    
    # Group by feature and calculate summary statistics
    feature_summary = {}
    
    for feature in sig_df['feature'].unique():
        feature_data = sig_df[sig_df['feature'] == feature]
        
        feature_summary[feature] = {
            'windows': len(feature_data),
            'mean_coefficient': feature_data['coefficient'].mean(),
            'std_coefficient': feature_data['coefficient'].std(),
            'mean_t_statistic': feature_data['t_statistic'].mean(),
            'std_t_statistic': feature_data['t_statistic'].std(),
            'mean_p_value': feature_data['p_value'].mean(),
            'std_p_value': feature_data['p_value'].std(),
            'significant_5pct_count': feature_data['is_significant_5pct'].sum(),
            'significant_1pct_count': feature_data['is_significant_1pct'].sum(),
            'significant_5pct_pct': (feature_data['is_significant_5pct'].sum() / len(feature_data)) * 100,
            'significant_1pct_pct': (feature_data['is_significant_1pct'].sum() / len(feature_data)) * 100
        }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(feature_summary).T
    summary_df = summary_df.round(4)
    
    print("📊 FEATURE STATISTICAL SIGNIFICANCE SUMMARY:")
    print("-" * 80)
    print(summary_df)
    
    # Save summary
    summary_df.to_csv('data_files/model3_statistical_significance_summary.csv')
    print(f"\n💾 Statistical significance summary saved to: data_files/model3_statistical_significance_summary.csv")
    
    return summary_df

def create_significance_visualization(all_significance):
    """
    Create visualizations for statistical significance analysis.
    """
    print(f"\n📊 CREATING STATISTICAL SIGNIFICANCE VISUALIZATIONS")
    print("="*60)
    
    if not all_significance:
        print("❌ No significance data available")
        return None
    
    # Convert to DataFrame
    sig_df = pd.DataFrame(all_significance)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model 3: Statistical Significance Analysis', fontsize=16, fontweight='bold')
    
    # 1. P-value distribution by feature
    ax1 = axes[0, 0]
    features = sig_df['feature'].unique()
    
    for feature in features:
        feature_data = sig_df[sig_df['feature'] == feature]
        ax1.hist(feature_data['p_value'], alpha=0.7, label=feature, bins=20)
    
    ax1.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='5% Significance')
    ax1.axvline(x=0.01, color='orange', linestyle='--', alpha=0.7, label='1% Significance')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('P-value Distribution by Feature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. T-statistic distribution by feature
    ax2 = axes[0, 1]
    for feature in features:
        feature_data = sig_df[sig_df['feature'] == feature]
        ax2.hist(feature_data['t_statistic'], alpha=0.7, label=feature, bins=20)
    
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('T-statistic')
    ax2.set_ylabel('Frequency')
    ax2.set_title('T-statistic Distribution by Feature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Significance rate by feature
    ax3 = axes[1, 0]
    features_list = []
    sig_5pct_rates = []
    sig_1pct_rates = []
    
    for feature in features:
        feature_data = sig_df[sig_df['feature'] == feature]
        sig_5pct_rate = (feature_data['is_significant_5pct'].sum() / len(feature_data)) * 100
        sig_1pct_rate = (feature_data['is_significant_1pct'].sum() / len(feature_data)) * 100
        
        features_list.append(feature)
        sig_5pct_rates.append(sig_5pct_rate)
        sig_1pct_rates.append(sig_1pct_rate)
    
    x = np.arange(len(features_list))
    width = 0.35
    
    ax3.bar(x - width/2, sig_5pct_rates, width, label='5% Significance', alpha=0.7)
    ax3.bar(x + width/2, sig_1pct_rates, width, label='1% Significance', alpha=0.7)
    
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Significance Rate (%)')
    ax3.set_title('Feature Significance Rates')
    ax3.set_xticks(x)
    ax3.set_xticklabels(features_list, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Coefficient vs T-statistic scatter
    ax4 = axes[1, 1]
    for feature in features:
        feature_data = sig_df[sig_df['feature'] == feature]
        ax4.scatter(feature_data['coefficient'], feature_data['t_statistic'], 
                   alpha=0.7, label=feature, s=30)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Coefficient Value')
    ax4.set_ylabel('T-statistic')
    ax4.set_title('Coefficient vs T-statistic')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/model3_statistical_significance_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def main():
    """
    Main function to run Model 3 with statistical significance analysis.
    """
    print("🔬 MODEL 3 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create time windows
    windows = create_time_windows(df)
    
    # Run Model 3 with statistical significance
    all_results, all_significance = run_model3_with_statistical_significance(df, windows)
    
    # Analyze statistical significance
    if all_significance:
        summary_df = analyze_statistical_significance(all_significance)
        
        # Create visualizations
        create_significance_visualization(all_significance)
        
        print(f"\n🎉 Model 3 statistical significance analysis completed!")
        print(f"📊 Analyzed {len(all_results)} windows")
        print(f"📊 Generated significance data for {len(all_significance)} feature-window combinations")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()

