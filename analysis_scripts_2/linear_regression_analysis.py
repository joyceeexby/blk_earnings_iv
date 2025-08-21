#!/usr/bin/env python3
"""
Linear Regression Analysis: IEVR vs REVR
Using the merged dataset to analyze the relationship between implied and realized volatility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the merged dataset and prepare it for regression analysis.
    """
    print("📊 LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load merged dataset
    file_path = 'data_files/merged_revr_ievr_comprehensive.csv'
    
    if not pd.io.common.file_exists(file_path):
        print(f"❌ Merged dataset not found: {file_path}")
        print("Please run merge_revr_ievr.py first")
        return None
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {len(df):,} observations")
    
    # Check for missing values
    print(f"\n🔍 Data Quality Check:")
    missing_revr = df['revr'].isna().sum()
    missing_ievr = df['ievr'].isna().sum()
    print(f"  Missing REVR values: {missing_revr}")
    print(f"  Missing IEVR values: {missing_ievr}")
    
    # Remove any rows with missing values
    df_clean = df.dropna(subset=['revr', 'ievr'])
    print(f"  Clean observations: {len(df_clean):,}")
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"  REVR - Mean: {df_clean['revr'].mean():.3f}, Std: {df_clean['revr'].std():.3f}")
    print(f"  IEVR - Mean: {df_clean['ievr'].mean():.3f}, Std: {df_clean['ievr'].std():.3f}")
    
    # Check for extreme outliers
    print(f"\n🔍 Outlier Detection:")
    
    # Calculate z-scores
    revr_zscore = np.abs(stats.zscore(df_clean['revr']))
    ievr_zscore = np.abs(stats.zscore(df_clean['ievr']))
    
    # Count extreme outliers (z-score > 3)
    extreme_revr = np.sum(revr_zscore > 3)
    extreme_ievr = np.sum(ievr_zscore > 3)
    
    print(f"  Extreme REVR outliers (z-score > 3): {extreme_revr}")
    print(f"  Extreme IEVR outliers (z-score > 3): {extreme_ievr}")
    
    # Remove extreme outliers
    df_final = df_clean[
        (revr_zscore <= 3) & 
        (ievr_zscore <= 3)
    ].copy()
    
    print(f"  Final clean dataset: {len(df_final):,} observations")
    
    return df_final

def run_linear_regression(df):
    """
    Run linear regression: REVR = α + β * IEVR + ε
    """
    print(f"\n🔬 LINEAR REGRESSION ANALYSIS")
    print("="*60)
    print("Model: REVR = α + β * IEVR + ε")
    
    # Prepare variables
    X = df['ievr'].values.reshape(-1, 1)  # Independent variable (IEVR)
    y = df['revr'].values  # Dependent variable (REVR)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate correlation
    correlation = df['revr'].corr(df['ievr'])
    
    # Print results
    print(f"\n📊 REGRESSION RESULTS:")
    print(f"  Intercept (α): {intercept:.4f}")
    print(f"  Slope (β): {slope:.4f}")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # Interpret results
    print(f"\n📝 INTERPRETATION:")
    print(f"  For every 1 unit increase in IEVR, REVR changes by {slope:.4f} units")
    if slope > 0:
        print(f"  Positive relationship: Higher implied volatility predicts higher realized volatility")
    else:
        print(f"  Negative relationship: Higher implied volatility predicts lower realized volatility")
    
    print(f"  R² = {r2:.4f}: {r2*100:.1f}% of variance in REVR is explained by IEVR")
    
    return model, X, y, y_pred, {
        'intercept': intercept,
        'slope': slope,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation
    }

def create_visualizations(df, model, X, y, y_pred, results):
    """
    Create visualizations for the regression analysis.
    """
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Analysis: IEVR vs REVR', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(df['ievr'], df['revr'], alpha=0.6, s=20, color='steelblue')
    
    # Plot regression line
    X_line = np.linspace(df['ievr'].min(), df['ievr'].max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax1.plot(X_line, y_line, color='red', linewidth=2, label=f'REVR = {results["intercept"]:.3f} + {results["slope"]:.3f} × IEVR')
    
    ax1.set_xlabel('IEVR (Implied Earnings Volatility Ratio)')
    ax1.set_ylabel('REVR (Realized Earnings Volatility Ratio)')
    ax1.set_title('Scatter Plot with Regression Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Predicted REVR')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot for normality
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/linear_regression_ievr_vs_revr.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return residuals

def additional_analyses(df, results):
    """
    Perform additional analyses and diagnostics.
    """
    print(f"\n🔍 ADDITIONAL ANALYSES")
    print("="*60)
    
    # 1. Season-based analysis
    if 'season' in df.columns:
        print(f"📅 SEASON-BASED ANALYSIS:")
        season_stats = df.groupby('season').agg({
            'revr': ['mean', 'std', 'count'],
            'ievr': ['mean', 'std']
        }).round(4)
        
        print(season_stats.head(10))
        
        # Save season analysis
        season_file = 'data_files/season_based_analysis.csv'
        season_stats.to_csv(season_file)
        print(f"💾 Season analysis saved to: {season_file}")
    
    # 2. Stock-based analysis
    print(f"\n📈 STOCK-BASED ANALYSIS:")
    stock_stats = df.groupby('ticker').agg({
        'revr': ['mean', 'std', 'count'],
        'ievr': ['mean', 'std']
    }).round(4)
    
    # Filter stocks with sufficient observations
    min_obs = 10
    stock_stats_filtered = stock_stats[stock_stats[('revr', 'count')] >= min_obs]
    
    print(f"Stocks with ≥{min_obs} observations: {len(stock_stats_filtered)}")
    print(stock_stats_filtered.head(10))
    
    # Save stock analysis
    stock_file = 'data_files/stock_based_analysis.csv'
    stock_stats_filtered.to_csv(stock_file)
    print(f"💾 Stock analysis saved to: {stock_file}")
    
    # 3. Time trend analysis
    if 'year' in df.columns:
        print(f"\n📈 TIME TREND ANALYSIS:")
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        year_stats = df.groupby('year').agg({
            'revr': ['mean', 'std', 'count'],
            'ievr': ['mean', 'std']
        }).round(4)
        
        print(year_stats)
        
        # Save time trend analysis
        time_file = 'data_files/time_trend_analysis.csv'
        year_stats.to_csv(time_file)
        print(f"💾 Time trend analysis saved to: {time_file}")
    
    # 4. Model diagnostics
    print(f"\n🔬 MODEL DIAGNOSTICS:")
    
    # Calculate residuals
    X = df['ievr'].values.reshape(-1, 1)
    y = df['revr'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Normality test
    normality_stat, normality_p = stats.normaltest(residuals)
    print(f"  Normality test (D'Agostino): p-value = {normality_p:.4f}")
    
    # Homoscedasticity test (Breusch-Pagan equivalent)
    # Simple approach: check if residuals variance changes with predicted values
    bp_model = LinearRegression()
    bp_model.fit(y_pred.reshape(-1, 1), residuals**2)
    bp_r2 = bp_model.score(y_pred.reshape(-1, 1), residuals**2)
    print(f"  Homoscedasticity check: R² of residuals² vs predicted = {bp_r2:.4f}")
    
    # Save diagnostics
    diagnostics = {
        'normality_test_statistic': normality_stat,
        'normality_test_pvalue': normality_p,
        'homoscedasticity_r2': bp_r2,
        'total_observations': len(df),
        'r_squared': results['r2'],
        'correlation': results['correlation']
    }
    
    diagnostics_df = pd.DataFrame([diagnostics])
    diagnostics_file = 'data_files/regression_diagnostics.csv'
    diagnostics_df.to_csv(diagnostics_file, index=False)
    print(f"💾 Diagnostics saved to: {diagnostics_file}")

def save_regression_results(results, df):
    """
    Save comprehensive regression results.
    """
    print(f"\n💾 SAVING REGRESSION RESULTS")
    print("="*60)
    
    # Create comprehensive results summary
    summary = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'merged_revr_ievr_comprehensive.csv',
        'total_observations': len(df),
        'independent_variable': 'IEVR (Implied Earnings Volatility Ratio)',
        'dependent_variable': 'REVR (Realized Earnings Volatility Ratio)',
        'model_equation': f'REVR = {results["intercept"]:.4f} + {results["slope"]:.4f} × IEVR',
        'intercept': results['intercept'],
        'slope': results['slope'],
        'r_squared': results['r2'],
        'correlation': results['correlation'],
        'mse': results['mse'],
        'rmse': results['rmse'],
        'mae': results['mae'],
        'interpretation': f'For every 1 unit increase in IEVR, REVR changes by {results["slope"]:.4f} units. R² = {results["r2"]:.4f} means {results["r2"]*100:.1f}% of variance in REVR is explained by IEVR.'
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_file = 'data_files/regression_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Regression summary saved to: {summary_file}")
    
    # Print final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"  Model: {summary['model_equation']}")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  Correlation: {results['correlation']:.4f}")
    print(f"  Observations: {len(df):,}")
    print(f"  All results saved to data_files/ directory")

def main():
    """
    Main function to run the complete linear regression analysis.
    """
    print("🔬 LINEAR REGRESSION: IEVR vs REVR")
    print("="*60)
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data()
        if df is None:
            return
        
        # 2. Run linear regression
        model, X, y, y_pred, results = run_linear_regression(df)
        
        # 3. Create visualizations
        residuals = create_visualizations(df, model, X, y, y_pred, results)
        
        # 4. Additional analyses
        additional_analyses(df, results)
        
        # 5. Save results
        save_regression_results(results, df)
        
        print(f"\n🎉 Linear regression analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
