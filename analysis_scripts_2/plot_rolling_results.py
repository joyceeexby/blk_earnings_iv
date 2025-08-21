#!/usr/bin/env python3
"""
Plot Rolling Regression Results
Visualize test R², test RMSE, sample size, and model performance over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """
    Load the rolling regression results.
    """
    print("📊 LOADING ROLLING REGRESSION RESULTS")
    print("="*60)
    
    file_path = 'data_files/rolling_regression_results.csv'
    df = pd.read_csv(file_path)
    print(f"✅ Loaded results: {len(df)} windows")
    
    # Convert date strings to datetime for better plotting
    df['train_end_date'] = pd.to_datetime(df['train_end'] + '-01')
    df['val_end_date'] = pd.to_datetime(df['val_end'] + '-01')
    df['test_end_date'] = pd.to_datetime(df['test_end'] + '-01')
    
    # Extract year and month for x-axis
    df['plot_date'] = df['test_end_date']  # Use test end date for x-axis
    
    return df

def create_performance_plots(df):
    """
    Create comprehensive performance plots.
    """
    print(f"\n📊 CREATING PERFORMANCE PLOTS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rolling Regression Performance Over Time', fontsize=16, fontweight='bold')
    
    # 1. Test R² over time
    ax1 = axes[0, 0]
    ax1.plot(df['plot_date'], df['test_r2'], marker='o', linewidth=2, markersize=6, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No predictive power (R²=0)')
    ax1.set_xlabel('Test Period End Date')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Test R² Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add annotations for best and worst performance
    best_idx = df['test_r2'].idxmax()
    worst_idx = df['test_r2'].idxmin()
    
    ax1.annotate(f'Best: {df.loc[best_idx, "test_r2"]:.3f}', 
                 xy=(df.loc[best_idx, 'plot_date'], df.loc[best_idx, 'test_r2']),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                 fontsize=9)
    
    ax1.annotate(f'Worst: {df.loc[worst_idx, "test_r2"]:.3f}', 
                 xy=(df.loc[worst_idx, 'plot_date'], df.loc[worst_idx, 'test_r2']),
                 xytext=(10, -15), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                 fontsize=9)
    
    # 2. Test RMSE over time
    ax2 = axes[0, 1]
    ax2.plot(df['plot_date'], df['test_rmse'], marker='s', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Test Period End Date')
    ax2.set_ylabel('Test RMSE')
    ax2.set_title('Test RMSE Performance Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add mean line
    mean_rmse = df['test_rmse'].mean()
    ax2.axhline(y=mean_rmse, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean RMSE: {mean_rmse:.3f}')
    ax2.legend()
    
    # 3. Sample sizes over time
    ax3 = axes[1, 0]
    x_pos = np.arange(len(df))
    width = 0.35
    
    ax3.bar(x_pos - width/2, df['train_obs'], width, label='Training', alpha=0.7, color='lightblue')
    ax3.bar(x_pos + width/2, df['test_obs'], width, label='Testing', alpha=0.7, color='lightcoral')
    
    ax3.set_xlabel('Window Number')
    ax3.set_ylabel('Number of Observations')
    ax3.set_title('Sample Sizes by Window')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'W{i+1}' for i in range(len(df))], rotation=45)
    
    # 4. Model coefficients over time
    ax4 = axes[1, 1]
    ax4.plot(df['plot_date'], df['intercept'], marker='o', linewidth=2, markersize=6, 
             color='green', label='Intercept (α)', alpha=0.8)
    ax4.plot(df['plot_date'], df['slope'], marker='s', linewidth=2, markersize=6, 
             color='purple', label='Slope (β)', alpha=0.8)
    
    ax4.set_xlabel('Test Period End Date')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Model Coefficients Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add mean lines
    mean_intercept = df['intercept'].mean()
    mean_slope = df['slope'].mean()
    ax4.axhline(y=mean_intercept, color='green', linestyle='--', alpha=0.5, 
                label=f'Mean α: {mean_intercept:.3f}')
    ax4.axhline(y=mean_slope, color='purple', linestyle='--', alpha=0.5, 
                label=f'Mean β: {mean_slope:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/rolling_regression_performance_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Performance plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_summary_statistics(df):
    """
    Create summary statistics and additional insights.
    """
    print(f"\n📊 SUMMARY STATISTICS")
    print("="*60)
    
    # Overall performance
    print(f"📈 OVERALL PERFORMANCE:")
    print(f"  Average Test R²: {df['test_r2'].mean():.4f}")
    print(f"  Average Test RMSE: {df['test_rmse'].mean():.4f}")
    print(f"  Best Test R²: {df['test_r2'].max():.4f} (Window {df['test_r2'].idxmax() + 1})")
    print(f"  Worst Test R²: {df['test_r2'].min():.4f} (Window {df['test_r2'].idxmin() + 1})")
    
    # Performance distribution
    print(f"\n📊 PERFORMANCE DISTRIBUTION:")
    positive_r2 = (df['test_r2'] > 0).sum()
    negative_r2 = (df['test_r2'] <= 0).sum()
    print(f"  Windows with positive R²: {positive_r2}/{len(df)} ({positive_r2/len(df)*100:.1f}%)")
    print(f"  Windows with negative R²: {negative_r2}/{len(df)} ({negative_r2/len(df)*100:.1f}%)")
    
    # Coefficient stability
    print(f"\n🔢 COEFFICIENT STABILITY:")
    print(f"  Intercept - Mean: {df['intercept'].mean():.4f}, Std: {df['intercept'].std():.4f}")
    print(f"  Slope - Mean: {df['slope'].mean():.4f}, Std: {df['slope'].std():.4f}")
    
    # Sample size analysis
    print(f"\n📏 SAMPLE SIZE ANALYSIS:")
    print(f"  Average training size: {df['train_obs'].mean():.0f}")
    print(f"  Average testing size: {df['test_obs'].mean():.0f}")
    print(f"  Total observations used: {df['train_obs'].sum() + df['test_obs'].sum():,}")
    
    # Time period analysis
    print(f"\n📅 TIME PERIOD ANALYSIS:")
    df['year'] = df['test_end_date'].dt.year
    yearly_perf = df.groupby('year').agg({
        'test_r2': ['mean', 'std', 'count'],
        'test_rmse': ['mean', 'std']
    }).round(4)
    
    print("Yearly Performance Summary:")
    print(yearly_perf)
    
    return yearly_perf

def create_additional_plots(df):
    """
    Create additional insightful plots.
    """
    print(f"\n📊 CREATING ADDITIONAL PLOTS")
    print("="*60)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Additional Rolling Regression Insights', fontsize=16, fontweight='bold')
    
    # 1. R² vs Sample Size scatter plot
    ax1.scatter(df['test_obs'], df['test_r2'], alpha=0.7, s=60, color='steelblue')
    ax1.set_xlabel('Test Sample Size')
    ax1.set_ylabel('Test R²')
    ax1.set_title('R² vs Sample Size')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df['test_obs'], df['test_r2'], 1)
    p = np.poly1d(z)
    ax1.plot(df['test_obs'], p(df['test_obs']), "r--", alpha=0.8, 
             label=f'Trend: y={z[0]:.6f}x+{z[1]:.3f}')
    ax1.legend()
    
    # 2. Coefficient correlation plot
    ax2.scatter(df['intercept'], df['slope'], alpha=0.7, s=60, color='purple')
    ax2.set_xlabel('Intercept (α)')
    ax2.set_ylabel('Slope (β)')
    ax2.set_title('Intercept vs Slope Correlation')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df['intercept'].corr(df['slope'])
    ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/rolling_regression_additional_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Additional plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def main():
    """
    Main function to create all plots and analysis.
    """
    print("📊 PLOTTING ROLLING REGRESSION RESULTS")
    print("="*60)
    
    try:
        # 1. Load results
        df = load_results()
        
        # 2. Create main performance plots
        main_fig = create_performance_plots(df)
        
        # 3. Create summary statistics
        yearly_perf = create_summary_statistics(df)
        
        # 4. Create additional plots
        additional_fig = create_additional_plots(df)
        
        # 5. Save yearly performance
        yearly_file = 'data_files/yearly_performance_summary.csv'
        yearly_perf.to_csv(yearly_file)
        print(f"\n💾 Yearly performance summary saved to: {yearly_file}")
        
        print(f"\n🎉 All plots and analysis completed successfully!")
        print(f"📁 Check the data_files/ directory for all output files")
        
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
