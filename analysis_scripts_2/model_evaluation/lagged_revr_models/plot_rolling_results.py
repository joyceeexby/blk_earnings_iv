#!/usr/bin/env python3
"""
Plot Rolling Regression Results
Visualize test R2, test RMSE, sample size, and model coefficients over time
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
    print("LOADING ROLLING REGRESSION RESULTS")
    print("="*60)
    
    file_path = 'data_files/rolling_regression_results.csv'
    df = pd.read_csv(file_path)
    print(f"Loaded results: {len(df)} windows")
    
    # Convert date strings to datetime for better plotting
    # The dates are in format 'YYYY-MM', so we'll parse them properly
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
    print(f"\nCREATING PERFORMANCE PLOTS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rolling Regression Performance Over Time', fontsize=16, fontweight='bold')
    
    # 1. Test R2 over time
    ax1 = axes[0, 0]
    ax1.plot(df['plot_date'], df['test_r2'], marker='o', linewidth=2, markersize=6, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No predictive power (R2=0)')
    ax1.set_xlabel('Test Period End Date')
    ax1.set_ylabel('Test R2')
    ax1.set_title('Test R2 Performance Over Time')
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
    
    # 4. Model coefficients over time (updated to use actual column names)
    ax4 = axes[1, 1]
    ax4.plot(df['plot_date'], df['intercept'], marker='o', linewidth=2, markersize=6, 
             color='green', label='Intercept (alpha)', alpha=0.8)
    ax4.plot(df['plot_date'], df['ievr_coef'], marker='s', linewidth=2, markersize=6, 
             color='purple', label='IEVR Coefficient (beta1)', alpha=0.8)
    ax4.plot(df['plot_date'], df['ratio_coef'], marker='^', linewidth=2, markersize=6, 
             color='red', label='Ratio Coefficient (beta2)', alpha=0.8)
    
    ax4.set_xlabel('Test Period End Date')
    ax4.set_ylabel('Coefficient Value')
    ax4.set_title('Model Coefficients Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add mean lines
    mean_intercept = df['intercept'].mean()
    mean_ievr = df['ievr_coef'].mean()
    mean_ratio = df['ratio_coef'].mean()
    ax4.axhline(y=mean_intercept, color='green', linestyle='--', alpha=0.5, 
                label=f'Mean alpha: {mean_intercept:.3f}')
    ax4.axhline(y=mean_ievr, color='purple', linestyle='--', alpha=0.5, 
                label=f'Mean beta1: {mean_ievr:.3f}')
    ax4.axhline(y=mean_ratio, color='red', linestyle='--', alpha=0.5, 
                label=f'Mean beta2: {mean_ratio:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/rolling_regression_performance_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_coefficient_analysis_plots(df):
    """
    Create detailed coefficient analysis plots.
    """
    print(f"\nCREATING COEFFICIENT ANALYSIS PLOTS")
    print("="*60)
    
    # Create a figure with 3 subplots for detailed coefficient analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Coefficient Analysis Over Time', fontsize=16, fontweight='bold')
    
    # 1. Intercept over time with confidence bands
    ax1 = axes[0, 0]
    ax1.plot(df['plot_date'], df['intercept'], marker='o', linewidth=2, markersize=6, 
             color='green', label='Intercept (alpha)')
    ax1.fill_between(df['plot_date'], 
                     df['intercept'] - df['intercept'].std(),
                     df['intercept'] + df['intercept'].std(),
                     alpha=0.2, color='green', label='+/-1 Std Dev')
    ax1.set_xlabel('Test Period End Date')
    ax1.set_ylabel('Intercept Value')
    ax1.set_title('Intercept Stability Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. IEVR Coefficient over time
    ax2 = axes[0, 1]
    ax2.plot(df['plot_date'], df['ievr_coef'], marker='s', linewidth=2, markersize=6, 
             color='purple', label='IEVR Coefficient (beta1)')
    ax2.fill_between(df['plot_date'], 
                     df['ievr_coef'] - df['ievr_coef'].std(),
                     df['ievr_coef'] + df['ievr_coef'].std(),
                     alpha=0.2, color='purple', label='+/-1 Std Dev')
    ax2.set_xlabel('Test Period End Date')
    ax2.set_ylabel('IEVR Coefficient Value')
    ax2.set_title('IEVR Coefficient Stability Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Ratio Coefficient over time
    ax3 = axes[1, 0]
    ax3.plot(df['plot_date'], df['ratio_coef'], marker='^', linewidth=2, markersize=6, 
             color='red', label='Ratio Coefficient (beta2)')
    ax3.fill_between(df['plot_date'], 
                     df['ratio_coef'] - df['ratio_coef'].std(),
                     df['ratio_coef'] + df['ratio_coef'].std(),
                     alpha=0.2, color='red', label='+/-1 Std Dev')
    ax3.set_xlabel('Test Period End Date')
    ax3.set_ylabel('Ratio Coefficient Value')
    ax3.set_title('Ratio Coefficient Stability Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Coefficient correlation heatmap
    ax4 = axes[1, 1]
    coeff_data = df[['intercept', 'ievr_coef', 'ratio_coef']].corr()
    sns.heatmap(coeff_data, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax4)
    ax4.set_title('Coefficient Correlation Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/rolling_regression_coefficient_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Coefficient analysis plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_summary_statistics(df):
    """
    Create summary statistics and additional insights.
    """
    print(f"\nSUMMARY STATISTICS")
    print("="*60)
    
    # Overall performance
    print(f"OVERALL PERFORMANCE:")
    print(f"  Average Test R2: {df['test_r2'].mean():.4f}")
    print(f"  Average Test RMSE: {df['test_rmse'].mean():.4f}")
    print(f"  Best Test R2: {df['test_r2'].max():.4f} (Window {df['test_r2'].idxmax() + 1})")
    print(f"  Worst Test R2: {df['test_r2'].min():.4f} (Window {df['test_r2'].idxmin() + 1})")
    
    # Performance distribution
    print(f"\nPERFORMANCE DISTRIBUTION:")
    positive_r2 = (df['test_r2'] > 0).sum()
    negative_r2 = (df['test_r2'] <= 0).sum()
    print(f"  Windows with positive R2: {positive_r2}/{len(df)} ({positive_r2/len(df)*100:.1f}%)")
    print(f"  Windows with negative R2: {negative_r2}/{len(df)} ({negative_r2/len(df)*100:.1f}%)")
    
    # Coefficient stability
    print(f"\nCOEFFICIENT STABILITY:")
    print(f"  Intercept - Mean: {df['intercept'].mean():.4f}, Std: {df['intercept'].std():.4f}")
    print(f"  IEVR Coefficient - Mean: {df['ievr_coef'].mean():.4f}, Std: {df['ievr_coef'].std():.4f}")
    print(f"  Ratio Coefficient - Mean: {df['ratio_coef'].mean():.4f}, Std: {df['ratio_coef'].std():.4f}")
    
    # Sample size analysis
    print(f"\nSAMPLE SIZE ANALYSIS:")
    print(f"  Average training size: {df['train_obs'].mean():.0f}")
    print(f"  Average testing size: {df['test_obs'].mean():.0f}")
    print(f"  Total observations used: {df['train_obs'].sum() + df['test_obs'].sum():,}")
    
    # Time period analysis
    print(f"\nTIME PERIOD ANALYSIS:")
    df['year'] = df['test_end_date'].dt.year
    yearly_perf = df.groupby('year').agg({
        'test_r2': ['mean', 'std', 'count'],
        'test_rmse': ['mean', 'std'],
        'intercept': ['mean', 'std'],
        'ievr_coef': ['mean', 'std'],
        'ratio_coef': ['mean', 'std']
    }).round(4)
    
    print("Yearly Performance Summary:")
    print(yearly_perf)
    
    return yearly_perf

def create_additional_plots(df):
    """
    Create additional insightful plots.
    """
    print(f"\nCREATING ADDITIONAL PLOTS")
    print("="*60)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Additional Rolling Regression Insights', fontsize=16, fontweight='bold')
    
    # 1. R2 vs Sample Size scatter plot
    ax1.scatter(df['test_obs'], df['test_r2'], alpha=0.7, s=60, color='steelblue')
    ax1.set_xlabel('Test Sample Size')
    ax1.set_ylabel('Test R2')
    ax1.set_title('R2 vs Sample Size')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df['test_obs'], df['test_r2'], 1)
    p = np.poly1d(z)
    ax1.plot(df['test_obs'], p(df['test_obs']), "r--", alpha=0.8, 
             label=f'Trend: y={z[0]:.6f}x+{z[1]:.3f}')
    ax1.legend()
    
    # 2. Coefficient correlation plot (IEVR vs Ratio)
    ax2.scatter(df['ievr_coef'], df['ratio_coef'], alpha=0.7, s=60, color='purple')
    ax2.set_xlabel('IEVR Coefficient (beta1)')
    ax2.set_ylabel('Ratio Coefficient (beta2)')
    ax2.set_title('IEVR vs Ratio Coefficient Correlation')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df['ievr_coef'].corr(df['ratio_coef'])
    ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/rolling_regression_additional_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Additional plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def verify_data_consistency(df):
    """
    Verify that the data matches the terminal output.
    """
    print(f"\nVERIFYING DATA CONSISTENCY WITH TERMINAL OUTPUT")
    print("="*60)
    
    # Check if we have the expected columns
    print(f"Data columns: {list(df.columns)}")
    print(f"Total windows: {len(df)}")
    
    # Show the data grouped by train_end_year to match terminal output
    print(f"\nPERFORMANCE OVER TIME (Grouped by Training End Year):")
    print("                val_r2  test_r2  val_rmse  test_rmse")
    print("train_end_year                                      ")
    
    for _, row in df.iterrows():
        year = int(row['train_end_year'])
        val_r2 = row['val_r2']
        test_r2 = row['test_r2']
        val_rmse = row['val_rmse']
        test_rmse = row['test_rmse']
        
        print(f"{year:<13} {val_r2:>8.4f} {test_r2:>8.4f} {val_rmse:>9.4f} {test_rmse:>9.4f}")
    
    # Verify key statistics match
    print(f"\nKEY STATISTICS VERIFICATION:")
    print(f"  Test R2 range: {df['test_r2'].min():.4f} to {df['test_r2'].max():.4f}")
    print(f"  Test RMSE range: {df['test_rmse'].min():.4f} to {df['test_rmse'].max():.4f}")
    print(f"  Years covered: {df['train_end_year'].min():.0f} to {df['train_end_year'].max():.0f}")
    
    return True

def main():
    """
    Main function to create all plots and analysis.
    """
    print("PLOTTING ROLLING REGRESSION RESULTS")
    print("="*60)
    
    try:
        # 1. Load results
        df = load_results()
        
        # 2. Create main performance plots
        main_fig = create_performance_plots(df)
        
        # 3. Create coefficient analysis plots
        coeff_fig = create_coefficient_analysis_plots(df)
        
        # 4. Create summary statistics
        yearly_perf = create_summary_statistics(df)
        
        # 5. Create additional plots
        additional_fig = create_additional_plots(df)
        
        # 6. Save yearly performance
        yearly_file = 'data_files/yearly_performance_summary.csv'
        yearly_perf.to_csv(yearly_file)
        print(f"\nYearly performance summary saved to: {yearly_file}")
        
        print(f"\nAll plots and analysis completed successfully!")
        print(f"Check the data_files/ directory for all output files")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
