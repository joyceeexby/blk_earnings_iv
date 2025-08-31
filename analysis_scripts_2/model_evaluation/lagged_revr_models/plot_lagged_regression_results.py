#!/usr/bin/env python3
"""
Plot Lagged Regression Results
Visualize and compare performance of original, REVR_lag1, and full lagged models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_all_results():
    """
    Load all three regression results for comparison.
    """
    print("LOADING ALL REGRESSION RESULTS")
    print("="*60)
    
    results = {}
    
    # Load original results
    try:
        original_file = 'data_files/rolling_regression_results.csv'
        original_df = pd.read_csv(original_file)
        original_df['model'] = 'Original (2 features)'
        results['original'] = original_df
        print(f"Loaded original model: {len(original_df)} windows")
    except FileNotFoundError:
        print("Original results file not found")
        
    # Load REVR_lag1 only results
    try:
        lag1_file = 'data_files/rolling_regression_results_lag1_only.csv'
        lag1_df = pd.read_csv(lag1_file)
        lag1_df['model'] = 'REVR_lag1 (3 features)'
        results['lag1'] = lag1_df
        print(f"Loaded REVR_lag1 model: {len(lag1_df)} windows")
    except FileNotFoundError:
        print("REVR_lag1 results file not found")
        
    # Load full lagged results
    try:
        full_lags_file = 'data_files/rolling_regression_results_with_lags.csv'
        full_lags_df = pd.read_csv(full_lags_file)
        full_lags_df['model'] = 'Full Lags (4 features)'
        results['full_lags'] = full_lags_df
        print(f"Loaded full lags model: {len(full_lags_df)} windows")
    except FileNotFoundError:
        print("Full lags results file not found")
    
    return results

def create_performance_comparison_plots(results):
    """
    Create comprehensive performance comparison plots.
    """
    print(f"\nCREATING PERFORMANCE COMPARISON PLOTS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Rolling Regression Performance Comparison: Original vs Lagged Features', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    plot_data = []
    colors = ['steelblue', 'orange', 'green']
    model_names = ['Original (2 features)', 'REVR_lag1 (3 features)', 'Full Lags (4 features)']
    
    # 1. Test R² comparison over time
    ax1 = axes[0, 0]
    for i, (key, df) in enumerate(results.items()):
        if df is not None and len(df) > 0:
            # Convert test_end to datetime for plotting
            df['test_end_date'] = pd.to_datetime(df['test_end'] + '-01')
            ax1.plot(df['test_end_date'], df['test_r2'], 
                    marker='o', linewidth=2, markersize=5, 
                    color=colors[i], label=model_names[i], alpha=0.8)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No predictive power (R²=0)')
    ax1.set_xlabel('Test Period End Date')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Test R² Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Performance distribution comparison
    ax2 = axes[0, 1]
    test_r2_data = []
    labels = []
    for key, df in results.items():
        if df is not None and len(df) > 0:
            test_r2_data.append(df['test_r2'].values)
            labels.append(model_names[list(results.keys()).index(key)])
    
    if test_r2_data:
        box_plot = ax2.boxplot(test_r2_data, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(test_r2_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Test R²')
    ax2.set_title('Test R² Distribution Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Coefficient evolution (for models with lagged features)
    ax3 = axes[1, 0]
    if 'lag1' in results and results['lag1'] is not None:
        lag1_df = results['lag1']
        lag1_df['test_end_date'] = pd.to_datetime(lag1_df['test_end'] + '-01')
        
        ax3.plot(lag1_df['test_end_date'], lag1_df['ievr_coef'], 
                marker='o', linewidth=2, markersize=5, color='purple', 
                label='IEVR Coefficient', alpha=0.8)
        ax3.plot(lag1_df['test_end_date'], lag1_df['ratio_coef'], 
                marker='s', linewidth=2, markersize=5, color='red', 
                label='Ratio Coefficient', alpha=0.8)
        ax3.plot(lag1_df['test_end_date'], lag1_df['revr_lag1_coef'], 
                marker='^', linewidth=2, markersize=5, color='green', 
                label='REVR_lag1 Coefficient', alpha=0.8)
        
        # Add mean lines
        ax3.axhline(y=lag1_df['ievr_coef'].mean(), color='purple', 
                   linestyle='--', alpha=0.5, 
                   label=f'Mean IEVR: {lag1_df["ievr_coef"].mean():.3f}')
        ax3.axhline(y=lag1_df['ratio_coef'].mean(), color='red', 
                   linestyle='--', alpha=0.5,
                   label=f'Mean Ratio: {lag1_df["ratio_coef"].mean():.3f}')
        ax3.axhline(y=lag1_df['revr_lag1_coef'].mean(), color='green', 
                   linestyle='--', alpha=0.5,
                   label=f'Mean REVR_lag1: {lag1_df["revr_lag1_coef"].mean():.3f}')
    
    ax3.set_xlabel('Test Period End Date')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Model Coefficients Over Time (REVR_lag1 Model)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Performance metrics summary
    ax4 = axes[1, 1]
    
    # Calculate summary statistics
    summary_data = []
    for key, df in results.items():
        if df is not None and len(df) > 0:
            summary_data.append({
                'Model': model_names[list(results.keys()).index(key)],
                'Avg_R2': df['test_r2'].mean(),
                'Avg_RMSE': df['test_rmse'].mean(),
                'Best_R2': df['test_r2'].max(),
                'Worst_R2': df['test_r2'].min(),
                'Std_R2': df['test_r2'].std()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create grouped bar chart
        x = np.arange(len(summary_df))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, summary_df['Avg_R2'], width, 
                       label='Average R²', alpha=0.8, color='steelblue')
        bars2 = ax4.bar(x + width/2, summary_df['Avg_RMSE'], width, 
                       label='Average RMSE', alpha=0.8, color='orange')
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Performance Metric')
        ax4.set_title('Average Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/lagged_regression_performance_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_coefficient_analysis_plots(results):
    """
    Create detailed coefficient analysis plots for lagged models.
    """
    print(f"\nCREATING COEFFICIENT ANALYSIS PLOTS")
    print("="*60)
    
    # Focus on lagged models
    lagged_results = {k: v for k, v in results.items() if k in ['lag1', 'full_lags'] and v is not None}
    
    if not lagged_results:
        print("No lagged model results found for coefficient analysis")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Coefficient Analysis for Lagged Models', fontsize=16, fontweight='bold')
    
    # 1. Coefficient stability over time (REVR_lag1 model)
    ax1 = axes[0, 0]
    if 'lag1' in lagged_results:
        lag1_df = lagged_results['lag1']
        lag1_df['test_end_date'] = pd.to_datetime(lag1_df['test_end'] + '-01')
        
        # Plot REVR_lag1 coefficient with confidence bands
        mean_coef = lag1_df['revr_lag1_coef'].mean()
        std_coef = lag1_df['revr_lag1_coef'].std()
        
        ax1.plot(lag1_df['test_end_date'], lag1_df['revr_lag1_coef'], 
                marker='o', linewidth=2, markersize=6, color='green', 
                label='REVR_lag1 Coefficient')
        ax1.fill_between(lag1_df['test_end_date'], 
                        mean_coef - std_coef, mean_coef + std_coef,
                        alpha=0.2, color='green', label='±1 Std Dev')
        ax1.axhline(y=mean_coef, color='green', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_coef:.4f}')
        
        ax1.set_xlabel('Test Period End Date')
        ax1.set_ylabel('REVR_lag1 Coefficient')
        ax1.set_title('REVR_lag1 Coefficient Stability Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Coefficient comparison between models
    ax2 = axes[0, 1]
    coef_comparison_data = []
    
    for key, df in lagged_results.items():
        model_name = 'REVR_lag1' if key == 'lag1' else 'Full Lags'
        coef_comparison_data.extend([
            {'Model': model_name, 'Coefficient': 'IEVR', 'Value': df['ievr_coef'].mean()},
            {'Model': model_name, 'Coefficient': 'Ratio', 'Value': df['ratio_coef'].mean()},
            {'Model': model_name, 'Coefficient': 'REVR_lag1', 'Value': df['revr_lag1_coef'].mean()}
        ])
        
        if 'revr_lag2_coef' in df.columns:
            coef_comparison_data.append({
                'Model': model_name, 'Coefficient': 'REVR_lag2', 'Value': df['revr_lag2_coef'].mean()
            })
    
    if coef_comparison_data:
        coef_df = pd.DataFrame(coef_comparison_data)
        
        # Create grouped bar chart
        coef_pivot = coef_df.pivot(index='Coefficient', columns='Model', values='Value')
        coef_pivot.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_xlabel('Coefficient Type')
        ax2.set_ylabel('Mean Coefficient Value')
        ax2.set_title('Mean Coefficient Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Coefficient correlation heatmap (for full lags model)
    ax3 = axes[1, 0]
    if 'full_lags' in lagged_results:
        full_lags_df = lagged_results['full_lags']
        coeff_cols = ['ievr_coef', 'ratio_coef', 'revr_lag1_coef', 'revr_lag2_coef']
        available_cols = [col for col in coeff_cols if col in full_lags_df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = full_lags_df[available_cols].corr()
            
            # Create custom labels
            labels = []
            for col in available_cols:
                if 'ievr' in col:
                    labels.append('IEVR')
                elif 'ratio' in col:
                    labels.append('Ratio')
                elif 'lag1' in col:
                    labels.append('REVR_lag1')
                elif 'lag2' in col:
                    labels.append('REVR_lag2')
                else:
                    labels.append(col)
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax3,
                       xticklabels=labels, yticklabels=labels)
            ax3.set_title('Coefficient Correlation Matrix (Full Lags Model)')
    
    # 4. Performance vs coefficient magnitude
    ax4 = axes[1, 1]
    if 'lag1' in lagged_results:
        lag1_df = lagged_results['lag1']
        
        # Scatter plot of REVR_lag1 coefficient vs test R²
        ax4.scatter(lag1_df['revr_lag1_coef'], lag1_df['test_r2'], 
                   alpha=0.7, s=60, color='green')
        
        # Add trend line
        z = np.polyfit(lag1_df['revr_lag1_coef'], lag1_df['test_r2'], 1)
        p = np.poly1d(z)
        ax4.plot(lag1_df['revr_lag1_coef'], p(lag1_df['revr_lag1_coef']), 
                "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.3f}')
        
        # Calculate correlation
        corr = lag1_df['revr_lag1_coef'].corr(lag1_df['test_r2'])
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        ax4.set_xlabel('REVR_lag1 Coefficient')
        ax4.set_ylabel('Test R²')
        ax4.set_title('REVR_lag1 Coefficient vs Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data_files/lagged_regression_coefficient_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Coefficient analysis plots saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_summary_statistics_table(results):
    """
    Create comprehensive summary statistics table.
    """
    print(f"\nCREATING SUMMARY STATISTICS TABLE")
    print("="*60)
    
    summary_stats = []
    
    for key, df in results.items():
        if df is not None and len(df) > 0:
            model_name = df['model'].iloc[0]
            
            stats = {
                'Model': model_name,
                'Windows': len(df),
                'Avg_Test_R2': df['test_r2'].mean(),
                'Std_Test_R2': df['test_r2'].std(),
                'Best_Test_R2': df['test_r2'].max(),
                'Worst_Test_R2': df['test_r2'].min(),
                'Avg_Test_RMSE': df['test_rmse'].mean(),
                'Std_Test_RMSE': df['test_rmse'].std(),
                'Positive_R2_Count': (df['test_r2'] > 0).sum(),
                'Positive_R2_Pct': (df['test_r2'] > 0).mean() * 100
            }
            
            # Add coefficient statistics for lagged models
            if 'revr_lag1_coef' in df.columns:
                stats['Avg_REVR_lag1_Coef'] = df['revr_lag1_coef'].mean()
                stats['Std_REVR_lag1_Coef'] = df['revr_lag1_coef'].std()
                
            if 'revr_lag2_coef' in df.columns:
                stats['Avg_REVR_lag2_Coef'] = df['revr_lag2_coef'].mean()
                stats['Std_REVR_lag2_Coef'] = df['revr_lag2_coef'].std()
            
            summary_stats.append(stats)
    
    # Convert to DataFrame for nice formatting
    summary_df = pd.DataFrame(summary_stats)
    
    # Round numerical columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    print("\nCOMPREHENSIVE SUMMARY STATISTICS:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_file = 'data_files/lagged_regression_summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary statistics saved to: {summary_file}")
    
    return summary_df

def create_yearly_performance_analysis(results):
    """
    Create yearly performance analysis for all models.
    """
    print(f"\nCREATING YEARLY PERFORMANCE ANALYSIS")
    print("="*60)
    
    yearly_data = []
    
    for key, df in results.items():
        if df is not None and len(df) > 0:
            model_name = df['model'].iloc[0]
            
            # Add year column if not present
            if 'train_end_year' not in df.columns:
                df['train_end_year'] = pd.to_datetime(df['train_end']).dt.year
            
            # Group by year
            yearly_stats = df.groupby('train_end_year').agg({
                'test_r2': ['mean', 'std', 'count'],
                'test_rmse': ['mean', 'std']
            }).round(4)
            
            # Flatten column names
            yearly_stats.columns = [f'{col[1]}_{col[0]}' if col[1] else col[0] 
                                   for col in yearly_stats.columns]
            yearly_stats = yearly_stats.reset_index()
            yearly_stats['Model'] = model_name
            
            yearly_data.append(yearly_stats)
    
    if yearly_data:
        # Combine all yearly data
        combined_yearly = pd.concat(yearly_data, ignore_index=True)
        
        print("\nYEARLY PERFORMANCE ANALYSIS:")
        print("=" * 80)
        for model in combined_yearly['Model'].unique():
            print(f"\n{model}:")
            model_data = combined_yearly[combined_yearly['Model'] == model]
            print(model_data[['train_end_year', 'mean_test_r2', 'std_test_r2', 
                             'count_test_r2', 'mean_test_rmse']].to_string(index=False))
        
        # Save to CSV
        yearly_file = 'data_files/lagged_regression_yearly_performance.csv'
        combined_yearly.to_csv(yearly_file, index=False)
        print(f"\nYearly performance analysis saved to: {yearly_file}")
        
        return combined_yearly
    
    return None

def main():
    """
    Main function to create all plots and analysis for lagged regression results.
    """
    print("LAGGED REGRESSION RESULTS ANALYSIS")
    print("="*60)
    
    try:
        # 1. Load all results
        results = load_all_results()
        
        if not results:
            print("No results found to analyze")
            return
        
        # 2. Create performance comparison plots
        comparison_fig = create_performance_comparison_plots(results)
        
        # 3. Create coefficient analysis plots
        coefficient_fig = create_coefficient_analysis_plots(results)
        
        # 4. Create summary statistics table
        summary_df = create_summary_statistics_table(results)
        
        # 5. Create yearly performance analysis
        yearly_df = create_yearly_performance_analysis(results)
        
        print(f"\nAll lagged regression analysis completed successfully!")
        print(f"Check the data_files/ directory for all output files:")
        print(f"  - lagged_regression_performance_comparison.png")
        print(f"  - lagged_regression_coefficient_analysis.png")
        print(f"  - lagged_regression_summary_statistics.csv")
        print(f"  - lagged_regression_yearly_performance.csv")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


