#!/usr/bin/env python3
"""
Model Performance Comparison Analysis
Compare the performance of different model approaches:
1. IEVR + normative_iv_rv_ratio (original)
2. IEVR + normative_iv_rv_ratio + 1 lagged REVR
3. IEVR + normative_iv_rv_ratio + 2 lagged REVR
4. IEVR + normative_iv_rv_ratio + optimal volatility features (new)
5. IEVR + normative_iv_rv_ratio + options features (new)
6. IEVR + normative_iv_rv_ratio + options + optimal volatility (new)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_results():
    """
    Load all model results and calculate performance metrics.
    """
    print("📊 LOADING AND ANALYZING MODEL RESULTS")
    print("="*60)
    
    # Load original model results (IEVR + normative_iv_rv_ratio)
    original_results = pd.read_csv('output_files/rolling_regression_results.csv')
    
    # Load lag1 model results (IEVR + normative_iv_rv_ratio + 1 lagged REVR)
    lag1_results = pd.read_csv('output_files/rolling_regression_results_lag1_only.csv')
    
    # Load lag2 model results (IEVR + normative_iv_rv_ratio + 2 lagged REVR)
    lag2_results = pd.read_csv('output_files/rolling_regression_results_with_lags.csv')
    
    # Load new model results
    model1_results = pd.read_csv('data_files/model_comparison_model_1_results.csv')
    model2_results = pd.read_csv('data_files/model_comparison_model_2_results.csv')
    model3_results = pd.read_csv('data_files/model_comparison_model_3_results.csv')
    
    # Calculate performance metrics for each model
    models = {
        'Original (IEVR + ratio)': {
            'val_r2_mean': original_results['val_r2'].mean(),
            'val_r2_std': original_results['val_r2'].std(),
            'test_r2_mean': original_results['test_r2'].mean(),
            'test_r2_std': original_results['test_r2'].std(),
            'val_rmse_mean': original_results['val_rmse'].mean(),
            'test_rmse_mean': original_results['test_rmse'].mean(),
            'windows': len(original_results)
        },
        'Lag1 (IEVR + ratio + REVR_lag1)': {
            'val_r2_mean': lag1_results['val_r2'].mean(),
            'val_r2_std': lag1_results['val_r2'].std(),
            'test_r2_mean': lag1_results['test_r2'].mean(),
            'test_r2_std': lag1_results['test_r2'].std(),
            'val_rmse_mean': lag1_results['val_rmse'].mean(),
            'test_rmse_mean': lag1_results['test_rmse'].mean(),
            'windows': len(lag1_results)
        },
        'Lag2 (IEVR + ratio + REVR_lag1 + REVR_lag2)': {
            'val_r2_mean': lag2_results['val_r2'].mean(),
            'val_r2_std': lag2_results['val_r2'].std(),
            'test_r2_mean': lag2_results['test_r2'].mean(),
            'test_r2_std': lag2_results['test_r2'].std(),
            'val_rmse_mean': lag2_results['val_rmse'].mean(),
            'test_rmse_mean': lag2_results['test_rmse'].mean(),
            'windows': len(lag2_results)
        },
        'Model 1 (IEVR + ratio + optimal vol)': {
            'val_r2_mean': model1_results['val_r2'].mean(),
            'val_r2_std': model1_results['val_r2'].std(),
            'test_r2_mean': model1_results['test_r2'].mean(),
            'test_r2_std': model1_results['test_r2'].std(),
            'val_rmse_mean': model1_results['val_rmse'].mean(),
            'test_rmse_mean': model1_results['test_rmse'].mean(),
            'windows': len(model1_results)
        },
        'Model 2 (IEVR + ratio + options)': {
            'val_r2_mean': model2_results['val_r2'].mean(),
            'val_r2_std': model2_results['val_r2'].std(),
            'test_r2_mean': model2_results['test_r2'].mean(),
            'test_r2_std': model2_results['test_r2'].std(),
            'val_rmse_mean': model2_results['val_rmse'].mean(),
            'test_rmse_mean': model2_results['test_rmse'].mean(),
            'windows': len(model2_results)
        },
        'Model 3 (IEVR + ratio + options + optimal vol)': {
            'val_r2_mean': model3_results['val_r2'].mean(),
            'val_r2_std': model3_results['val_r2'].std(),
            'test_r2_mean': model3_results['test_r2'].mean(),
            'test_r2_std': model3_results['test_r2'].std(),
            'val_rmse_mean': model3_results['val_rmse'].mean(),
            'test_rmse_mean': model3_results['test_rmse'].mean(),
            'windows': len(model3_results)
        }
    }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models).T
    comparison_df = comparison_df.round(4)
    
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(comparison_df)
    
    # Save comparison results
    comparison_df.to_csv('data_files/model_performance_comparison_summary.csv')
    print(f"\n💾 Comparison summary saved to: data_files/model_performance_comparison_summary.csv")
    
    return comparison_df, models

def create_performance_visualization(comparison_df):
    """
    Create visualizations comparing all models.
    """
    print(f"\n📊 CREATING PERFORMANCE COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison: All Approaches', fontsize=16, fontweight='bold')
    
    # 1. Test R² comparison
    ax1 = axes[0, 0]
    models = comparison_df.index
    test_r2_means = comparison_df['test_r2_mean']
    test_r2_stds = comparison_df['test_r2_std']
    
    bars1 = ax1.bar(range(len(models)), test_r2_means, yerr=test_r2_stds, 
                     alpha=0.7, capsize=5)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test R² Score')
    ax1.set_title('Test R² Performance Comparison')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([name.replace(' (', '\n(') for name in models], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars1, test_r2_means)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Validation vs Test R²
    ax2 = axes[0, 1]
    val_r2_means = comparison_df['val_r2_mean']
    
    x = np.arange(len(models))
    width = 0.35
    
    bars2_val = ax2.bar(x - width/2, val_r2_means, width, label='Validation R²', alpha=0.7)
    bars2_test = ax2.bar(x + width/2, test_r2_means, width, label='Test R²', alpha=0.7)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Validation vs Test R² Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.replace(' (', '\n(') for name in models], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RMSE comparison
    ax3 = axes[1, 0]
    val_rmse_means = comparison_df['val_rmse_mean']
    test_rmse_means = comparison_df['test_rmse_mean']
    
    bars3_val = ax3.bar(x - width/2, val_rmse_means, width, label='Validation RMSE', alpha=0.7)
    bars3_test = ax3.bar(x + width/2, test_rmse_means, width, label='Test RMSE', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name.replace(' (', '\n(') for name in models], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance ranking
    ax4 = axes[1, 1]
    # Sort by test R² performance
    sorted_models = comparison_df.sort_values('test_r2_mean', ascending=True)
    
    bars4 = ax4.barh(range(len(sorted_models)), sorted_models['test_r2_mean'], alpha=0.7)
    ax4.set_yticks(range(len(sorted_models)))
    ax4.set_yticklabels([name.replace(' (', '\n(') for name in sorted_models.index])
    ax4.set_xlabel('Test R² Score')
    ax4.set_title('Model Performance Ranking (by Test R²)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars4, sorted_models['test_r2_mean'])):
        ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{mean:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/model_performance_comparison_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def detailed_analysis(comparison_df):
    """
    Provide detailed analysis of the results.
    """
    print(f"\n🔍 DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Find best performing model
    best_model = comparison_df.loc[comparison_df['test_r2_mean'].idxmax()]
    best_model_name = comparison_df.loc[comparison_df['test_r2_mean'].idxmax()].name
    
    print(f"🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"  - Test R²: {best_model['test_r2_mean']:.4f}")
    print(f"  - Validation R²: {best_model['val_r2_mean']:.4f}")
    print(f"  - Test RMSE: {best_model['test_rmse_mean']:.4f}")
    
    # Performance improvement analysis
    baseline = comparison_df.loc['Original (IEVR + ratio)']
    print(f"\n📈 PERFORMANCE IMPROVEMENTS vs BASELINE:")
    print(f"Baseline (Original): Test R² = {baseline['test_r2_mean']:.4f}")
    
    for model_name, row in comparison_df.iterrows():
        if model_name != 'Original (IEVR + ratio)':
            improvement = row['test_r2_mean'] - baseline['test_r2_mean']
            improvement_pct = (improvement / baseline['test_r2_mean']) * 100
            print(f"  {model_name}: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    # Model stability analysis (lower std = more stable)
    most_stable = comparison_df.loc[comparison_df['test_r2_std'].idxmin()]
    most_stable_name = comparison_df.loc[comparison_df['test_r2_std'].idxmin()].name
    
    print(f"\n📊 MODEL STABILITY ANALYSIS:")
    print(f"Most Stable Model: {most_stable_name}")
    print(f"  - Test R² Std: {most_stable['test_r2_std']:.4f}")
    print(f"  - Validation R² Std: {most_stable['val_r2_std']:.4f}")
    
    # Feature effectiveness analysis
    print(f"\n🔬 FEATURE EFFECTIVENESS ANALYSIS:")
    
    # Compare lagged vs non-lagged models
    lagged_models = ['Lag1 (IEVR + ratio + REVR_lag1)', 'Lag2 (IEVR + ratio + REVR_lag1 + REVR_lag2)']
    non_lagged_models = ['Model 1 (IEVR + ratio + optimal vol)', 'Model 2 (IEVR + ratio + options)', 'Model 3 (IEVR + ratio + options + optimal vol)']
    
    lagged_avg = comparison_df.loc[lagged_models, 'test_r2_mean'].mean()
    non_lagged_avg = comparison_df.loc[non_lagged_models, 'test_r2_mean'].mean()
    
    print(f"  Lagged REVR Models Average: {lagged_avg:.4f}")
    print(f"  Non-Lagged Models Average: {non_lagged_avg:.4f}")
    print(f"  Difference: {non_lagged_avg - lagged_avg:+.4f}")
    
    if non_lagged_avg > lagged_avg:
        print(f"  ✅ Non-lagged models perform better on average")
    else:
        print(f"  ❌ Lagged models perform better on average")
    
    return best_model_name, most_stable_name

def main():
    """
    Main function to run the model performance comparison.
    """
    print("🔬 MODEL PERFORMANCE COMPARISON ANALYSIS")
    print("="*60)
    
    # Load and analyze results
    comparison_df, models = load_and_analyze_results()
    
    # Create visualizations
    create_performance_visualization(comparison_df)
    
    # Detailed analysis
    best_model, most_stable = detailed_analysis(comparison_df)
    
    print(f"\n🎉 Model performance comparison completed!")
    print(f"📊 Best performing model: {best_model}")
    print(f"📊 Most stable model: {most_stable}")

if __name__ == "__main__":
    main()

