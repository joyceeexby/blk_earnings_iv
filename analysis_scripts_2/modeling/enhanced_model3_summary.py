#!/usr/bin/env python3
"""
Enhanced Model 3 Summary
Provide comprehensive analysis of Linear Regression vs Random Forest vs XGBoost results
"""

import pandas as pd
import numpy as np

def load_and_analyze_results():
    """
    Load and analyze all enhanced Model 3 results.
    """
    print("📊 ENHANCED MODEL 3 COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Load results for each algorithm
    algorithms = {
        'Linear_Regression': pd.read_csv('data_files/enhanced_model3_linear_regression_results.csv'),
        'Random_Forest': pd.read_csv('data_files/enhanced_model3_random_forest_results.csv'),
        'XGBoost': pd.read_csv('data_files/enhanced_model3_xgboost_results.csv')
    }
    
    # Load feature importance for each algorithm
    feature_importance = {
        'Linear_Regression': pd.read_csv('data_files/enhanced_model3_linear_regression_feature_importance.csv'),
        'Random_Forest': pd.read_csv('data_files/enhanced_model3_random_forest_feature_importance.csv'),
        'XGBoost': pd.read_csv('data_files/enhanced_model3_xgboost_feature_importance.csv')
    }
    
    return algorithms, feature_importance

def performance_comparison(algorithms):
    """
    Compare performance across all algorithms.
    """
    print("🏆 PERFORMANCE COMPARISON")
    print("="*60)
    
    performance_summary = {}
    
    for algo_name, results in algorithms.items():
        if len(results) > 0:
            performance_summary[algo_name] = {
                'Windows': len(results),
                'Avg_Val_R2': results['val_r2'].mean(),
                'Avg_Test_R2': results['val_r2'].mean(),
                'Best_Test_R2': results['test_r2'].max(),
                'Worst_Test_R2': results['test_r2'].min(),
                'Val_R2_Std': results['val_r2'].std(),
                'Test_R2_Std': results['test_r2'].std(),
                'Avg_Val_RMSE': results['val_rmse'].mean(),
                'Avg_Test_RMSE': results['test_rmse'].mean()
            }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(performance_summary).T
    comparison_df = comparison_df.round(4)
    
    print("📊 ALGORITHM PERFORMANCE SUMMARY:")
    print("-" * 50)
    print(comparison_df)
    
    # Find best performing algorithm
    best_algo = comparison_df.loc[comparison_df['Avg_Test_R2'].idxmax()]
    best_algo_name = comparison_df.loc[comparison_df['Avg_Test_R2'].idxmax()].name
    
    print(f"\n🏆 BEST PERFORMING ALGORITHM: {best_algo_name}")
    print(f"  - Average Test R²: {best_algo['Avg_Test_R2']:.4f}")
    print(f"  - Best Test R²: {best_algo['Best_Test_R2']:.4f}")
    print(f"  - Stability (Test R² Std): {best_algo['Test_R2_Std']:.4f}")
    
    return comparison_df, best_algo_name

def feature_importance_analysis(feature_importance, algorithms):
    """
    Analyze feature importance across all algorithms.
    """
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Features to analyze
    features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
    
    feature_summary = {}
    
    for feature in features:
        feature_summary[feature] = {}
        
        for algo_name, imp_df in feature_importance.items():
            if len(imp_df) > 0:
                imp_col = f'importance_{feature}'
                if imp_col in imp_df.columns:
                    feature_summary[feature][algo_name] = {
                        'Mean_Importance': imp_df[imp_col].mean(),
                        'Std_Importance': imp_df[imp_col].std(),
                        'Min_Importance': imp_df[imp_col].min(),
                        'Max_Importance': imp_df[imp_col].max()
                    }
    
    # Create feature importance comparison
    print("📊 FEATURE IMPORTANCE COMPARISON:")
    print("-" * 50)
    
    for feature in features:
        print(f"\n🔬 {feature.upper()}:")
        for algo_name in algorithms.keys():
            if algo_name in feature_summary[feature]:
                imp_data = feature_summary[feature][algo_name]
                print(f"  {algo_name}: {imp_data['Mean_Importance']:.4f} ± {imp_data['Std_Importance']:.4f}")
    
    return feature_summary

def algorithm_stability_analysis(algorithms):
    """
    Analyze algorithm stability across time windows.
    """
    print(f"\n📈 ALGORITHM STABILITY ANALYSIS")
    print("="*60)
    
    stability_summary = {}
    
    for algo_name, results in algorithms.items():
        if len(results) > 0:
            # Calculate stability metrics
            val_r2_std = results['val_r2'].std()
            test_r2_std = results['test_r2'].std()
            
            # Calculate performance consistency (lower std = more stable)
            stability_score = 1 - (test_r2_std / abs(results['test_r2'].mean())) if results['test_r2'].mean() != 0 else 0
            
            stability_summary[algo_name] = {
                'Val_R2_Std': val_r2_std,
                'Test_R2_Std': test_r2_std,
                'Stability_Score': stability_score,
                'Performance_Consistency': 'High' if test_r2_std < 0.1 else 'Medium' if test_r2_std < 0.15 else 'Low'
            }
    
    # Create stability DataFrame
    stability_df = pd.DataFrame(stability_summary).T
    stability_df = stability_df.round(4)
    
    print("📊 ALGORITHM STABILITY SUMMARY:")
    print("-" * 50)
    print(stability_df)
    
    # Find most stable algorithm
    most_stable = stability_df.loc[stability_df['Stability_Score'].idxmax()]
    most_stable_name = stability_df.loc[stability_df['Stability_Score'].idxmax()].name
    
    print(f"\n🛡️  MOST STABLE ALGORITHM: {most_stable_name}")
    print(f"  - Stability Score: {most_stable['Stability_Score']:.4f}")
    print(f"  - Test R² Standard Deviation: {most_stable['Test_R2_Std']:.4f}")
    print(f"  - Performance Consistency: {most_stable['Performance_Consistency']}")
    
    return stability_df, most_stable_name

def time_period_analysis(algorithms):
    """
    Analyze performance across different time periods.
    """
    print(f"\n🕒 TIME PERIOD PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Add test year for analysis
    for algo_name, results in algorithms.items():
        if len(results) > 0:
            results['test_year'] = results['test_start'].str[:4].astype(int)
    
    # Analyze performance by year
    yearly_performance = {}
    
    for algo_name, results in algorithms.items():
        if len(results) > 0:
            yearly_means = results.groupby('test_year')['test_r2'].mean()
            yearly_performance[algo_name] = yearly_means
    
    print("📊 YEARLY PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    # Get all years
    all_years = set()
    for algo_name, yearly_data in yearly_performance.items():
        all_years.update(yearly_data.index)
    
    all_years = sorted(list(all_years))
    
    for year in all_years:
        print(f"\n{year}:")
        for algo_name in algorithms.keys():
            if algo_name in yearly_performance and year in yearly_performance[algo_name]:
                r2 = yearly_performance[algo_name][year]
                print(f"  {algo_name}: {r2:.4f}")
    
    return yearly_performance

def create_comprehensive_summary(comparison_df, feature_summary, stability_df, yearly_performance):
    """
    Create a comprehensive summary of all findings.
    """
    print(f"\n🎯 COMPREHENSIVE SUMMARY")
    print("="*60)
    
    print("🏆 PERFORMANCE RANKING (by Average Test R²):")
    print("-" * 50)
    
    # Sort by performance
    performance_ranking = comparison_df.sort_values('Avg_Test_R2', ascending=False)
    
    for i, (algo_name, row) in enumerate(performance_ranking.iterrows()):
        rank = i + 1
        print(f"{rank}. {algo_name}")
        print(f"   - Average Test R²: {row['Avg_Test_R2']:.4f}")
        print(f"   - Best Test R²: {row['Best_Test_R2']:.4f}")
        print(f"   - Stability: {stability_df.loc[algo_name, 'Performance_Consistency']}")
    
    print(f"\n🛡️  STABILITY RANKING (by Stability Score):")
    print("-" * 50)
    
    # Sort by stability
    stability_ranking = stability_df.sort_values('Stability_Score', ascending=False)
    
    for i, (algo_name, row) in enumerate(stability_ranking.iterrows()):
        rank = i + 1
        print(f"{rank}. {algo_name}")
        print(f"   - Stability Score: {row['Stability_Score']:.4f}")
        print(f"   - Test R² Std: {row['Test_R2_Std']:.4f}")
        print(f"   - Consistency: {row['Performance_Consistency']}")
    
    print(f"\n🔬 KEY INSIGHTS:")
    print("-" * 50)
    
    best_performer = comparison_df.loc[comparison_df['Avg_Test_R2'].idxmax()]
    most_stable = stability_df.loc[stability_df['Stability_Score'].idxmax()]
    
    print(f"1. **Best Performance**: {best_performer.name} (Test R²: {best_performer['Avg_Test_R2']:.4f})")
    print(f"2. **Most Stable**: {most_stable.name} (Stability: {most_stable['Stability_Score']:.4f})")
    print(f"3. **Performance Range**: {comparison_df['Avg_Test_R2'].max():.4f} to {comparison_df['Avg_Test_R2'].min():.4f}")
    print(f"4. **Stability Range**: {stability_df['Stability_Score'].max():.4f} to {stability_df['Stability_Score'].min():.4f}")
    
    # Check if best performer is also most stable
    if best_performer.name == most_stable.name:
        print(f"5. **Optimal Choice**: {best_performer.name} - Best performance AND most stable!")
    else:
        print(f"5. **Trade-off Required**: Choose between performance ({best_performer.name}) and stability ({most_stable.name})")
    
    return performance_ranking, stability_ranking

def main():
    """
    Main function to run the comprehensive analysis.
    """
    print("🔬 ENHANCED MODEL 3 COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load and analyze results
    algorithms, feature_importance = load_and_analyze_results()
    
    # Performance comparison
    comparison_df, best_algo = performance_comparison(algorithms)
    
    # Feature importance analysis
    feature_summary = feature_importance_analysis(feature_importance, algorithms)
    
    # Stability analysis
    stability_df, most_stable = algorithm_stability_analysis(algorithms)
    
    # Time period analysis
    yearly_performance = time_period_analysis(algorithms)
    
    # Create comprehensive summary
    performance_ranking, stability_ranking = create_comprehensive_summary(
        comparison_df, feature_summary, stability_df, yearly_performance
    )
    
    print(f"\n🎉 Enhanced Model 3 comprehensive analysis completed!")
    print(f"📊 Best performing algorithm: {best_algo}")
    print(f"📊 Most stable algorithm: {most_stable}")

if __name__ == "__main__":
    main()

