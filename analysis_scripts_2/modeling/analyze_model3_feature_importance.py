#!/usr/bin/env python3
"""
Analyze Model 3 Feature Importance
Extract and analyze which features contribute most to Model 3's performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model3_results():
    """
    Load Model 3 results and extract feature coefficients.
    """
    print("📊 LOADING MODEL 3 RESULTS FOR FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load Model 3 results
    model3_results = pd.read_csv('data_files/model_comparison_model_3_results.csv')
    
    print(f"✅ Loaded Model 3 results: {len(model3_results)} windows")
    
    # Extract coefficient columns
    coef_columns = [col for col in model3_results.columns if col.startswith('coef_')]
    print(f"🔍 Found {len(coef_columns)} feature coefficients: {coef_columns}")
    
    # Show the features
    features = [col.replace('coef_', '') for col in coef_columns]
    print(f"📋 Features: {features}")
    
    return model3_results, coef_columns, features

def analyze_feature_importance(model3_results, coef_columns, features):
    """
    Analyze feature importance based on coefficients.
    """
    print(f"\n🔍 ANALYZING FEATURE IMPORTANCE")
    print("="*60)
    
    # Create feature importance DataFrame
    feature_importance = {}
    
    for i, feature in enumerate(features):
        coef_col = f'coef_{feature}'
        if coef_col in model3_results.columns:
            coefficients = model3_results[coef_col].dropna()
            
            feature_importance[feature] = {
                'mean_coef': coefficients.mean(),
                'std_coef': coefficients.std(),
                'abs_mean_coef': np.abs(coefficients).mean(),
                'min_coef': coefficients.min(),
                'max_coef': coefficients.max(),
                'positive_count': (coefficients > 0).sum(),
                'negative_count': (coefficients < 0).sum(),
                'total_count': len(coefficients),
                'stability': 1 - (coefficients.std() / np.abs(coefficients.mean())) if coefficients.mean() != 0 else 0
            }
    
    # Convert to DataFrame
    importance_df = pd.DataFrame(feature_importance).T
    importance_df = importance_df.round(4)
    
    print("📊 FEATURE IMPORTANCE SUMMARY:")
    print("-" * 50)
    print(importance_df)
    
    # Save feature importance
    importance_df.to_csv('data_files/model3_feature_importance.csv')
    print(f"\n💾 Feature importance saved to: data_files/model3_feature_importance.csv")
    
    return importance_df

def create_feature_importance_visualization(importance_df, features):
    """
    Create visualizations for feature importance.
    """
    print(f"\n📊 CREATING FEATURE IMPORTANCE VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model 3 Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Mean coefficient values with error bars
    ax1 = axes[0, 0]
    means = importance_df['mean_coef']
    stds = importance_df['std_coef']
    
    bars1 = ax1.bar(range(len(features)), means, yerr=stds, 
                     alpha=0.7, capsize=5, color='skyblue')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Mean Coefficient Value')
    ax1.set_title('Feature Coefficients (Mean ± Std)')
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars1, means)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Absolute mean coefficient values (magnitude of impact)
    ax2 = axes[0, 1]
    abs_means = importance_df['abs_mean_coef']
    
    bars2 = ax2.bar(range(len(features)), abs_means, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Absolute Mean Coefficient')
    ax2.set_title('Feature Impact Magnitude (Absolute Values)')
    ax2.set_xticks(range(len(features)))
    ax2.set_xticklabels(features, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, abs_mean) in enumerate(zip(bars2, abs_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{abs_mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Coefficient stability (lower std/mean ratio = more stable)
    ax3 = axes[1, 0]
    stability = importance_df['stability']
    
    bars3 = ax3.bar(range(len(features)), stability, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Stability Score')
    ax3.set_title('Feature Coefficient Stability (Higher = More Stable)')
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(features, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, stab) in enumerate(zip(bars3, stability)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{stab:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Feature ranking by importance
    ax4 = axes[1, 1]
    # Sort by absolute mean coefficient
    sorted_features = importance_df.sort_values('abs_mean_coef', ascending=True)
    
    bars4 = ax4.barh(range(len(sorted_features)), sorted_features['abs_mean_coef'], 
                      alpha=0.7, color='gold')
    ax4.set_yticks(range(len(sorted_features)))
    ax4.set_yticklabels(sorted_features.index)
    ax4.set_xlabel('Absolute Mean Coefficient')
    ax4.set_title('Feature Importance Ranking')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, abs_mean) in enumerate(zip(bars4, sorted_features['abs_mean_coef'])):
        ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{abs_mean:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/model3_feature_importance_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Feature importance visualization saved to: {output_file}")
    
    plt.show()
    
    return fig

def analyze_coefficient_trends(model3_results, coef_columns, features):
    """
    Analyze how feature coefficients change over time.
    """
    print(f"\n📈 ANALYZING COEFFICIENT TRENDS OVER TIME")
    print("="*60)
    
    # Add test year for time analysis
    model3_results['test_year'] = model3_results['test_start'].str[:4].astype(int)
    
    # Analyze coefficient trends by year
    yearly_coefficients = {}
    
    for feature in features:
        coef_col = f'coef_{feature}'
        if coef_col in model3_results.columns:
            yearly_means = model3_results.groupby('test_year')[coef_col].mean()
            yearly_coefficients[feature] = yearly_means
    
    # Create trend visualization
    plt.figure(figsize=(16, 10))
    
    for feature in features:
        if feature in yearly_coefficients:
            plt.plot(yearly_coefficients[feature].index, yearly_coefficients[feature].values, 
                    marker='o', label=feature, linewidth=2, markersize=6)
    
    plt.xlabel('Test Year')
    plt.ylabel('Mean Coefficient Value')
    plt.title('Model 3 Feature Coefficient Trends Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Highlight 2018
    plt.axvline(x=2018, color='red', linestyle='--', alpha=0.7, label='2018 (Regime Change)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'data_files/model3_coefficient_trends.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Coefficient trends visualization saved to: {output_file}")
    
    plt.show()
    
    return yearly_coefficients

def detailed_feature_analysis(importance_df, features):
    """
    Provide detailed analysis of each feature.
    """
    print(f"\n🔬 DETAILED FEATURE ANALYSIS")
    print("="*60)
    
    # Sort features by absolute importance
    sorted_importance = importance_df.sort_values('abs_mean_coef', ascending=False)
    
    print("🏆 FEATURE IMPORTANCE RANKING:")
    print("-" * 50)
    
    for i, (feature, row) in enumerate(sorted_importance.iterrows()):
        rank = i + 1
        print(f"\n{rank}. {feature.upper()}")
        print(f"   - Mean Coefficient: {row['mean_coef']:.4f}")
        print(f"   - Impact Magnitude: {row['abs_mean_coef']:.4f}")
        print(f"   - Stability: {row['stability']:.4f}")
        print(f"   - Direction: {'Positive' if row['mean_coef'] > 0 else 'Negative'}")
        print(f"   - Consistency: {row['positive_count']}/{row['total_count']} positive")
        
        # Interpret the feature
        interpret_feature(feature, row)
    
    return sorted_importance

def interpret_feature(feature, row):
    """
    Provide interpretation of what each feature means.
    """
    if feature == 'ievr':
        print(f"   - Interpretation: Implied Earnings Volatility Ratio")
        print(f"     Higher values predict higher realized earnings volatility")
    elif feature == 'normative_iv_rv_ratio':
        print(f"   - Interpretation: Normative IV/RV Ratio")
        print(f"     Measures relative implied vs realized volatility")
    elif feature == 'SKEW':
        print(f"   - Interpretation: Options Skew")
        print(f"     Measures asymmetry in options pricing")
    elif feature == 'KURT':
        print(f"   - Interpretation: Options Kurtosis")
        print(f"     Measures tail risk in options pricing")
    elif feature == 'IV_RATIO':
        print(f"   - Interpretation: Implied Volatility Ratio")
        print(f"     Relative implied volatility measure")
    elif feature == 'SMIRK':
        print(f"   - Interpretation: Volatility Smirk")
        print(f"     Measures volatility smile asymmetry")
    elif feature.startswith('vol_hl'):
        days = feature.split('_')[-1]
        print(f"   - Interpretation: {days}-day Historical Volatility")
        print(f"     Past volatility over {days} trading days")
    else:
        print(f"   - Interpretation: {feature} feature")

def main():
    """
    Main function to analyze Model 3 feature importance.
    """
    print("🔍 MODEL 3 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load Model 3 results
    model3_results, coef_columns, features = load_model3_results()
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model3_results, coef_columns, features)
    
    # Create visualizations
    create_feature_importance_visualization(importance_df, features)
    
    # Analyze coefficient trends
    yearly_coefficients = analyze_coefficient_trends(model3_results, coef_columns, features)
    
    # Detailed feature analysis
    sorted_importance = detailed_feature_analysis(importance_df, features)
    
    print(f"\n🎉 Model 3 feature importance analysis completed!")
    print(f"📊 Top feature: {sorted_importance.index[0]}")
    print(f"📊 Most stable feature: {importance_df.loc[importance_df['stability'].idxmax()].name}")

if __name__ == "__main__":
    main()

