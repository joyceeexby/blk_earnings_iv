#!/usr/bin/env python3
"""
Momentum Features Analysis for Model 3 Enhancement
Analyze correlation and statistical relationships between momentum features and REVR
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the final merged dataset with momentum features and prepare it for analysis.
    """
    print("LOADING AND PREPARING FINAL MERGED DATASET WITH MOMENTUM FEATURES")
    print("="*70)
    
    # Load the final merged dataset with momentum features
    file_path = 'data_files/final_merged_dataset_with_momentum_final.csv'
    df = pd.read_csv(file_path)
    print("Loaded dataset: {} observations".format(len(df)))
    
    # Convert dates
    df['earnings_date'] = pd.to_datetime(df['earnings_date'])
    
    # Calculate normative_iv_rv_ratio feature
    print("Creating normative_iv_rv_ratio feature...")
    df['normative_iv_rv_ratio'] = df['avg_pre'] / df['normative_realized_vol']
    
    # Handle infinite values and NaN
    df['normative_iv_rv_ratio'] = df['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Check momentum features availability
    momentum_features = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'rolling_momentum_3m', 'z_score_momentum']
    print("\nMomentum features availability:")
    for feature in momentum_features:
        if feature in df.columns:
            valid_count = df[feature].notna().sum()
            total_count = len(df)
            coverage = 100.0 * valid_count / total_count
            print("  {}: {} ({:.1f}% coverage)".format(feature, valid_count, coverage))
        else:
            print("  {}: Not found in dataset".format(feature))
    
    # Sort by date and remove extreme outliers
    df = df.sort_values('earnings_date').reset_index(drop=True)
    
    # Remove extreme outliers (z-score > 3)
    revr_zscore = np.abs((df['revr'] - df['revr'].mean()) / df['revr'].std())
    ievr_zscore = np.abs((df['ievr'] - df['ievr'].mean()) / df['ievr'].std())
    
    df_clean = df[(revr_zscore <= 3) & (ievr_zscore <= 3)].copy()
    print("After outlier removal: {} observations".format(len(df_clean)))
    
    return df_clean

def analyze_momentum_correlations(df):
    """
    Analyze correlations between momentum features and REVR.
    """
    print("\nANALYZING MOMENTUM FEATURE CORRELATIONS WITH REVR")
    print("="*60)
    
    # Define all features
    momentum_features = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'rolling_momentum_3m', 'z_score_momentum']
    base_features = ['ievr', 'normative_iv_rv_ratio', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'vol_hl7', 'vol_hl10', 'vol_hl21']
    
    # Available features
    available_momentum = [f for f in momentum_features if f in df.columns and df[f].notna().sum() > 500]
    available_base = [f for f in base_features if f in df.columns]
    
    print("Available momentum features: {}".format(available_momentum))
    print("Available base features: {}".format(available_base))
    
    # Calculate correlations with REVR
    correlations = {}
    
    print("\nCorrelations with REVR:")
    print("-" * 40)
    
    # Base features correlations
    print("Base Model 3 features:")
    for feature in available_base:
        if df[feature].notna().sum() > 100:
            corr = df[['revr', feature]].corr().iloc[0, 1]
            correlations[feature] = corr
            print("  {}: {:.4f}".format(feature, corr))
    
    print("\nMomentum features:")
    for feature in available_momentum:
        corr = df[['revr', feature]].corr().iloc[0, 1]
        correlations[feature] = corr
        print("  {}: {:.4f}".format(feature, corr))
    
    return correlations, available_momentum, available_base

def analyze_feature_combinations_simple(df, available_momentum, available_base):
    """
    Simple analysis of feature combinations using correlation and basic statistics.
    """
    print("\nANALYZING FEATURE COMBINATIONS")
    print("="*60)
    
    # Create different feature combinations
    combinations = {
        'Original_Model3': available_base,
        'Model3_plus_momentum_1m': available_base + ['momentum_1m'] if 'momentum_1m' in available_momentum else available_base,
        'Model3_plus_momentum_3m': available_base + ['momentum_3m'] if 'momentum_3m' in available_momentum else available_base,
        'Model3_plus_z_score_momentum': available_base + ['z_score_momentum'] if 'z_score_momentum' in available_momentum else available_base,
        'Model3_plus_best_momentum': available_base + ['momentum_1m', 'z_score_momentum'] if all(f in available_momentum for f in ['momentum_1m', 'z_score_momentum']) else available_base,
        'Model3_plus_all_momentum': available_base + available_momentum
    }
    
    results = []
    
    for combo_name, features in combinations.items():
        print("\nAnalyzing: {}".format(combo_name))
        print("Features: {}".format(features))
        
        # Filter data where all features are available
        feature_data = df[features + ['revr']].dropna()
        
        if len(feature_data) < 100:
            print("  Insufficient data: {} observations".format(len(feature_data)))
            continue
        
        # Calculate basic statistics
        n_obs = len(feature_data)
        revr_mean = feature_data['revr'].mean()
        revr_std = feature_data['revr'].std()
        
        # Calculate correlations with REVR for each feature
        feature_corrs = []
        for feature in features:
            if feature in feature_data.columns:
                corr = feature_data[['revr', feature]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    feature_corrs.append(abs(corr))
        
        # Calculate average absolute correlation
        avg_abs_corr = np.mean(feature_corrs) if feature_corrs else 0
        
        # Simple performance proxy: weighted correlation score
        performance_score = avg_abs_corr * np.sqrt(n_obs / 1000.0)  # Adjust for sample size
        
        results.append({
            'combination': combo_name,
            'n_features': len(features),
            'n_observations': n_obs,
            'revr_mean': revr_mean,
            'revr_std': revr_std,
            'avg_abs_correlation': avg_abs_corr,
            'performance_score': performance_score,
            'features': features
        })
        
        print("  Observations: {}".format(n_obs))
        print("  REVR mean: {:.4f}".format(revr_mean))
        print("  REVR std: {:.4f}".format(revr_std))
        print("  Avg abs correlation: {:.4f}".format(avg_abs_corr))
        print("  Performance score: {:.4f}".format(performance_score))
    
    return results

def analyze_temporal_stability(df, available_momentum, available_base):
    """
    Analyze how momentum features perform across different time periods.
    """
    print("\nANALYZING TEMPORAL STABILITY")
    print("="*60)
    
    # Create year-based splits
    df['year'] = df['earnings_date'].dt.year
    years = sorted(df['year'].unique())
    
    print("Available years: {} to {}".format(min(years), max(years)))
    
    # Split into periods
    mid_year = years[len(years)//2]
    early_period = df[df['year'] <= mid_year]
    late_period = df[df['year'] > mid_year]
    
    print("Early period: {} to {} ({} observations)".format(
        min(early_period['year']), max(early_period['year']), len(early_period)
    ))
    print("Late period: {} to {} ({} observations)".format(
        min(late_period['year']), max(late_period['year']), len(late_period)
    ))
    
    # Analyze correlations in each period
    print("\nCorrelation stability analysis:")
    print("-" * 40)
    
    all_features = available_base + available_momentum
    stability_results = []
    
    for feature in all_features:
        if feature not in df.columns:
            continue
            
        # Calculate correlations in each period
        early_data = early_period[['revr', feature]].dropna()
        late_data = late_period[['revr', feature]].dropna()
        
        if len(early_data) > 50 and len(late_data) > 50:
            early_corr = early_data.corr().iloc[0, 1]
            late_corr = late_data.corr().iloc[0, 1]
            
            if not (np.isnan(early_corr) or np.isnan(late_corr)):
                corr_diff = abs(early_corr - late_corr)
                avg_corr = (abs(early_corr) + abs(late_corr)) / 2
                
                stability_results.append({
                    'feature': feature,
                    'early_corr': early_corr,
                    'late_corr': late_corr,
                    'avg_abs_corr': avg_corr,
                    'correlation_diff': corr_diff,
                    'is_momentum': feature in available_momentum
                })
                
                print("  {}: Early={:.4f}, Late={:.4f}, Diff={:.4f}".format(
                    feature, early_corr, late_corr, corr_diff
                ))
    
    return stability_results

def create_summary_report(correlations, combination_results, stability_results):
    """
    Create a comprehensive summary report.
    """
    print("\nCOMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    # Sort combination results by performance score
    sorted_combinations = sorted(combination_results, key=lambda x: x['performance_score'], reverse=True)
    
    print("FEATURE COMBINATION RANKINGS:")
    print("-" * 40)
    for i, result in enumerate(sorted_combinations, 1):
        print("{}. {} (Score: {:.4f})".format(i, result['combination'], result['performance_score']))
        print("   Features: {}".format(len(result['features'])))
        print("   Avg correlation: {:.4f}".format(result['avg_abs_correlation']))
        print("   Observations: {}".format(result['n_observations']))
        print()
    
    # Find best momentum features
    momentum_correlations = {k: v for k, v in correlations.items() 
                           if k in ['momentum_1m', 'momentum_3m', 'momentum_6m', 'rolling_momentum_3m', 'z_score_momentum']}
    
    if momentum_correlations:
        best_momentum = max(momentum_correlations.items(), key=lambda x: abs(x[1]))
        print("BEST INDIVIDUAL MOMENTUM FEATURE:")
        print("-" * 40)
        print("{}: {:.4f} correlation with REVR".format(best_momentum[0], best_momentum[1]))
        print()
    
    # Stability analysis
    if stability_results:
        stable_features = sorted(stability_results, key=lambda x: x['correlation_diff'])[:5]
        print("MOST STABLE FEATURES:")
        print("-" * 40)
        for feature_info in stable_features:
            print("{}: Avg corr={:.4f}, Stability={:.4f}".format(
                feature_info['feature'], 
                feature_info['avg_abs_corr'],
                1.0 - feature_info['correlation_diff']  # Higher = more stable
            ))
        print()
    
    # Best performing combination
    best_combo = sorted_combinations[0]
    baseline_combo = next((c for c in sorted_combinations if c['combination'] == 'Original_Model3'), None)
    
    print("RECOMMENDATION:")
    print("-" * 40)
    print("Best combination: {}".format(best_combo['combination']))
    print("Features: {}".format(best_combo['features']))
    
    if baseline_combo:
        improvement = ((best_combo['performance_score'] - baseline_combo['performance_score']) / 
                      baseline_combo['performance_score']) * 100
        print("Improvement over baseline: {:.1f}%".format(improvement))
    
    # Save results
    df_results = pd.DataFrame(combination_results)
    output_file = 'output_files/momentum_features_analysis_results.csv'
    df_results.to_csv(output_file, index=False)
    print("\nResults saved to: {}".format(output_file))

def main():
    """
    Main function to run momentum features analysis.
    """
    print("MOMENTUM FEATURES ANALYSIS FOR MODEL 3 ENHANCEMENT")
    print("="*70)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Analyze correlations
    correlations, available_momentum, available_base = analyze_momentum_correlations(df)
    
    # Analyze feature combinations
    combination_results = analyze_feature_combinations_simple(df, available_momentum, available_base)
    
    # Analyze temporal stability
    stability_results = analyze_temporal_stability(df, available_momentum, available_base)
    
    # Create summary report
    create_summary_report(correlations, combination_results, stability_results)
    
    print("\nMomentum features analysis completed successfully!")

if __name__ == "__main__":
    main()

