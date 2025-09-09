"""
Analysis script for examining the saved regression results from the expanded earnings analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_results():
    """Load all saved results files."""
    try:
        # Load pooled regression results
        try:
            pooled_summary = pd.read_csv('data_files/pooled_regression_summary.csv')
            print(f"✓ Loaded {len(pooled_summary)} pooled regression models")
        except FileNotFoundError:
            print("⚠ Pooled regression summary not found")
            pooled_summary = None
        
        try:
            pooled_details = pd.read_csv('data_files/pooled_regression_details.csv')
            print(f"✓ Loaded {len(pooled_details)} detailed regression results")
        except FileNotFoundError:
            print("⚠ Pooled regression details not found")
            pooled_details = None
        
        # Year-by-year analysis removed - insufficient data per year
        print("⚠ Year-by-year analysis has been disabled due to insufficient data points per year")
        year_results = pd.DataFrame()  # Empty DataFrame
        
        # Load raw data
        raw_data = pd.read_csv('data_files/expanded_earnings_analysis_results.csv')
        print(f"✓ Loaded {len(raw_data)} raw data points")
        
        return pooled_summary, pooled_details, year_results, raw_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the main analysis first to generate the results files.")
        return None, None, None, None

# Year-by-year analysis functions removed due to insufficient data per year

def analyze_pooled_results(pooled_summary, pooled_details):
    """Analyze pooled regression results."""
    print("\n" + "="*80)
    print("POOLED REGRESSION ANALYSIS")
    print("="*80)
    
    if pooled_summary is None or pooled_details is None:
        print("No pooled regression results to analyze.")
        return
    
    print("Pooled Regression Model Results:")
    print("="*50)
    
    for _, row in pooled_summary.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  IEVR Coefficient: {row['IEVR Coefficient']}")
        print(f"  IEVR t-stat: {row['IEVR t-stat']}")
        print(f"  IEVR p-value: {row['IEVR p-value']}")
        print(f"  R-squared: {row['R-squared']}")
        print(f"  Adjusted R-squared: {row['Adj R-squared']}")
        print(f"  Observations: {row['N']}")
    
    # Analyze basic model (Model 1)
    basic_model = pooled_details[pooled_details['model_number'] == 1]
    if len(basic_model) > 0:
        model = basic_model.iloc[0]
        print(f"\n{'='*50}")
        print(f"BASIC MODEL ANALYSIS (REVR = α + β × IEVR)")
        print(f"{'='*50}")
        print(f"IEVR coefficient: {model['ievr_coef']:.4f}")
        print(f"T-statistic: {model['ievr_tstat']:.3f}")
        print(f"P-value: {model['ievr_pvalue']:.4f}")
        print(f"R-squared: {model['r_squared']:.4f}")
        print(f"Adjusted R-squared: {model['adj_r_squared']:.4f}")
        print(f"Observations: {model['nobs']}")
        
        # Significance interpretation
        if model['ievr_pvalue'] < 0.01:
            significance = "highly significant (p < 0.01)"
        elif model['ievr_pvalue'] < 0.05:
            significance = "significant (p < 0.05)"
        elif model['ievr_pvalue'] < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant"
        
        print(f"Conclusion: IEVR coefficient is {significance}")
        
        # Economic significance
        if abs(model['ievr_coef']) > 0.5:
            economic_significance = "large"
        elif abs(model['ievr_coef']) > 0.2:
            economic_significance = "moderate"
        else:
            economic_significance = "small"
        
        print(f"Economic significance: {economic_significance} effect size")
    
    # Compare models
    print(f"\n{'='*50}")
    print(f"MODEL COMPARISON")
    print(f"{'='*50}")
    
    comparison_data = []
    for _, model in pooled_details.iterrows():
        comparison_data.append({
            'Model': model['model_type'],
            'R²': model['r_squared'],
            'Adj R²': model['adj_r_squared'],
            'IEVR Coef': model.get('ievr_coef', 'N/A'),
            'IEVR P-value': model.get('ievr_pvalue', 'N/A'),
            'Observations': model['nobs']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    return pooled_details

# Temporal pattern analysis removed due to insufficient year-by-year data

def create_summary_plots(pooled_details):
    """Create summary plots of the pooled regression results."""
    print("\n" + "="*80)
    print("CREATING SUMMARY PLOTS")
    print("="*80)
    
    if pooled_details is None or len(pooled_details) == 0:
        print("No pooled results to plot.")
        return
    
    # Filter models with IEVR coefficients
    models_with_ievr = pooled_details[pooled_details['ievr_coef'].notna()]
    
    if len(models_with_ievr) == 0:
        print("No models with IEVR coefficients to plot.")
        return
    
    plt.figure(figsize=(15, 10))
    
    # R-squared by model
    plt.subplot(2, 3, 1)
    models_with_ievr.plot(x='model_type', y='r_squared', kind='bar', ax=plt.gca())
    plt.title('R-squared by Model')
    plt.xlabel('Model')
    plt.ylabel('R-squared')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # IEVR coefficients by model
    plt.subplot(2, 3, 2)
    models_with_ievr.plot(x='model_type', y='ievr_coef', kind='bar', ax=plt.gca())
    plt.title('IEVR Coefficients by Model')
    plt.xlabel('Model')
    plt.ylabel('IEVR Coefficient')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # P-values by model
    plt.subplot(2, 3, 3)
    models_with_ievr.plot(x='model_type', y='ievr_pvalue', kind='bar', ax=plt.gca())
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% significance')
    plt.title('IEVR P-values by Model')
    plt.xlabel('Model')
    plt.ylabel('P-value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # T-statistics by model
    plt.subplot(2, 3, 4)
    models_with_ievr.plot(x='model_type', y='ievr_tstat', kind='bar', ax=plt.gca())
    plt.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='5% significance')
    plt.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
    plt.title('IEVR T-statistics by Model')
    plt.xlabel('Model')
    plt.ylabel('T-statistic')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R-squared vs IEVR coefficient
    plt.subplot(2, 3, 5)
    plt.scatter(models_with_ievr['ievr_coef'], models_with_ievr['r_squared'], alpha=0.7)
    for _, row in models_with_ievr.iterrows():
        plt.annotate(row['model_type'], (row['ievr_coef'], row['r_squared']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('IEVR Coefficient')
    plt.ylabel('R-squared')
    plt.title('R-squared vs IEVR Coefficient')
    plt.grid(True, alpha=0.3)
    
    # Model comparison
    plt.subplot(2, 3, 6)
    x = range(len(models_with_ievr))
    plt.bar(x, models_with_ievr['r_squared'], alpha=0.7, label='R-squared')
    plt.bar(x, models_with_ievr['adj_r_squared'], alpha=0.5, label='Adj R-squared')
    plt.xticks(x, models_with_ievr['model_type'], rotation=45)
    plt.title('Model Fit Comparison')
    plt.xlabel('Model')
    plt.ylabel('R-squared')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_files/pooled_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_detailed_results(pooled_details):
    """Export detailed results for further analysis."""
    print("\n" + "="*80)
    print("EXPORTING DETAILED RESULTS")
    print("="*80)
    
    if pooled_details is None:
        print("No pooled results to export.")
        return
    
    # Export all model details
    pooled_details.to_csv('data_files/all_pooled_models.csv', index=False)
    print(f"✓ Exported all pooled model details to data_files/all_pooled_models.csv")
    
    # Export models with significant IEVR coefficients
    models_with_ievr = pooled_details[pooled_details['ievr_coef'].notna()]
    significant_models = models_with_ievr[models_with_ievr['ievr_pvalue'] < 0.05]
    if len(significant_models) > 0:
        significant_models.to_csv('data_files/significant_pooled_models.csv', index=False)
        print(f"✓ Exported {len(significant_models)} significant models to data_files/significant_pooled_models.csv")
    
    # Export summary statistics
    numeric_cols = ['r_squared', 'adj_r_squared', 'nobs', 'f_stat', 'f_pvalue', 'aic', 'bic']
    numeric_cols = [col for col in numeric_cols if col in pooled_details.columns]
    if numeric_cols:
        summary_stats = pooled_details[numeric_cols].describe()
        summary_stats.to_csv('data_files/pooled_summary_statistics.csv')
        print(f"✓ Exported summary statistics to data_files/pooled_summary_statistics.csv")
    
    # Create correlation matrix for models with IEVR
    models_with_ievr = pooled_details[pooled_details['ievr_coef'].notna()]
    if len(models_with_ievr) > 1:
        numeric_cols = ['ievr_coef', 'ievr_tstat', 'ievr_pvalue', 'r_squared', 'adj_r_squared']
        numeric_cols = [col for col in numeric_cols if col in models_with_ievr.columns]
        if len(numeric_cols) > 1:
            corr_matrix = models_with_ievr[numeric_cols].corr()
            corr_matrix.to_csv('data_files/pooled_correlation_matrix.csv')
            print(f"✓ Exported correlation matrix to data_files/pooled_correlation_matrix.csv")

def main():
    """Main analysis function."""
    print("EARNINGS VOLATILITY ANALYSIS - RESULTS EXAMINATION")
    print("="*80)
    
    # Load results
    pooled_summary, pooled_details, year_results, raw_data = load_results()
    
    if pooled_summary is None and pooled_details is None:
        return
    
    # Analyze pooled results
    analyze_pooled_results(pooled_summary, pooled_details)
    
    # Temporal pattern analysis removed due to insufficient year-by-year data
    
    # Create summary plots
    if pooled_details is not None:
        create_summary_plots(pooled_details)
        export_detailed_results(pooled_details)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated files:")
    print("  - output_files/pooled_regression_analysis.png")
    print("  - data_files/all_pooled_models.csv")
    print("  - data_files/significant_pooled_models.csv")
    print("  - data_files/pooled_summary_statistics.csv")
    print("  - data_files/pooled_correlation_matrix.csv")

if __name__ == "__main__":
    main() 