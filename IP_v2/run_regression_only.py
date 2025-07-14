#!/usr/bin/env python3
"""
Run regression analysis only on existing data
"""

import pandas as pd
import numpy as np
from regression_analysis import FixedRegressionAnalysis

def run_regression_analysis():
    """
    Run regression analysis on existing data.
    """
    print("REGRESSION ANALYSIS ON EXISTING DATA")
    print("="*80)
    
    try:
        # Run regression analysis on existing data
        regression_analyzer = FixedRegressionAnalysis('expanded_earnings_analysis_results.csv')
        
        # Descriptive statistics
        regression_analyzer.descriptive_statistics()
        regression_analyzer.plot_descriptive_analysis()
        
        # Run all regression models
        basic_models = regression_analyzer.run_basic_regressions()
        extended_models = regression_analyzer.run_extended_regressions()
        
        # Save regression results
        print(f"\n{'='*80}")
        print(f"SAVING REGRESSION RESULTS")
        print(f"{'='*80}")
        
        # Create summary of all models
        all_models = []
        if basic_models:
            all_models.extend(basic_models)
        if extended_models:
            all_models.extend(extended_models)
        
        # Filter out None models
        valid_models = [model for model in all_models if model is not None]
        
        if valid_models:
            regression_summary = regression_analyzer.create_regression_summary(valid_models)
            regression_summary.to_csv('pooled_regression_summary.csv', index=False)
            print(f"✓ Pooled regression summary saved to pooled_regression_summary.csv")
            
            # Save detailed results for each model
            model_details = []
            for i, model in enumerate(valid_models, 1):
                model_info = {
                    'model_number': i,
                    'model_type': f'Model {i}',
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'nobs': model.nobs,
                    'f_stat': model.fvalue,
                    'f_pvalue': model.f_pvalue,
                    'aic': model.aic,
                    'bic': model.bic
                }
                
                # Add IEVR coefficient info if available
                if 'ievr' in model.params.index:
                    model_info.update({
                        'ievr_coef': model.params['ievr'],
                        'ievr_tstat': model.tvalues['ievr'],
                        'ievr_pvalue': model.pvalues['ievr'],
                        'ievr_std_error': model.bse['ievr']
                    })
                
                model_details.append(model_info)
            
            model_details_df = pd.DataFrame(model_details)
            model_details_df.to_csv('pooled_regression_details.csv', index=False)
            print(f"✓ Detailed regression results saved to pooled_regression_details.csv")
            
            # Diagnostic tests for main model
            if basic_models and len(basic_models) > 0 and basic_models[0] is not None:
                regression_analyzer.diagnostic_tests(basic_models[0])
            
            # Create plots
            regression_analyzer.plot_regression_results(valid_models)
        else:
            print("✗ No valid regression models were successfully estimated")
        
        # Print key findings
        print(f"\n{'='*80}")
        print(f"KEY FINDINGS FROM POOLED ANALYSIS")
        print(f"{'='*80}")
        
        if basic_models and len(basic_models) > 0 and basic_models[0] is not None:
            model1 = basic_models[0]  # Basic model
            if 'ievr' in model1.params.index:
                print(f"Basic Model (REVR = α + β × IEVR):")
                print(f"  IEVR coefficient: {model1.params['ievr']:.4f}")
                print(f"  T-statistic: {model1.tvalues['ievr']:.3f}")
                print(f"  P-value: {model1.pvalues['ievr']:.4f}")
                print(f"  R-squared: {model1.rsquared:.4f}")
                print(f"  Adjusted R-squared: {model1.rsquared_adj:.4f}")
                print(f"  Observations: {model1.nobs}")
                
                # Significance interpretation
                if model1.pvalues['ievr'] < 0.01:
                    significance = "highly significant (p < 0.01)"
                elif model1.pvalues['ievr'] < 0.05:
                    significance = "significant (p < 0.05)"
                elif model1.pvalues['ievr'] < 0.10:
                    significance = "marginally significant (p < 0.10)"
                else:
                    significance = "not significant"
                
                print(f"  Conclusion: IEVR coefficient is {significance}")
        else:
            print("✗ No valid basic regression model available for key findings")
        
        # Year-by-year analysis
        print(f"\n{'='*80}")
        print(f"YEAR-BY-YEAR ANALYSIS")
        print(f"{'='*80}")
        
        combined_results = pd.read_csv('expanded_earnings_analysis_results.csv')
        combined_results['year'] = pd.to_datetime(combined_results['earnings_date']).dt.year
        year_results = []
        
        for year in sorted(combined_results['year'].unique()):
            year_data = combined_results[combined_results['year'] == year]
            print(f"\nYear {year} (n={len(year_data)}):")
            
            # Save year data temporarily
            year_data.to_csv(f'_temp_year_{year}.csv', index=False)
            
            try:
                year_regression = FixedRegressionAnalysis(f'_temp_year_{year}.csv')
                basic_models = year_regression.run_basic_regressions()
                if basic_models and len(basic_models) > 0 and basic_models[0] is not None:
                    model = basic_models[0]  # First model (basic REVR on IEVR)
                    if 'ievr' in model.params.index:
                        year_result = {
                            'year': year,
                            'n_events': len(year_data),
                            'ievr_coef': model.params['ievr'],
                            'ievr_tstat': model.tvalues['ievr'],
                            'ievr_pvalue': model.pvalues['ievr'],
                            'r_squared': model.rsquared,
                            'adj_r_squared': model.rsquared_adj
                        }
                        year_results.append(year_result)
                        print(f"  IEVR coefficient: {model.params['ievr']:.3f}")
                        print(f"  T-stat: {model.tvalues['ievr']:.3f}")
                        print(f"  P-value: {model.pvalues['ievr']:.3f}")
                        print(f"  R-squared: {model.rsquared:.3f}")
                else:
                    print(f"  No valid regression model for year {year}")
            except Exception as e:
                print(f"  Error: {e}")
            
            # Clean up temp file
            import os
            if os.path.exists(f'_temp_year_{year}.csv'):
                os.remove(f'_temp_year_{year}.csv')
        
        # Save year-by-year results
        if year_results:
            year_df = pd.DataFrame(year_results)
            year_df.to_csv('year_by_year_regression_results.csv', index=False)
            print(f"\n✓ Year-by-year results saved to year_by_year_regression_results.csv")
        
        print(f"\n✓ Regression analysis completed successfully!")
        print(f"✓ Files generated:")
        print(f"  - pooled_regression_summary.csv (model summaries)")
        print(f"  - pooled_regression_details.csv (detailed results)")
        print(f"  - year_by_year_regression_results.csv (temporal analysis)")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function to run regression analysis.
    """
    run_regression_analysis()

if __name__ == "__main__":
    main() 