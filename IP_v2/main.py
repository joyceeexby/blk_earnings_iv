#!/usr/bin/env python3
"""
Main execution script for Earnings Implied Volatility Analysis
Expanded to 100+ stocks with year-by-year analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds
from automated_analysis import AutomatedEarningsAnalysis
from regression_analysis import FixedRegressionAnalysis

def get_large_cap_stocks():
    """
    Get a list of 50 large-cap stocks for analysis.
    Focus on S&P 500 constituents with good options liquidity.
    """
    # 50 large-cap stocks with good options liquidity
    stocks = [
        # Technology (15 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'TXN',
        
        # Financial (10 stocks)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'AXP', 'BLK', 'SCHW',
        
        # Healthcare (10 stocks)
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR', 'BMY', 'AMGN',
        
        # Consumer (10 stocks)
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT',
        
        # Industrial (5 stocks)
        'BA', 'CAT', 'GE', 'MMM', 'HON'
    ]
    
    return stocks  # Exactly 50 stocks

def run_expanded_analysis():
    """
    Run expanded analysis on 100+ stocks with year-by-year breakdown.
    """
    print("EXPANDED EARNINGS VOLATILITY ANALYSIS")
    print("="*80)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami",
                           password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Get stock list
        stocks = get_large_cap_stocks()
        print(f"✓ Selected {len(stocks)} stocks for analysis")
        
        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)
        
        # Analysis parameters - Extended to 2000 for more data per stock
        start_date = '2000-01-01'
        end_date = '2023-12-31'
        analysis_days_before = 30
        
        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Analysis window: {analysis_days_before} days before earnings")
        
        # Store all results
        all_results = []
        
        # Analyze each stock
        for i, ticker in enumerate(stocks):
            print(f"\n{'='*60}")
            print(f"ANALYZING STOCK {i+1}/{len(stocks)}: {ticker}")
            print(f"{'='*60}")
            
            try:
                # Analyze earnings events for this stock
                results_df = analyzer.analyze_multiple_events(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    analysis_days_before=analysis_days_before
                )
                
                if results_df is not None and not results_df.empty:
                    # Add ticker column
                    results_df['ticker'] = ticker
                    all_results.append(results_df)
                    print(f"✓ {ticker}: {len(results_df)} events analyzed")
                else:
                    print(f"✗ {ticker}: No valid events found")
                    
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")
                continue
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            print(f"\n{'='*80}")
            print(f"COMBINED ANALYSIS RESULTS")
            print(f"{'='*80}")
            print(f"Total events: {len(combined_results)}")
            print(f"Stocks with data: {combined_results['ticker'].nunique()}")
            print(f"Date range: {combined_results['earnings_date'].min()} to {combined_results['earnings_date'].max()}")
            
            # Save results
            combined_results.to_csv('expanded_earnings_analysis_results.csv', index=False)
            print(f"✓ Results saved to expanded_earnings_analysis_results.csv")
            
            # Run pooled regression analysis using the working approach
            print(f"\n{'='*80}")
            print(f"POOLED REGRESSION ANALYSIS")
            print(f"{'='*80}")
            
            try:
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
                    regression_analyzer.plot_regression_analysis()
                    
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
                            
                            if model1.pvalues['ievr'] < 0.05:
                                print(f"  Conclusion: IEVR coefficient is significant (p < 0.05)")
                            else:
                                print(f"  Conclusion: IEVR coefficient is not significant (p >= 0.05)")
                    
                    # Year-by-year analysis
                    print(f"\n{'='*80}")
                    print(f"YEAR-BY-YEAR ANALYSIS")
                    print(f"{'='*80}")
                    
                    combined_results['year'] = pd.to_datetime(combined_results['earnings_date']).dt.year
                    year_results = []
                    
                    for year in sorted(combined_results['year'].unique()):
                        year_data = combined_results[combined_results['year'] == year]
                        print(f"\nYear {year} (n={len(year_data)}):")
                        
                        # Save year data to temporary file
                        temp_file = f'_temp_year_{year}.csv'
                        year_data.to_csv(temp_file, index=False)
                        
                        try:
                            year_regression = FixedRegressionAnalysis(temp_file)
                            basic_models = year_regression.run_basic_regressions()
                            if basic_models and len(basic_models) > 0:
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
                                    print(f"  Error: No IEVR coefficient found")
                            else:
                                print(f"  Error: No valid models")
                        except Exception as e:
                            print(f"  Error: {e}")
                        finally:
                            # Clean up temporary file
                            import os
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                    
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
                    
                else:
                    print("✗ No valid regression models found")
                    
            except Exception as e:
                print(f"✗ Regression analysis failed: {e}")
            
            print(f"\n✓ Expanded analysis completed successfully!")
            
        else:
            print("✗ No results found for any stocks")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function to run expanded analysis.
    """
    run_expanded_analysis()

if __name__ == "__main__":
    main() 