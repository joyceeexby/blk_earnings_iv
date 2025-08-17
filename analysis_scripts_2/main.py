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
from scipy import stats


def get_large_cap_stocks():
    """
    Get a comprehensive list of large-cap stocks for analysis across multiple sectors.
    """
    stocks = [
        # Technology (20 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'ADI', 'KLAC',
        
        # Financial Services (15 stocks)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CME', 'ICE',
        
        # Healthcare (15 stocks)
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'CVS', 'CI', 'ANTM', 'HUM',
        
        # Consumer Discretionary (15 stocks)
        'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR', 'YUM', 'CMG',
        'TGT', 'COST', 'WMT', 'PM', 'MO',
        
        # Consumer Staples (10 stocks)
        'PG', 'KO', 'PEP', 'EL', 'CL', 'GIS', 'KMB', 'HSY', 'SJM', 'CPB',
        
        # Energy (10 stocks)
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
        
        # Industrials (10 stocks)
        'BA', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
        
        # Materials (5 stocks)
        'LIN', 'APD', 'FCX', 'NEM', 'AA',
        
        # Communication Services (5 stocks)
        'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
        
        # Real Estate (5 stocks)
        'AMT', 'PLD', 'CCI', 'EQIX', 'DLR'
    ]
    return stocks

def calculate_rolling_betas(db, ticker, earnings_date, analysis_days_before=30, beta_window=60):
    """
    Calculate rolling betas for market, SMB, and HML factors.
    
    Parameters:
    -----------
    db : wrds.Connection
        WRDS database connection
    ticker : str
        Stock ticker symbol
    earnings_date : datetime
        Earnings announcement date
    analysis_days_before : int
        Days before earnings to end beta calculation (default: 30)
    beta_window : int
        Rolling window for beta calculation (default: 60 days)
    
    Returns:
    --------
    dict : Dictionary with beta values for market, SMB, and HML
    """
    try:
        # Calculate date range for beta calculation
        end_date = earnings_date - timedelta(days=analysis_days_before)
        start_date = end_date - timedelta(days=beta_window)
        
        # Get stock returns from CRSP
        stock_query = f"""
        SELECT date, ret
        FROM crsp.dsf
        WHERE permno IN (
            SELECT permno 
            FROM crsp.dsenames 
            WHERE ticker = '{ticker}'
            AND date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        )
        AND date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY date
        """
        
        stock_returns = db.raw_sql(stock_query)
        
        if stock_returns.empty:
            return {'beta_market': np.nan, 'beta_smb': np.nan, 'beta_hml': np.nan}
        
        # Get factor returns from French data library
        factor_query = f"""
        SELECT date, mktrf, smb, hml
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY date
        """
        
        factor_returns = db.raw_sql(factor_query)
        
        if factor_returns.empty:
            return {'beta_market': np.nan, 'beta_smb': np.nan, 'beta_hml': np.nan}
        
        # Merge stock and factor returns
        merged_data = pd.merge(stock_returns, factor_returns, on='date', how='inner')
        
        if len(merged_data) < 30:  # Need at least 30 observations for reliable beta
            return {'beta_market': np.nan, 'beta_smb': np.nan, 'beta_hml': np.nan}
        
        # Calculate betas using OLS regression
        betas = {}
        
        # Market beta (excess returns)
        if 'mktrf' in merged_data.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_data['mktrf'], merged_data['ret']
            )
            betas['beta_market'] = slope
        
        # SMB beta
        if 'smb' in merged_data.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_data['smb'], merged_data['ret']
            )
            betas['beta_smb'] = slope
        
        # HML beta
        if 'hml' in merged_data.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_data['hml'], merged_data['ret']
            )
            betas['beta_hml'] = slope
        
        return betas
        
    except Exception as e:
        print(f"Error calculating betas for {ticker}: {e}")
        return {'beta_market': np.nan, 'beta_smb': np.nan, 'beta_hml': np.nan}

def add_beta_features(db, results_df):
    """
    Add rolling beta features to the results dataframe.
    
    Parameters:
    -----------
    db : wrds.Connection
        WRDS database connection
    results_df : pd.DataFrame
        DataFrame with earnings analysis results
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added beta features
    """
    print("Adding rolling beta features...")
    
    # Initialize beta columns
    results_df['beta_market'] = np.nan
    results_df['beta_smb'] = np.nan
    results_df['beta_hml'] = np.nan
    
    total_events = len(results_df)
    successful_betas = 0
    
    for idx, row in results_df.iterrows():
        ticker = row['ticker']
        earnings_date = pd.to_datetime(row['earnings_date'])
        
        # Calculate betas for this event
        betas = calculate_rolling_betas(db, ticker, earnings_date)
        
        # Store beta values
        results_df.loc[idx, 'beta_market'] = betas.get('beta_market', np.nan)
        results_df.loc[idx, 'beta_smb'] = betas.get('beta_smb', np.nan)
        results_df.loc[idx, 'beta_hml'] = betas.get('beta_hml', np.nan)
        
        if not np.isnan(betas.get('beta_market', np.nan)):
            successful_betas += 1
        
        # Progress update every 50 events
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total_events} events ({successful_betas} successful betas)")
    
    print(f"✓ Beta calculation complete: {successful_betas}/{total_events} events have valid betas")
    
    # Print beta statistics
    print(f"Beta Statistics:")
    print(f"  Market Beta - Mean: {results_df['beta_market'].mean():.3f}, Std: {results_df['beta_market'].std():.3f}")
    print(f"  SMB Beta - Mean: {results_df['beta_smb'].mean():.3f}, Std: {results_df['beta_smb'].std():.3f}")
    print(f"  HML Beta - Mean: {results_df['beta_hml'].mean():.3f}, Std: {results_df['beta_hml'].std():.3f}")
    
    return results_df

def run_expanded_analysis():
    """
    Run expanded analysis on 100+ stocks with year-by-year breakdown.
    """
    print("EXPANDED EARNINGS VOLATILITY ANALYSIS")
    print("="*80)

    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="",
                             password="")
        print("✓ Connected to WRDS")

        # Get stock list
        stocks = get_large_cap_stocks()
        print(f"✓ Selected {len(stocks)} stocks for analysis")

        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)

        # Analysis parameters - Extended to 2005 for more data per stock
        start_date = '2005-01-01'
        end_date = '2024-12-31'
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

            # Add rolling beta features
            print(f"\n{'='*80}")
            print(f"ADDING ROLLING BETA FEATURES")
            print(f"{'='*80}")
            
            try:
                combined_results = add_beta_features(db, combined_results)
                print(f"✓ Beta features added successfully")
            except Exception as e:
                print(f"✗ Error adding beta features: {e}")
                print("Continuing with analysis without beta features...")
            
            # Save results with beta features
            combined_results.to_csv('data_files/expanded_earnings_analysis_results.csv', index=False)
            print(f"✓ Results saved to data_files/expanded_earnings_analysis_results.csv")

            # Run pooled regression analysis using the working approach
            print(f"\n{'='*80}")
            print(f"POOLED REGRESSION ANALYSIS")
            print(f"{'='*80}")

            try:
                regression_analyzer = FixedRegressionAnalysis('data_files/expanded_earnings_analysis_results.csv')

                # Descriptive statistics
                regression_analyzer.descriptive_statistics()
                # regression_analyzer.plot_descriptive_analysis()  # Only plot at the end

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
                    regression_summary.to_csv('data_files/pooled_regression_summary.csv', index=False)
                    print(f"✓ Pooled regression summary saved to data_files/pooled_regression_summary.csv")

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
                    model_details_df.to_csv('data_files/pooled_regression_details.csv', index=False)
                    print(f"✓ Detailed regression results saved to data_files/pooled_regression_details.csv")

                    # Diagnostic tests for main model
                    if basic_models and len(basic_models) > 0 and basic_models[0] is not None:
                        regression_analyzer.diagnostic_tests(basic_models[0])

                    # Only plot at the end
                    regression_analyzer.plot_descriptive_analysis()
                    regression_analyzer.plot_regression_results(basic_models)
                    regression_analyzer.plot_sector_analysis()

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

                    # Year-by-year analysis removed - insufficient data per year
                    print(f"\n{'='*80}")
                    print(f"YEAR-BY-YEAR ANALYSIS SKIPPED")
                    print(f"{'='*80}")
                    print("Year-by-year analysis has been disabled due to insufficient data points per year.")
                    print("Most years have only 4 observations (quarterly earnings), which is insufficient")
                    print("for reliable regression analysis. Focus on pooled analysis instead.")

            except Exception as e:
                print(f"  Regression analysis error: {e}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main function to run expanded analysis.
    """
    run_expanded_analysis()

if __name__ == "__main__":
    main()
    # At the end, print and save sector regression summary with stock dummies
    from regression_analysis import FixedRegressionAnalysis
    print("\n=== FINAL SECTOR REGRESSION SUMMARY (WITH STOCK DUMMIES) ===")
    analysis = FixedRegressionAnalysis('data_files/expanded_earnings_analysis_results.csv')
    sector_results = analysis.run_sector_specific_regressions()
    
    if not sector_results.empty:
        print("\nSector Regression Summary (with stock dummies):")
        print(sector_results.to_string(index=False))
        sector_results.to_csv('data_files/sector_regression_results.csv', index=False)
        print("✓ Sector regression results with stock dummies saved to data_files/sector_regression_results.csv")
    else:
        print("✗ No sector regression results generated") 
