#!/usr/bin/env python3
"""
Main execution script for Earnings Implied Volatility Analysis
Expanded to top 100 market cap stocks with year-by-year analysis
Now includes dispersion coefficient calculation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds
from automated_analysis import AutomatedEarningsAnalysis
from regression_analysis import FixedRegressionAnalysis

# Add dispersion analysis functionality
def calculate_dispersion_coefficient(db, ticker, earnings_date, days_before=21):
    """
    Calculate dispersion coefficient for a stock around earnings.
    Dispersion = Standard deviation of analyst estimates / Mean of analyst estimates
    
    Args:
        db: WRDS database connection
        ticker: Stock ticker
        earnings_date: Earnings announcement date (can be string or datetime)
        days_before: Days before earnings to get estimates (default: 21)
    
    Returns:
        dispersion: Dispersion coefficient or None if not available
    """
    try:
        # Ensure earnings_date is a datetime object
        if isinstance(earnings_date, str):
            earnings_date = pd.to_datetime(earnings_date)
        elif not isinstance(earnings_date, (datetime, pd.Timestamp)):
            print(f"Error: earnings_date must be string or datetime, got {type(earnings_date)}")
            return None
        
        # Get CUSIP for the ticker
        cusip_query = f"""
        SELECT DISTINCT cusip
        FROM comp.fundq
        WHERE tic = '{ticker}'
        AND rdq = '{earnings_date.strftime('%Y-%m-%d')}'
        LIMIT 1
        """
        print(f"Executing CUSIP query: {cusip_query}")
        cusip_result = db.raw_sql(cusip_query)
        
        # Debug: Print result type and structure
        print(f"CUSIP result type: {type(cusip_result)}")
        print(f"CUSIP result: {cusip_result}")
        
        # Handle different WRDS result types
        if hasattr(cusip_result, 'empty'):
            # DataFrame result
            if cusip_result.empty:
                print(f"No CUSIP found for {ticker} on {earnings_date.strftime('%Y-%m-%d')}")
                # Try to find any CUSIP for this ticker
                fallback_query = f"""
                SELECT DISTINCT cusip, rdq
                FROM comp.fundq
                WHERE tic = '{ticker}'
                AND rdq IS NOT NULL
                ORDER BY rdq DESC
                LIMIT 5
                """
                fallback_result = db.raw_sql(fallback_query)
                if hasattr(fallback_result, 'empty') and not fallback_result.empty:
                    print(f"Available dates for {ticker}: {fallback_result['rdq'].tolist()}")
                return None
        else:
            # Handle other result types (list, dict, etc.)
            print(f"Unexpected result type: {type(cusip_result)}")
            if isinstance(cusip_result, list) and len(cusip_result) > 0:
                cusip_result = pd.DataFrame(cusip_result)
            elif isinstance(cusip_result, dict):
                cusip_result = pd.DataFrame([cusip_result])
            else:
                print(f"Cannot handle result type: {type(cusip_result)}")
                return None
        
        # Now extract CUSIP from the result
        if hasattr(cusip_result, 'iloc'):
            cusip = cusip_result.iloc[0]['cusip']
        else:
            print(f"Result does not have expected structure")
            return None
            
        if pd.isna(cusip):
            print(f"CUSIP is null for {ticker} on {earnings_date.strftime('%Y-%m-%d')}")
            return None
            
        cusip8 = str(cusip)[:8]
        print(f"Found CUSIP {cusip8} for {ticker}")
        
        # Get IBES estimates for one-quarter-ahead EPS
        estimate_date = earnings_date - timedelta(days=days_before)
        estimate_date_str = estimate_date.strftime('%Y-%m-%d')
        
        print(f"Looking for IBES estimates on {estimate_date_str} (21 days before earnings)")
        
        ibes_query = f"""
        SELECT meanest, stdev, numest
        FROM tr_ibes.statsum_epsus
        WHERE cusip LIKE '{cusip8}%'
        AND statpers = '{estimate_date_str}'
        AND measure = 'EPS'
        AND fiscalp = 'QTR'
        AND fpi = 1  -- One-quarter-ahead
        LIMIT 1
        """
        
        ibes_result = db.raw_sql(ibes_query)
        
        # Debug: Print IBES result type and structure
        print(f"IBES result type: {type(ibes_result)}")
        print(f"IBES result: {ibes_result}")
        
        # Handle different WRDS result types for IBES
        if hasattr(ibes_result, 'empty'):
            # DataFrame result
            if ibes_result.empty:
                print(f"No IBES estimates found for CUSIP {cusip8} on {estimate_date_str}")
                return None
        else:
            # Handle other result types
            print(f"IBES result type: {type(ibes_result)}")
            if isinstance(ibes_result, list) and len(ibes_result) > 0:
                ibes_result = pd.DataFrame(ibes_result)
            elif isinstance(ibes_result, dict):
                ibes_result = pd.DataFrame([ibes_result])
            else:
                print(f"Cannot handle IBES result type: {type(ibes_result)}")
                return None
        
        # Extract data from IBES result
        if hasattr(ibes_result, 'iloc'):
            mean_est = ibes_result.iloc[0]['meanest']
            stdev_est = ibes_result.iloc[0]['stdev']
        else:
            print(f"IBES result does not have expected structure")
            return None
        
        if pd.isna(mean_est) or pd.isna(stdev_est) or mean_est == 0:
            print(f"Invalid IBES data: mean={mean_est}, stdev={stdev_est}")
            return None
            
        # Calculate dispersion
        dispersion = abs(stdev_est / mean_est)
        print(f"Calculated dispersion: {dispersion:.4f} (stdev={stdev_est:.4f}, mean={mean_est:.4f})")
        return dispersion
        
    except Exception as e:
        print(f"Error calculating dispersion for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_top_market_cap_quarterly(db, start_year, end_year, num_top_stocks=100):
    """
    Fetches the top N stocks by market capitalization for the first trading day
    of each quarter from a given start year to an end year using CRSP monthly data.

    Parameters:
    - db: WRDS database connection object.
    - start_year (int): The starting year (e.g., 2005).
    - end_year (int): The ending year (e.g., 2023).
    - num_top_stocks (int): The number of top stocks to retrieve for each quarter (default is 100).

    Returns:
    - pandas.DataFrame: DataFrame containing the top stocks by market cap
                        for the first trading day of each quarter.
    """
    # Construct the start and end date strings
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # SQL query to fetch relevant monthly data from CRSP.msf
    query = f"""
    SELECT
        permno,
        date,
        prc,
        shrout,
        prc * shrout AS mktcap -- Calculate market cap
    FROM
        crsp.msf
    WHERE
        date >= '{start_date}'
        AND date <= '{end_date}'
        AND prc IS NOT NULL -- Ensure price is not null for calculation
        AND shrout IS NOT NULL -- Ensure shares outstanding is not null for calculation
    ORDER BY
        date, mktcap DESC;
    """

    monthly_market_cap_df = db.raw_sql(query)
    print(f"Retrieved {len(monthly_market_cap_df)} monthly market cap records")

    # Convert the 'date' column to datetime objects
    monthly_market_cap_df['date'] = pd.to_datetime(monthly_market_cap_df['date'])

    # Filter for the approximate first trading day of each quarter (month-end of Jan, Apr, Jul, Oct)
    quarterly_starts_df = monthly_market_cap_df[
        monthly_market_cap_df['date'].dt.month.isin([1, 4, 7, 10])
    ].copy()

    print(f"Found {len(quarterly_starts_df)} quarterly start dates")

    # Initialize an empty list to store the top stocks for each quarter
    top_stocks_quarterly = []

    # Group by date and get the top N for each date
    for date, group in quarterly_starts_df.groupby('date'):
        # Sort by market cap in descending order and get the top N
        top_n_for_date = group.sort_values(by='mktcap', ascending=False).head(num_top_stocks)
        top_stocks_quarterly.append(top_n_for_date)

    # Concatenate the results from all quarters into a single DataFrame
    top_by_market_cap_quarterly_df = pd.concat(top_stocks_quarterly, ignore_index=True)

    return top_by_market_cap_quarterly_df


def get_top_stocks_for_analysis(db, start_year, end_year, num_top_stocks=100):
    """
    Get the top N stocks by market cap for the analysis period.
    Returns a list of unique stock identifiers that can be used for analysis.
    """
    print(f"Fetching top {num_top_stocks} stocks by market cap for {start_year}-{end_year}...")
    
    # Get quarterly top stocks
    quarterly_df = get_top_market_cap_quarterly(db, start_year, end_year, num_top_stocks)
    
    # Get unique PERMNOs and convert to tickers
    unique_permnos = quarterly_df['permno'].unique()
    print(f"Found {len(unique_permnos)} unique stocks across quarters")
    
    # Convert PERMNO to ticker using CRSP names file
    ticker_query = f"""
    SELECT DISTINCT permno, ticker
    FROM crsp.names
    WHERE permno IN ({','.join(map(str, unique_permnos))})
    AND ticker IS NOT NULL
    AND ticker != ''
    """
    
    ticker_df = db.raw_sql(ticker_query)
    print(f"Successfully mapped {len(ticker_df)} stocks to tickers")
    
    # Return list of tickers
    return ticker_df['ticker'].tolist()


def get_large_cap_stocks():
    """
    Legacy function - kept for backward compatibility but now returns top market cap stocks.
    Legacy function - kept for backward compatibility but now returns top market cap stocks.
    """
    # This function is now deprecated in favor of get_top_stocks_for_analysis
    # Return a default list for fallback
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Minimal fallback list

    # This function is now deprecated in favor of get_top_stocks_for_analysis
    # Return a default list for fallback
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Minimal fallback list


def run_expanded_analysis():
    """
    Run expanded analysis on top 100 market cap stocks with year-by-year breakdown.
    Run expanded analysis on top 100 market cap stocks with year-by-year breakdown.
    """
    print("EXPANDED EARNINGS VOLATILITY ANALYSIS - TOP 100 MARKET CAP STOCKS")
    print("EXPANDED EARNINGS VOLATILITY ANALYSIS - TOP 100 MARKET CAP STOCKS")
    print("="*80)

    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        print("✓ Connected to WRDS")

        # Analysis parameters
        start_date = '2005-01-01'
        end_date = '2024-12-31'
        start_year = 2005
        end_year = 2024
        analysis_days_before = 30

        # Get top market cap stocks dynamically
        stocks = get_top_stocks_for_analysis(db, start_year, end_year, num_top_stocks=100)
        print(f"✓ Selected {len(stocks)} top market cap stocks for analysis")

        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)

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
                    
                    # Add dispersion coefficient for each earnings event
                    print(f"  Calculating dispersion coefficients...")
                    for idx, row in results_df.iterrows():
                        earnings_date = pd.to_datetime(row['earnings_date'])
                        dispersion = calculate_dispersion_coefficient(db, ticker, earnings_date)
                        results_df.loc[idx, 'dispersion'] = dispersion
                    
                    # Count how many dispersion values were successfully calculated
                    dispersion_count = results_df['dispersion'].notna().sum()
                    print(f"  ✓ Dispersion calculated for {dispersion_count}/{len(results_df)} events")
                    
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
                        
                        # Add dispersion coefficient info if available
                        if 'dispersion' in model.params.index:
                            model_info.update({
                                'dispersion_coef': model.params['dispersion'],
                                'dispersion_tstat': model.tvalues['dispersion'],
                                'dispersion_pvalue': model.pvalues['dispersion'],
                                'dispersion_std_error': model.bse['dispersion']
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
                        
                        # Check if dispersion is available and significant
                        if 'dispersion' in model1.params.index:
                            print(f"\nDispersion Analysis:")
                            print(f"  Dispersion coefficient: {model1.params['dispersion']:.4f}")
                            print(f"  T-statistic: {model1.tvalues['dispersion']:.3f}")
                            print(f"  P-value: {model1.pvalues['dispersion']:.4f}")
                            
                            if model1.pvalues['dispersion'] < 0.05:
                                print(f"  Conclusion: Dispersion coefficient is significant (p < 0.05)")
                            else:
                                print(f"  Conclusion: Dispersion coefficient is not significant (p >= 0.05)")

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
        
        # Print summary of factors included in the analysis
        print(f"\n{'='*80}")
        print(f"FACTORS INCLUDED IN THE ANALYSIS")
        print(f"{'='*80}")
        print(f"✓ Core Factors:")
        print(f"  - REVR (Realized Earnings Volatility Ratio)")
        print(f"  - IEVR (Implied Earnings Volatility Ratio)")
        print(f"  - Dispersion (Analyst Estimate Dispersion) - NEWLY ADDED")
        
        print(f"\n✓ Option Surface Features:")
        print(f"  - Skew Ratio (90Put/110Call)")
        print(f"  - Normative Implied Volatility")
        print(f"  - Normative Realized Volatility")
        print(f"  - Short-term vs Medium-term Volatility (ST/MT)")
        print(f"  - Moneyness and Time-to-Expiry")
        
        print(f"\n✓ Market Conditions:")
        print(f"  - COVID Period Dummies")
        print(f"  - Post-COVID Period Dummies")
        print(f"  - Sector Dummies")
        print(f"  - Stock Fixed Effects")
        print(f"  - Time Fixed Effects")
        
        print(f"\n⚠ Missing Factors (Not Currently Included):")
        print(f"  - Fama-French Factors (MKT, SMB, HML, RMW, CMA, UMD)")
        print(f"  - Market Volatility (VIX)")
        print(f"  - Interest Rate Factors")
        print(f"  - Liquidity Measures")
        
        print(f"\n✓ Dispersion Integration Complete:")
        print(f"  - Added to main analysis pipeline")
        print(f"  - Included in regression models")
        print(f"  - Saved to output CSV files")
        print(f"  - Available for further analysis") 