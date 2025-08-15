#!/usr/bin/env python3
"""
Main execution script for Earnings Implied Volatility Analysis
Expanded to top 100 market cap stocks with year-by-year analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds
from automated_analysis import AutomatedEarningsAnalysis
from regression_analysis import FixedRegressionAnalysis


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
    """
    # This function is now deprecated in favor of get_top_stocks_for_analysis
    # Return a default list for fallback
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Minimal fallback list


def run_expanded_analysis():
    """
    Run expanded analysis on top 100 market cap stocks with year-by-year breakdown.
    """
    print("EXPANDED EARNINGS VOLATILITY ANALYSIS - TOP 100 MARKET CAP STOCKS")
    print("="*80)

    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        print("✓ Connected to WRDS")

        # Analysis parameters
        start_date = '2015-01-01'
        end_date = '2024-12-31'
        start_year = 2015
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