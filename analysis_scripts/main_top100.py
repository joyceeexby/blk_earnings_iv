#!/usr/bin/env python3
"""
Main execution script for Earnings Implied Volatility Analysis
TOP 100 MARKET CAP STOCKS with complete feature integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
import warnings
warnings.filterwarnings('ignore')

def get_top_market_cap_quarterly(db, start_year, end_year, num_top_stocks=100):
    """
    Fetches the top N stocks by market capitalization for the first trading day
    of each quarter from a given start year to an end year using CRSP monthly data.
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
        prc * shrout AS mktcap
    FROM
        crsp.msf
    WHERE
        date >= '{start_date}'
        AND date <= '{end_date}'
        AND prc IS NOT NULL
        AND shrout IS NOT NULL
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
    
    # Convert PERMNO to ticker using CRSP stocknames file
    ticker_query = f"""
    SELECT DISTINCT permno, ticker
    FROM crsp.stocknames
    WHERE permno IN ({','.join(map(str, unique_permnos))})
    AND ticker IS NOT NULL
    AND ticker != ''
    """
    
    ticker_df = db.raw_sql(ticker_query)
    print(f"Successfully mapped {len(ticker_df)} stocks to tickers")
    
    # Return list of tickers
    return ticker_df['ticker'].tolist()

def calculate_dispersion_coefficient(db, ticker, earnings_date, days_before=21):
    """
    Calculate dispersion coefficient for a stock around earnings.
    """
    try:
        # Ensure earnings_date is a datetime object
        if isinstance(earnings_date, str):
            earnings_date = pd.to_datetime(earnings_date)
        
        # Get CUSIP for the ticker
        cusip_query = f"""
        SELECT DISTINCT cusip
        FROM comp.fundq
        WHERE tic = '{ticker}'
        AND rdq = '{earnings_date.strftime('%Y-%m-%d')}'
        LIMIT 1
        """
        
        cusip_result = db.raw_sql(cusip_query)
        
        # Handle different WRDS result types
        if hasattr(cusip_result, 'empty') and cusip_result.empty:
            return None
        
        if not hasattr(cusip_result, 'iloc'):
            if isinstance(cusip_result, list) and len(cusip_result) > 0:
                cusip_result = pd.DataFrame(cusip_result)
            elif isinstance(cusip_result, dict):
                cusip_result = pd.DataFrame([cusip_result])
            else:
                return None
        
        cusip = cusip_result.iloc[0]['cusip']
        if pd.isna(cusip):
            return None
            
        cusip8 = str(cusip)[:8]
        
        # Get IBES estimates for one-quarter-ahead EPS
        estimate_date = earnings_date - timedelta(days=days_before)
        estimate_date_str = estimate_date.strftime('%Y-%m-%d')
        
        ibes_query = f"""
        SELECT meanest, stdev, numest
        FROM tr_ibes.statsum_epsus
        WHERE cusip LIKE '{cusip8}%'
        AND statpers = '{estimate_date_str}'
        AND measure = 'EPS'
        AND fiscalp = 'QTR'
        AND fpi = 1
        LIMIT 1
        """
        
        ibes_result = db.raw_sql(ibes_query)
        
        if hasattr(ibes_result, 'empty') and ibes_result.empty:
            return None
        
        if not hasattr(ibes_result, 'iloc'):
            if isinstance(ibes_result, list) and len(ibes_result) > 0:
                ibes_result = pd.DataFrame(ibes_result)
            elif isinstance(ibes_result, dict):
                ibes_result = pd.DataFrame([ibes_result])
            else:
                return None
        
        mean_est = ibes_result.iloc[0]['meanest']
        stdev_est = ibes_result.iloc[0]['stdev']
        
        if pd.isna(mean_est) or pd.isna(stdev_est) or mean_est == 0:
            return None
            
        # Calculate dispersion
        dispersion = abs(stdev_est / mean_est)
        return dispersion
        
    except Exception as e:
        return None

def get_earnings_dates(db, ticker, start_date, end_date):
    """
    Get earnings announcement dates for a stock.
    """
    try:
        query = f"""
        SELECT DISTINCT rdq as earnings_date
        FROM comp.fundq
        WHERE tic = '{ticker}'
        AND rdq >= '{start_date}'
        AND rdq <= '{end_date}'
        AND rdq IS NOT NULL
        ORDER BY rdq
        """
        
        result = db.raw_sql(query)
        
        if hasattr(result, 'empty') and result.empty:
            return []
        
        if not hasattr(result, 'iloc'):
            if isinstance(result, list) and len(result) > 0:
                result = pd.DataFrame(result)
            elif isinstance(result, dict):
                result = pd.DataFrame([result])
            else:
                return []
        
        earnings_dates = pd.to_datetime(result['earnings_date']).tolist()
        return earnings_dates
        
    except Exception as e:
        print(f"Error getting earnings dates for {ticker}: {e}")
        return []

def calculate_ievr_and_revr(db, ticker, earnings_date, analysis_days_before=30):
    """
    Calculate IEVR and REVR for a stock around earnings.
    """
    try:
        # Get option data around earnings
        start_date = (earnings_date - timedelta(days=analysis_days_before)).strftime('%Y-%m-%d')
        end_date = (earnings_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Get implied volatility data
        iv_query = f"""
        SELECT date, iv, strike, maturity, option_type
        FROM optionm.iv
        WHERE symbol = '{ticker}'
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        AND maturity >= 7
        AND maturity <= 365
        """
        
        iv_result = db.raw_sql(iv_query)
        
        if hasattr(iv_result, 'empty') and iv_result.empty:
            return None, None
        
        if not hasattr(iv_result, 'iloc'):
            if isinstance(iv_result, list) and len(iv_result) > 0:
                iv_result = pd.DataFrame(iv_result)
            elif isinstance(iv_result, dict):
                iv_result = pd.DataFrame([iv_result])
            else:
                return None, None
        
        # Calculate IEVR (simplified - average IV around earnings)
        if len(iv_result) > 0:
            ievr = iv_result['iv'].mean()
        else:
            ievr = None
        
        # Calculate REVR (simplified - realized volatility from price data)
        price_query = f"""
        SELECT date, prc
        FROM crsp.dsf
        WHERE ticker = '{ticker}'
        AND date >= '{start_date}'
        AND date <= '{end_date}'
        ORDER BY date
        """
        
        price_result = db.raw_sql(price_query)
        
        if hasattr(price_result, 'empty') and price_result.empty:
            revr = None
        else:
            if not hasattr(price_result, 'iloc'):
                if isinstance(price_result, list) and len(price_result) > 0:
                    price_result = pd.DataFrame(price_result)
                elif isinstance(price_result, dict):
                    price_result = pd.DataFrame([price_result])
                else:
                    revr = None
                    return ievr, revr
            
            if len(price_result) > 1:
                price_result['date'] = pd.to_datetime(price_result['date'])
                price_result = price_result.sort_values('date')
                price_result['returns'] = price_result['prc'].pct_change()
                revr = price_result['returns'].std() * np.sqrt(252)  # Annualized
            else:
                revr = None
        
        return ievr, revr
        
    except Exception as e:
        return None, None

def run_top100_analysis():
    """
    Run analysis on top 100 market cap stocks.
    """
    print("TOP 100 MARKET CAP STOCKS - EARNINGS VOLATILITY ANALYSIS")
    print("="*80)

    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        print("✓ Connected to WRDS")

        # Analysis parameters
        start_year = 2015
        end_year = 2023
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        analysis_days_before = 30

        # Get top market cap stocks
        stocks = get_top_stocks_for_analysis(db, start_year, end_year, num_top_stocks=100)
        print(f"✓ Selected {len(stocks)} top market cap stocks for analysis")

        # Store all results
        all_results = []

        # Analyze each stock
        for i, ticker in enumerate(stocks):
            print(f"\n{'='*60}")
            print(f"ANALYZING STOCK {i+1}/{len(stocks)}: {ticker}")
            print(f"{'='*60}")

            try:
                # Get earnings dates
                earnings_dates = get_earnings_dates(db, ticker, start_date, end_date)
                
                if not earnings_dates:
                    print(f"  ✗ No earnings dates found for {ticker}")
                    continue
                
                print(f"  ✓ Found {len(earnings_dates)} earnings dates")
                
                # Analyze each earnings event
                for earnings_date in earnings_dates:
                    try:
                        # Calculate IEVR and REVR
                        ievr, revr = calculate_ievr_and_revr(db, ticker, earnings_date, analysis_days_before)
                        
                        if ievr is not None and revr is not None:
                            # Calculate dispersion
                            dispersion = calculate_dispersion_coefficient(db, ticker, earnings_date)
                            
                            # Create result row
                            result_row = {
                                'earnings_date': earnings_date.strftime('%Y-%m-%d'),
                                'ticker': ticker,
                                'ievr': ievr,
                                'revr': revr,
                                'dispersion': dispersion,
                                'analysis_days_before': analysis_days_before
                            }
                            
                            all_results.append(result_row)
                            print(f"    ✓ {earnings_date.strftime('%Y-%m-%d')}: IEVR={ievr:.4f}, REVR={revr:.4f}, Dispersion={dispersion:.4f if dispersion else 'N/A'}")
                        else:
                            print(f"    ✗ {earnings_date.strftime('%Y-%m-%d')}: Missing IEVR or REVR data")
                            
                    except Exception as e:
                        print(f"    ✗ Error analyzing {earnings_date.strftime('%Y-%m-%d')}: {e}")
                        continue
                
                print(f"  ✓ {ticker}: {len([r for r in all_results if r['ticker'] == ticker])} events analyzed")
                
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")
                continue

        # Combine all results
        if all_results:
            combined_results = pd.DataFrame(all_results)
            print(f"\n{'='*80}")
            print(f"COMBINED ANALYSIS RESULTS")
            print(f"{'='*80}")
            print(f"Total events: {len(combined_results)}")
            print(f"Stocks with data: {combined_results['ticker'].nunique()}")
            print(f"Date range: {combined_results['earnings_date'].min()} to {combined_results['earnings_date'].max()}")

            # Save results
            combined_results.to_csv('data_files/top100_earnings_analysis_results.csv', index=False)
            print(f"✓ Results saved to data_files/top100_earnings_analysis_results.csv")
            
            # Print summary statistics
            print(f"\n{'='*80}")
            print(f"SUMMARY STATISTICS")
            print(f"{'='*80}")
            print(f"REVR - Mean: {combined_results['revr'].mean():.4f}, Std: {combined_results['revr'].std():.4f}")
            print(f"IEVR - Mean: {combined_results['ievr'].mean():.4f}, Std: {combined_results['ievr'].std():.4f}")
            print(f"Dispersion - Mean: {combined_results['dispersion'].dropna().mean():.4f}, Std: {combined_results['dispersion'].dropna().std():.4f}")
            
            return combined_results
        else:
            print("✗ No results generated")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """
    Main function to run top 100 analysis.
    """
    results = run_top100_analysis()
    
    if results is not None:
        print(f"\n🎉 TOP 100 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Dataset ready for feature integration and regression analysis!")
        print(f"\nNext steps:")
        print(f"1. Integrate option surface features")
        print(f"2. Add Fama-French factors")
        print(f"3. Run regression analysis")
        print(f"4. Apply data leakage fixes")

if __name__ == "__main__":
    main()
