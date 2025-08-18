#!/usr/bin/env python3
"""
Debug script for Top 100 Market Cap Stocks analysis
Identifies exactly why earnings dates or events are not being found
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
import warnings
warnings.filterwarnings('ignore')

def debug_top100_analysis():
    """
    Debug the top 100 analysis step by step
    """
    print("DEBUGGING TOP 100 MARKET CAP STOCKS ANALYSIS")
    print("="*80)

    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        print("✓ Connected to WRDS")

        # Test with a smaller, more recent period first
        start_year = 2023
        end_year = 2023
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        print(f"Testing period: {start_date} to {end_date}")
        
        # Step 1: Test market cap query
        print(f"\n{'='*60}")
        print("STEP 1: TESTING MARKET CAP QUERY")
        print(f"{'='*60}")
        
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
            date, mktcap DESC
        LIMIT 1000;
        """
        
        print(f"Executing market cap query...")
        result = db.raw_sql(query)
        
        if hasattr(result, 'empty') and not result.empty:
            print(f"✓ Market cap query successful: {len(result)} records")
            print(f"✓ Date range: {result['date'].min()} to {result['date'].max()}")
            
            # Check quarterly dates
            result['date'] = pd.to_datetime(result['date'])
            quarterly_dates = result[result['date'].dt.month.isin([1, 4, 7, 10])]
            print(f"✓ Found {len(quarterly_dates)} quarterly dates")
            
            if len(quarterly_dates) > 0:
                # Get top stocks for one quarter
                sample_date = quarterly_dates['date'].iloc[0]
                print(f"✓ Sample quarter: {sample_date}")
                
                quarter_stocks = result[result['date'] == sample_date].sort_values('mktcap', ascending=False).head(10)
                print(f"✓ Top 10 stocks for {sample_date}:")
                print(quarter_stocks[['permno', 'mktcap']].to_string())
                
                # Test ticker mapping for these stocks
                unique_permnos = quarter_stocks['permno'].unique()
                print(f"\nTesting ticker mapping for {len(unique_permnos)} stocks...")
                
                ticker_query = f"""
                SELECT DISTINCT permno, ticker
                FROM crsp.stocknames
                WHERE permno IN ({','.join(map(str, unique_permnos))})
                AND ticker IS NOT NULL
                AND ticker != ''
                """
                
                ticker_result = db.raw_sql(ticker_query)
                
                if hasattr(ticker_result, 'empty') and not ticker_result.empty:
                    print(f"✓ Ticker mapping successful: {len(ticker_result)} stocks mapped")
                    print(f"Sample tickers: {ticker_result['ticker'].tolist()}")
                    
                    # Test earnings dates for first ticker
                    test_ticker = ticker_result['ticker'].iloc[0]
                    print(f"\nTesting earnings dates for {test_ticker}...")
                    
                    # Step 2: Test earnings dates query
                    print(f"\n{'='*60}")
                    print("STEP 2: TESTING EARNINGS DATES QUERY")
                    print(f"{'='*60}")
                    
                    earnings_query = f"""
                    SELECT DISTINCT rdq as earnings_date
                    FROM comp.fundq
                    WHERE tic = '{test_ticker}'
                    AND rdq >= '{start_date}'
                    AND rdq <= '{end_date}'
                    AND rdq IS NOT NULL
                    ORDER BY rdq
                    """
                    
                    print(f"Executing earnings query: {earnings_query}")
                    earnings_result = db.raw_sql(earnings_query)
                    
                    if hasattr(earnings_result, 'empty') and not earnings_result.empty:
                        print(f"✓ Earnings query successful: {len(earnings_result)} dates found")
                        print(f"Earnings dates: {earnings_result['earnings_date'].tolist()}")
                        
                        # Test IEVR/REVR calculation for first earnings date
                        if len(earnings_result) > 0:
                            test_earnings_date = pd.to_datetime(earnings_result['earnings_date'].iloc[0])
                            print(f"\nTesting IEVR/REVR calculation for {test_earnings_date}...")
                            
                            # Step 3: Test IEVR/REVR calculation
                            print(f"\n{'='*60}")
                            print("STEP 3: TESTING IEVR/REVR CALCULATION")
                            print(f"{'='*60}")
                            
                            # Test option data query
                            analysis_days_before = 30
                            start_date_opt = (test_earnings_date - timedelta(days=analysis_days_before)).strftime('%Y-%m-%d')
                            end_date_opt = (test_earnings_date + timedelta(days=5)).strftime('%Y-%m-%d')
                            
                            print(f"Option data period: {start_date_opt} to {end_date_opt}")
                            
                            # Test optionm.iv table
                            iv_query = f"""
                            SELECT COUNT(*) as count
                            FROM optionm.iv
                            WHERE symbol = '{test_ticker}'
                            AND date >= '{start_date_opt}'
                            AND date <= '{end_date_opt}'
                            """
                            
                            print(f"Testing optionm.iv table...")
                            iv_count = db.raw_sql(iv_query)
                            
                            if hasattr(iv_count, 'iloc'):
                                count = iv_count.iloc[0]['count']
                                print(f"✓ Option data: {count} records found")
                            else:
                                print("⚠ Option data query failed")
                            
                            # Test CRSP price data
                            price_query = f"""
                            SELECT COUNT(*) as count
                            FROM crsp.dsf
                            WHERE ticker = '{test_ticker}'
                            AND date >= '{start_date_opt}'
                            AND date <= '{end_date_opt}'
                            """
                            
                            print(f"Testing CRSP price data...")
                            price_count = db.raw_sql(price_query)
                            
                            if hasattr(price_count, 'iloc'):
                                count = price_count.iloc[0]['count']
                                print(f"✓ Price data: {count} records found")
                            else:
                                print("⚠ Price data query failed")
                            
                        else:
                            print("❌ No earnings dates found")
                    else:
                        print("❌ Earnings query failed or returned no results")
                        print(f"Query: {earnings_query}")
                        
                        # Try alternative approach - check if comp.fundq exists
                        print(f"\nChecking if comp.fundq table exists...")
                        check_query = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'comp' 
                        AND table_name LIKE '%fund%'
                        """
                        
                        check_result = db.raw_sql(check_query)
                        if hasattr(check_result, 'empty') and not check_result.empty:
                            print(f"Available comp tables: {check_result['table_name'].tolist()}")
                        else:
                            print("⚠ Could not check available tables")
                else:
                    print("❌ Ticker mapping failed")
            else:
                print("❌ No quarterly dates found")
        else:
            print("❌ Market cap query failed")
            
        print(f"\n{'='*80}")
        print("DEBUG SUMMARY")
        print(f"{'='*80}")
        print("Check the output above to identify which step is failing.")
        print("Common issues:")
        print("1. Table names (crsp.msf, crsp.stocknames, comp.fundq, optionm.iv)")
        print("2. Date formats and ranges")
        print("3. Ticker symbols vs company names")
        print("4. Data availability for the specified period")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run debug
    """
    debug_top100_analysis()

if __name__ == "__main__":
    main()
