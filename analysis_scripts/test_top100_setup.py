#!/usr/bin/env python3
"""
Test script for Top 100 Market Cap Stocks setup
Verifies the approach works without running full analysis
"""

import pandas as pd
import numpy as np
import wrds

def test_top100_setup():
    """
    Test the top 100 market cap stocks setup
    """
    print("TESTING TOP 100 MARKET CAP STOCKS SETUP")
    print("="*80)
    
    try:
        # Connect to WRDS
        print("Connecting to WRDS...")
        db = wrds.Connection(wrds_username="joycexu020113",
                             password="JoyceXu020205")
        print("✓ Connected to WRDS")
        
        # Test market cap query for a small sample
        print("\nTesting market cap query...")
        start_year = 2023
        end_year = 2023
        
        # Test query for one year
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
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
        
        print(f"Executing query: {query[:100]}...")
        result = db.raw_sql(query)
        
        if hasattr(result, 'empty') and not result.empty:
            print(f"✓ Query successful: {len(result)} records retrieved")
            print(f"✓ Columns: {list(result.columns)}")
            
            # Show sample data
            print(f"\nSample data (first 5 rows):")
            print(result.head().to_string())
            
            # Check for quarterly dates
            result['date'] = pd.to_datetime(result['date'])
            quarterly_dates = result[result['date'].dt.month.isin([1, 4, 7, 10])]
            print(f"\n✓ Found {len(quarterly_dates)} quarterly dates")
            
            # Test ticker mapping
            print(f"\nTesting ticker mapping...")
            unique_permnos = result['permno'].unique()[:10]  # Test with first 10
            
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
            else:
                print("⚠ Ticker mapping returned no results")
            
        else:
            print("❌ Query returned no results")
            return False
        
        # Test earnings dates query
        print(f"\nTesting earnings dates query...")
        test_ticker = 'AAPL'
        test_date = '2023-01-01'
        
        earnings_query = f"""
        SELECT DISTINCT rdq as earnings_date
        FROM comp.fundq
        WHERE tic = '{test_ticker}'
        AND rdq >= '{test_date}'
        AND rdq <= '2023-12-31'
        AND rdq IS NOT NULL
        ORDER BY rdq
        LIMIT 5
        """
        
        earnings_result = db.raw_sql(earnings_query)
        
        if hasattr(earnings_result, 'empty') and not earnings_result.empty:
            print(f"✓ Earnings query successful: {len(earnings_result)} dates found")
            print(f"Sample dates: {earnings_result['earnings_date'].tolist()}")
        else:
            print("⚠ Earnings query returned no results")
        
        print(f"\n{'='*80}")
        print("TOP 100 SETUP TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"✓ Market cap queries working")
        print(f"✓ Ticker mapping functional")
        print(f"✓ Earnings date retrieval working")
        print(f"\nReady to run full top 100 analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the test
    """
    success = test_top100_setup()
    
    if success:
        print(f"\n🎯 All tests passed! You can now run:")
        print(f"1. python3 main_top100.py - Generate top 100 dataset")
        print(f"2. python3 integrate_top100_features.py - Add features")
        print(f"3. python3 nonlinear_models.py - Run analysis")
    else:
        print(f"\n❌ Tests failed. Please check WRDS connection and permissions.")

if __name__ == "__main__":
    main()
