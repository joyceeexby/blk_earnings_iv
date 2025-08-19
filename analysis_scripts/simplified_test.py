#!/usr/bin/env python3
"""
Simplified test script for Top 100 analysis
Test with minimal scope to identify issues
"""

import wrds
import pandas as pd
from datetime import datetime

def test_minimal_scope():
    """
    Test with minimal scope: 2023, top 10 stocks, single earnings event
    """
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="your_username",
                             password="your_password")
        print("✓ Connected to WRDS")
        
        # Test 1: Basic market cap query (2023 only)
        print("\nTesting market cap query...")
        query1 = """
        SELECT permno, date, prc, shrout
        FROM crsp.msf
        WHERE date >= '2023-01-01' AND date <= '2023-12-31'
        AND prc IS NOT NULL AND shrout IS NOT NULL
        LIMIT 100;
        """
        
        result1 = db.raw_sql(query1)
        print(f"Market cap query: {len(result1)} records")
        
        # Test 2: Stock names
        print("\nTesting stock names...")
        query2 = """
        SELECT permno, ticker FROM crsp.stocknames
        WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
        LIMIT 10;
        """
        
        result2 = db.raw_sql(query2)
        print(f"Stock names: {len(result2)} records")
        
        # Test 3: Earnings dates for AAPL
        print("\nTesting earnings dates...")
        query3 = """
        SELECT rdq FROM comp.fundq
        WHERE tic = 'AAPL' AND rdq >= '2023-01-01'
        ORDER BY rdq LIMIT 5;
        """
        
        result3 = db.raw_sql(query3)
        print(f"Earnings dates: {len(result3)} records")
        
        # Test 4: Option data
        print("\nTesting option data...")
        query4 = """
        SELECT COUNT(*) as count FROM optionm.iv
        WHERE symbol = 'AAPL' AND date >= '2023-01-01'
        LIMIT 1;
        """
        
        result4 = db.raw_sql(query4)
        print(f"Option data: {result4.iloc[0]['count'] if not result4.empty else 'N/A'} records")
        
        print("\n✓ All basic tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_scope()
