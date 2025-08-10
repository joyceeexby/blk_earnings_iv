#!/usr/bin/env python3
"""
Debug script to check available columns in S&P 500 constituents data
"""

import pandas as pd
import wrds

def debug_sp500_columns():
    """
    Check what columns are available in the S&P 500 constituents data
    """
    print("DEBUGGING S&P 500 CONSTITUENTS COLUMNS")
    print("="*50)
    
    # Connect to WRDS
    try:
        db = wrds.Connection()
        print("✓ Connected to WRDS")
    except Exception as e:
        print(f"✗ Error connecting to WRDS: {e}")
        return
    
    # Get S&P 500 constituents
    print("Getting S&P 500 constituents...")
    sp500_query = """
        SELECT *
        FROM comp_na_daily_all.wrds_idx_cst_current t
        WHERE indexname = 'S&P 500'
    """
    sp500_constituents = db.raw_sql(sp500_query)
    print(f"Retrieved {len(sp500_constituents)} S&P 500 constituents")
    
    # Check columns
    print(f"\nAvailable columns:")
    print(f"Total columns: {len(sp500_constituents.columns)}")
    for i, col in enumerate(sp500_constituents.columns):
        print(f"{i+1:2d}. {col}")
    
    # Show first few rows
    print(f"\nFirst 3 rows of data:")
    print(sp500_constituents.head(3))
    
    # Check for CUSIP-related columns
    print(f"\nColumns containing 'cusip' (case insensitive):")
    cusip_cols = [col for col in sp500_constituents.columns if 'cusip' in col.lower()]
    if cusip_cols:
        for col in cusip_cols:
            print(f"  - {col}")
            print(f"    Sample values: {sp500_constituents[col].head(3).tolist()}")
    else:
        print("  No columns containing 'cusip' found")
    
    # Check for ticker-related columns
    print(f"\nColumns containing 'ticker' or 'tic' (case insensitive):")
    ticker_cols = [col for col in sp500_constituents.columns if 'ticker' in col.lower() or 'tic' in col.lower()]
    if ticker_cols:
        for col in ticker_cols:
            print(f"  - {col}")
            print(f"    Sample values: {sp500_constituents[col].head(3).tolist()}")
    else:
        print("  No ticker-related columns found")
    
    # Check for GVKEY (alternative identifier)
    print(f"\nColumns containing 'gvkey' (case insensitive):")
    gvkey_cols = [col for col in sp500_constituents.columns if 'gvkey' in col.lower()]
    if gvkey_cols:
        for col in gvkey_cols:
            print(f"  - {col}")
            print(f"    Sample values: {sp500_constituents[col].head(3).tolist()}")
    else:
        print("  No GVKEY-related columns found")
    
    return sp500_constituents

if __name__ == "__main__":
    sp500_data = debug_sp500_columns() 