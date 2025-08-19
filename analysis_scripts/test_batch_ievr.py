#!/usr/bin/env python3
"""
Test script for batch IEVR analysis
Run with a small number of stocks to verify functionality
"""

import pandas as pd
from datetime import datetime
import wrds
from batch_ievr_analysis import BatchIEVRAnalysis

def test_batch_ievr():
    """
    Test the batch IEVR analysis with a small sample
    """
    print("TESTING BATCH IEVR ANALYSIS")
    print("="*50)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami", password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize analyzer with fewer workers for testing
        analyzer = BatchIEVRAnalysis(db, max_workers=2)
        
        # Test with a small sample first
        earnings_date = pd.to_datetime('2023-01-31')
        num_stocks = 10  # Very small sample for testing
        
        print(f"\nTesting with {num_stocks} stocks for earnings date {earnings_date}")
        
        # Run analysis
        results = analyzer.run_batch_analysis(
            earnings_date=earnings_date,
            num_stocks=num_stocks,
            analysis_days_before=30
        )
        
        if results:
            print(f"\n✓ Test successful! Generated {len(results)} results")
            
            # Save test results
            filename = analyzer.save_results("data_files/test_batch_ievr_results.csv")
            
            # Show sample results
            print(f"\nSample results:")
            results_df = pd.DataFrame(results)
            print(results_df[['ticker', 'ievr', 'skew_ratio', 'iv_data_points']].head())
            
        else:
            print("\n✗ Test failed - no results generated")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_ievr()
