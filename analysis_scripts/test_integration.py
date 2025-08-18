#!/usr/bin/env python3
"""
Test script to verify the integration of dispersion analysis, Fama-French factors,
and option surface features into the automated earnings analysis pipeline.
"""

import pandas as pd
import numpy as np
import wrds
from automated_analysis import AutomatedEarningsAnalysis
import traceback

def test_integration():
    """
    Test the integrated features with a single stock.
    """
    print("TESTING INTEGRATED FEATURES")
    print("="*60)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                           password="JoyceXu020205")
        print("✓ Connected to WRDS")
        
        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)
        
        # Test with a single stock and short time period
        ticker = 'AAPL'
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        analysis_days_before = 30
        
        print(f"\nTesting integration with {ticker} from {start_date} to {end_date}")
        
        # Analyze multiple events
        results_df = analyzer.analyze_multiple_events(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            analysis_days_before=analysis_days_before
        )
        
        if results_df is not None and not results_df.empty:
            print(f"\n✓ Integration test successful!")
            print(f"Retrieved {len(results_df)} events")
            
            # Check that new columns exist
            expected_columns = [
                'earnings_date', 'revr', 'ievr', 'ratio',
                'analyst_dispersion', 'num_analysts',
                'mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'mkt_return',
                'TERM_RATIO', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'surface_date'
            ]
            
            missing_columns = [col for col in expected_columns if col not in results_df.columns]
            if missing_columns:
                print(f"⚠ Missing columns: {missing_columns}")
            else:
                print(f"✓ All expected columns present")
            
            # Show data quality
            print(f"\nData Quality Check:")
            for col in ['analyst_dispersion', 'mkt_rf', 'smb', 'TERM_RATIO', 'SKEW']:
                if col in results_df.columns:
                    coverage = results_df[col].notna().sum()
                    print(f"  {col}: {coverage}/{len(results_df)} ({coverage/len(results_df)*100:.1f}%)")
            
            # Show sample data
            print(f"\nSample Results:")
            sample_cols = ['earnings_date', 'revr', 'ievr', 'analyst_dispersion', 'mkt_rf', 'TERM_RATIO', 'SKEW']
            available_cols = [col for col in sample_cols if col in results_df.columns]
            print(results_df[available_cols].head().to_string(index=False))
            
            # Save test results
            results_df.to_csv('data_files/test_integration_results.csv', index=False)
            print(f"\n✓ Test results saved to data_files/test_integration_results.csv")
            
        else:
            print("✗ Integration test failed - no results returned")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        traceback.print_exc()

def test_individual_features():
    """
    Test individual features separately to isolate any issues.
    """
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL FEATURES")
    print("="*60)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="joycexu020113",
                           password="JoyceXu020205")
        print("✓ Connected to WRDS")
        
        # Initialize analysis
        analyzer = AutomatedEarningsAnalysis(db)
        
        # Test ticker and date
        ticker = 'AAPL'
        earnings_date = '2023-02-02'  # Example earnings date
        
        print(f"\n1. Testing analyst dispersion for {ticker} on {earnings_date}")
        print("-" * 50)
        dispersion, num_analysts = analyzer.get_analyst_dispersion(ticker, earnings_date, lookback_days=21)
        print(f"Result: dispersion={dispersion}, num_analysts={num_analysts}")
        
        print(f"\n2. Testing Fama-French factors for {earnings_date}")
        print("-" * 50)
        ff_factors = analyzer.get_fama_french_factors(earnings_date, lookback_days=21)
        print(f"Result: {ff_factors}")
        
        print(f"\n3. Testing option surface features for {ticker} on {earnings_date}")
        print("-" * 50)
        option_features = analyzer.get_option_surface_features(ticker, earnings_date, n_lag=15)
        print(f"Result: {option_features}")
        
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error during individual feature test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Test individual features first
    test_individual_features()
    
    # Then test full integration
    test_integration()