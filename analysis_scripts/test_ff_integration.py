#!/usr/bin/env python3
"""
Test script to verify Fama-French 5-factor integration results.
"""

import pandas as pd
import numpy as np

def test_ff_integration():
    """
    Test the Fama-French integration results.
    """
    print("="*80)
    print("TESTING FAMA-FRENCH 5-FACTOR INTEGRATION RESULTS")
    print("="*80)
    
    # Test monthly match data
    print("\n1. Testing Monthly Match Integration:")
    print("-" * 50)
    
    try:
        monthly_data = pd.read_csv('data_files/earnings_with_fama_french_5factor_monthly_match.csv')
        print("Successfully loaded monthly match data")
        print("Shape: {} observations, {} columns".format(monthly_data.shape[0], monthly_data.shape[1]))
        
        # Check Fama-French factors
        ff_columns = [col for col in monthly_data.columns if any(factor in col for factor in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]
        print("Fama-French factors found: {}".format(len(ff_columns)))
        print("Factor columns: {}".format(ff_columns))
        
        # Show correlations with REVR
        if 'revr' in monthly_data.columns:
            print("\nREVR correlations with Fama-French factors:")
            for col in ff_columns:
                if col in monthly_data.columns:
                    corr = monthly_data['revr'].corr(monthly_data[col])
                    print("  {}: {:.4f}".format(col, corr))
        
        # Show correlations with IEVR
        if 'ievr' in monthly_data.columns:
            print("\nIEVR correlations with Fama-French factors:")
            for col in ff_columns:
                if col in monthly_data.columns:
                    corr = monthly_data['ievr'].corr(monthly_data[col])
                    print("  {}: {:.4f}".format(col, corr))
        
    except Exception as e:
        print("Error loading monthly match data: {}".format(e))
    
    # Test lagged monthly data
    print("\n2. Testing Lagged Monthly Integration:")
    print("-" * 50)
    
    try:
        lagged_data = pd.read_csv('data_files/earnings_with_fama_french_5factor_lagged_monthly.csv')
        print("Successfully loaded lagged monthly data")
        print("Shape: {} observations, {} columns".format(lagged_data.shape[0], lagged_data.shape[1]))
        
        # Check Fama-French factors
        ff_columns = [col for col in lagged_data.columns if any(factor in col for factor in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]
        print("Fama-French factors found: {}".format(len(ff_columns)))
        
        # Show correlations with REVR
        if 'revr' in lagged_data.columns:
            print("\nREVR correlations with Fama-French factors (lagged):")
            for col in ff_columns:
                if col in lagged_data.columns:
                    corr = lagged_data['revr'].corr(lagged_data[col])
                    print("  {}: {:.4f}".format(col, corr))
        
    except Exception as e:
        print("Error loading lagged monthly data: {}".format(e))
    
    # Summary statistics
    print("\n3. Summary Statistics:")
    print("-" * 50)
    
    try:
        data = pd.read_csv('data_files/earnings_with_fama_french_5factor_monthly_match.csv')
        
        # Basic stats for key factors
        key_factors = ['SMB', 'HML', 'RMW', 'CMA', 'RF']
        print("Fama-French 5-Factor Summary Statistics:")
        for factor in key_factors:
            if factor in data.columns:
                mean_val = data[factor].mean()
                std_val = data[factor].std()
                min_val = data[factor].min()
                max_val = data[factor].max()
                print("  {}: Mean={:.4f}, Std={:.4f}, Range=[{:.4f}, {:.4f}]".format(
                    factor, mean_val, std_val, min_val, max_val))
        
        # Target variable stats
        if 'revr' in data.columns:
            print("\nREVR Summary Statistics:")
            print("  Mean: {:.4f}".format(data['revr'].mean()))
            print("  Std: {:.4f}".format(data['revr'].std()))
            print("  Min: {:.4f}".format(data['revr'].min()))
            print("  Max: {:.4f}".format(data['revr'].max()))
        
        if 'ievr' in data.columns:
            print("\nIEVR Summary Statistics:")
            print("  Mean: {:.4f}".format(data['ievr'].mean()))
            print("  Std: {:.4f}".format(data['ievr'].std()))
            print("  Min: {:.4f}".format(data['ievr'].min()))
            print("  Max: {:.4f}".format(data['ievr'].max()))
        
    except Exception as e:
        print("Error computing summary statistics: {}".format(e))
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)
    print("Files created:")
    print("  - data_files/earnings_with_fama_french_5factor_monthly_match.csv")
    print("  - data_files/earnings_with_fama_french_5factor_lagged_monthly.csv")
    print("\nNext steps:")
    print("  1. Run nonlinear_models_with_ff.py (after fixing f-string syntax)")
    print("  2. Analyze feature importance")
    print("  3. Compare model performance with and without Fama-French factors")

if __name__ == "__main__":
    test_ff_integration()
