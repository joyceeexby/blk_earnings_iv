#!/usr/bin/env python3
"""
Simple script to check the current state of the data file.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def check_data_file():
    """
    Check the data file and provide basic statistics.
    """
    print("="*80)
    print("DATA FILE CHECK")
    print("="*80)
    
    data_file = 'data_files/expanded_earnings_analysis_results.csv'
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Get file info
    file_stat = os.stat(data_file)
    file_size = file_stat.st_size / (1024 * 1024)  # MB
    file_modified = datetime.fromtimestamp(file_stat.st_mtime)
    
    print(f"File: {data_file}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Last modified: {file_modified}")
    
    # Load and analyze data
    try:
        data = pd.read_csv(data_file)
        print(f"\nData shape: {data.shape}")
        print(f"Observations: {len(data)}")
        print(f"Columns: {len(data.columns)}")
        
        # Check key columns
        key_columns = ['revr', 'ievr', 'ticker', 'earnings_date']
        print(f"\nKey columns present: {[col in data.columns for col in key_columns]}")
        
        # Check for NaN values
        print(f"\nNaN values in key columns:")
        for col in key_columns:
            if col in data.columns:
                nan_count = data[col].isna().sum()
                print(f"  {col}: {nan_count} ({nan_count/len(data)*100:.1f}%)")
        
        # Check for infinite values
        print(f"\nInfinite values in key columns:")
        for col in key_columns:
            if col in data.columns:
                inf_count = np.isinf(data[col]).sum()
                print(f"  {col}: {inf_count}")
        
        # Check unique values
        if 'ticker' in data.columns:
            print(f"\nUnique tickers: {data['ticker'].nunique()}")
            print(f"Tickers: {sorted(data['ticker'].unique())}")
        
        if 'earnings_date' in data.columns:
            print(f"\nDate range: {data['earnings_date'].min()} to {data['earnings_date'].max()}")
        
        # Calculate expected train/test split
        clean_data = data.dropna(subset=['revr', 'ievr'])
        clean_data = clean_data[np.isfinite(clean_data['revr']) & np.isfinite(clean_data['ievr'])]
        
        expected_train = int(len(clean_data) * 0.8)
        expected_test = len(clean_data) - expected_train
        
        print(f"\nExpected train/test split: {expected_train}/{expected_test}")
        print(f"Teammate's results: 2254/564")
        
        if expected_train == 2254 and expected_test == 564:
            print("✓ Your data should match teammate's results!")
        else:
            print(f"⚠ Your data differs from teammate's results")
            print(f"  Difference: {expected_train - 2254} training, {expected_test - 564} testing")
        
    except Exception as e:
        print(f"❌ Error reading data: {e}")

if __name__ == "__main__":
    check_data_file() 