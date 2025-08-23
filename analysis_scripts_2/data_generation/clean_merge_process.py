#!/usr/bin/env python3
"""
Clean 3-Step Merge Process:
Step 1: Add permno to merged_revr_ievr_comprehensive.csv based on mapping in top500_liquidity_2005_2023.csv
Step 2: Left join vol_df.csv based on permno and date, aligned with closest_quote_date
Step 3: Left join options features (SKEW, KURT, IV_RATIO, SMIRK) based on ticker and surface_date, matching to closest_quote_date ±3 days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

def check_files():
    """Check if all required files exist"""
    print("🔍 CHECKING REQUIRED FILES")
    print("="*50)
    
    required_files = [
        'data_files/merged_revr_ievr_comprehensive.csv',
        'data_files/top500_liquidity_2005_2023.csv',
        'data_files/vol_df.csv',
        'data_files/option_features_combined_20250821_222431.xlsx'
    ]
    
    for file_path in required_files:
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ {file_path} - ERROR: {e}")
            return False
    
    print("✅ All required files found!")
    return True

def step1_add_permno():
    """Step 1: Add permno based on mapping in top500_liquidity_2005_2023.csv"""
    print("\n🔄 STEP 1: ADDING PERMNO")
    print("="*50)
    
    # Load data
    print("Loading comprehensive data...")
    comp_df = pd.read_csv('data_files/merged_revr_ievr_comprehensive.csv')
    print(f"✅ Loaded: {len(comp_df):,} rows")
    
    print("Loading mapping data...")
    mapping_df = pd.read_csv('data_files/top500_liquidity_2005_2023.csv')
    print(f"✅ Loaded: {len(mapping_df):,} rows")
    
    # Convert dates
    comp_df['earnings_date'] = pd.to_datetime(comp_df['earnings_date'])
    mapping_df['quarter_start_date'] = pd.to_datetime(mapping_df['quarter_start_date'])
    mapping_df['quarter_end_date'] = pd.to_datetime(mapping_df['quarter_end_date'])
    
    # Initialize new columns
    comp_df['permno'] = np.nan
    comp_df['company_name'] = np.nan
    comp_df['cusip'] = np.nan
    
    print("Adding permno mapping...")
    
    # Process in batches for memory efficiency
    batch_size = 5000
    total_batches = (len(comp_df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(comp_df))
        
        # Process this batch
        for idx in range(start_idx, end_idx):
            ticker = comp_df.loc[idx, 'ticker']
            earnings_date = comp_df.loc[idx, 'earnings_date']
            
            # Find matching mapping
            mask = (mapping_df['ticker'] == ticker) & \
                   (mapping_df['quarter_start_date'] <= earnings_date) & \
                   (mapping_df['quarter_end_date'] >= earnings_date)
            
            if mask.any():
                mapping_row = mapping_df[mask].iloc[0]
                comp_df.loc[idx, 'permno'] = mapping_row['permno']
                comp_df.loc[idx, 'company_name'] = mapping_row['comnam']
                comp_df.loc[idx, 'cusip'] = mapping_row['cusip']
        
        # Progress indicator
        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            print(f"  Processed batch {batch_idx + 1}/{total_batches}")
    
    # Check results
    permno_coverage = comp_df['permno'].notna().sum()
    coverage_pct = (permno_coverage / len(comp_df)) * 100
    
    print(f"✅ Step 1 completed:")
    print(f"  - Total rows: {len(comp_df):,}")
    print(f"  - Rows with permno: {permno_coverage:,} ({coverage_pct:.1f}%)")
    print(f"  - Unique permnos: {comp_df['permno'].nunique():,}")
    
    return comp_df

def step2_join_volatility(comp_df):
    """Step 2: Left join vol_df.csv based on permno and date, aligned with closest_quote_date"""
    print("\n🔄 STEP 2: JOINING VOLATILITY FEATURES")
    print("="*50)
    
    # Load volatility data in chunks
    print("Loading volatility data...")
    vol_chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv('data_files/vol_df.csv', chunksize=chunk_size):
        vol_chunks.append(chunk)
    
    vol_df = pd.concat(vol_chunks, ignore_index=True)
    print(f"✅ Loaded: {len(vol_df):,} rows")
    
    # Convert dates
    vol_df['date'] = pd.to_datetime(vol_df['date'])
    comp_df['closest_quote_date'] = pd.to_datetime(comp_df['closest_quote_date'])
    
    # Filter to rows with permno
    comp_with_permno = comp_df[comp_df['permno'].notna()].copy()
    print(f"Processing {len(comp_with_permno):,} rows with permno")
    
    # Initialize volatility columns
    vol_columns = ['ret', 'vol_hl5', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'vol_hl63', 'vol_hl126']
    for col in vol_columns:
        if col in vol_df.columns:
            comp_with_permno[col] = np.nan
    
    comp_with_permno['vol_date_match_type'] = 'no_match'
    comp_with_permno['vol_date_diff_days'] = np.nan
    
    print("Joining volatility features...")
    
    # Process in batches
    batch_size = 2000
    total_batches = (len(comp_with_permno) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(comp_with_permno))
        batch = comp_with_permno.iloc[start_idx:end_idx]
        
        for idx in batch.index:
            permno = batch.loc[idx, 'permno']
            closest_quote_date = batch.loc[idx, 'closest_quote_date']
            
            # Get volatility data for this permno
            vol_subset = vol_df[vol_df['permno'] == permno]
            
            if len(vol_subset) > 0:
                # Try exact date match first
                exact_match = vol_subset[vol_subset['date'] == closest_quote_date]
                
                if len(exact_match) > 0:
                    vol_row = exact_match.iloc[0]
                    match_type = 'exact'
                    date_diff = 0
                else:
                    # Find closest date within ±5 days
                    vol_subset['date_diff'] = abs(vol_subset['date'] - closest_quote_date)
                    min_diff_idx = vol_subset['date_diff'].idxmin()
                    vol_row = vol_subset.loc[min_diff_idx]
                    
                    if vol_row['date_diff'].days <= 5:
                        match_type = 'closest'
                        date_diff = vol_row['date_diff'].days
                    else:
                        continue
                
                # Add volatility features
                for col in vol_columns:
                    if col in vol_row.index:
                        comp_with_permno.loc[idx, col] = vol_row[col]
                
                comp_with_permno.loc[idx, 'vol_date_match_type'] = match_type
                comp_with_permno.loc[idx, 'vol_date_diff_days'] = date_diff
        
        # Progress indicator
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Processed batch {batch_idx + 1}/{total_batches}")
    
    # Check results
    vol_coverage = comp_with_permno[vol_columns[0]].notna().sum()
    coverage_pct = (vol_coverage / len(comp_with_permno)) * 100
    
    print(f"✅ Step 2 completed:")
    print(f"  - Rows with volatility data: {vol_coverage:,} ({coverage_pct:.1f}%)")
    
    # Add rows without permno back
    comp_without_permno = comp_df[comp_df['permno'].isna()].copy()
    for col in vol_columns + ['vol_date_match_type', 'vol_date_diff_days']:
        if col in comp_without_permno.columns:
            comp_without_permno[col] = np.nan
    
    final_df = pd.concat([comp_with_permno, comp_without_permno], ignore_index=True)
    
    return final_df

def step3_join_options(comp_df):
    """Step 3: Left join options features based on ticker and surface_date, matching to closest_quote_date ±3 days"""
    print("\n🔄 STEP 3: JOINING OPTIONS FEATURES")
    print("="*50)
    
    # Load options data
    print("Loading options data...")
    options_df = pd.read_excel('data_files/option_features_combined_20250821_222431.xlsx')
    print(f"✅ Loaded: {len(options_df):,} rows")
    
    # Convert dates
    options_df['surface_date'] = pd.to_datetime(options_df['surface_date'])
    comp_df['closest_quote_date'] = pd.to_datetime(comp_df['closest_quote_date'])
    
    # Select only the features we want
    key_features = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
    available_features = [col for col in key_features if col in options_df.columns]
    
    if not available_features:
        print("❌ None of the key options features found!")
        print("Available columns:", list(options_df.columns))
        return comp_df
    
    print(f"Joining options features: {available_features}")
    
    # Initialize options columns
    for feature in available_features:
        comp_df[feature] = np.nan
    
    comp_df['surface_date'] = np.nan
    comp_df['options_date_match_type'] = 'no_match'
    comp_df['options_date_diff_days'] = np.nan
    
    # Filter to rows with ticker and closest_quote_date
    comp_valid = comp_df[comp_df['ticker'].notna() & comp_df['closest_quote_date'].notna()].copy()
    print(f"Processing {len(comp_valid):,} rows with ticker and date")
    
    print("Joining options features...")
    
    # Process in batches
    batch_size = 2000
    total_batches = (len(comp_valid) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(comp_valid))
        batch = comp_valid.iloc[start_idx:end_idx]
        
        for idx in batch.index:
            ticker = batch.loc[idx, 'ticker']
            closest_quote_date = batch.loc[idx, 'closest_quote_date']
            
            # Get options data for this ticker
            ticker_options = options_df[options_df['ticker'] == ticker]
            
            if len(ticker_options) > 0:
                # Try exact date match first
                exact_match = ticker_options[ticker_options['surface_date'] == closest_quote_date]
                
                if len(exact_match) > 0:
                    options_row = exact_match.iloc[0]
                    match_type = 'exact'
                    date_diff = 0
                else:
                    # Find closest date within ±3 days
                    try:
                        # Create a completely fresh copy to avoid any view issues
                        ticker_options_fresh = ticker_options.reset_index(drop=True).copy()
                        
                        # Add date difference column
                        ticker_options_fresh['date_diff'] = abs(ticker_options_fresh['surface_date'] - closest_quote_date)
                        
                        # Debug: Check if we still have data
                        if len(ticker_options_fresh) == 0:
                            print(f"    Debug: ticker_options_fresh became empty for ticker {ticker}")
                            continue
                        
                        # Find minimum date difference
                        if ticker_options_fresh['date_diff'].isna().all():
                            print(f"    Debug: All date_diff values are NaN for ticker {ticker}")
                            continue
                        
                        min_diff_idx = ticker_options_fresh['date_diff'].idxmin()
                        options_row = ticker_options_fresh.loc[min_diff_idx]
                        
                        if options_row['date_diff'].days <= 3:
                            match_type = 'closest'
                            date_diff = options_row['date_diff'].days
                        else:
                            continue
                            
                    except Exception as e:
                        print(f"    Debug: Error processing ticker {ticker}: {e}")
                        print(f"    Debug: ticker_options length: {len(ticker_options)}")
                        print(f"    Debug: ticker_options_fresh length: {len(ticker_options_fresh) if 'ticker_options_fresh' in locals() else 'N/A'}")
                        continue
                
                # Add options features
                for feature in available_features:
                    comp_df.loc[idx, feature] = options_row[feature]
                
                comp_df.loc[idx, 'surface_date'] = options_row['surface_date']
                comp_df.loc[idx, 'options_date_match_type'] = match_type
                comp_df.loc[idx, 'options_date_diff_days'] = date_diff
        
        # Progress indicator
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Processed batch {batch_idx + 1}/{total_batches}")
    
    # Check results
    options_coverage = comp_df[available_features[0]].notna().sum()
    coverage_pct = (options_coverage / len(comp_df)) * 100
    
    print(f"✅ Step 3 completed:")
    print(f"  - Rows with options data: {options_coverage:,} ({coverage_pct:.1f}%)")
    
    return comp_df

def save_results(final_df):
    """Save the final merged dataset"""
    print("\n💾 SAVING RESULTS")
    print("="*50)
    
    output_file = 'data_files/final_merged_dataset.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"✅ Final dataset saved to: {output_file}")
    print(f"Final dataset: {len(final_df):,} rows x {len(final_df.columns)} columns")
    
    # Show column summary
    print(f"\nColumn breakdown:")
    print(f"  - Original columns: {len([col for col in final_df.columns if col not in ['permno', 'company_name', 'cusip', 'ret', 'vol_hl5', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'vol_hl63', 'vol_hl126', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK', 'surface_date', 'vol_date_match_type', 'vol_date_diff_days', 'options_date_match_type', 'options_date_diff_days']])}")
    print(f"  - Mapping columns: {len(['permno', 'company_name', 'cusip'])}")
    print(f"  - Volatility columns: {len(['ret', 'vol_hl5', 'vol_hl7', 'vol_hl10', 'vol_hl21', 'vol_hl63', 'vol_hl126'])}")
    print(f"  - Options columns: {len(['SKEW', 'KURT', 'IV_RATIO', 'SMIRK'])}")
    print(f"  - Metadata columns: {len(['surface_date', 'vol_date_match_type', 'vol_date_diff_days', 'options_date_match_type', 'options_date_diff_days'])}")
    
    return output_file

def main():
    """Main function to execute the 3-step merge process"""
    print("CLEAN 3-STEP MERGE PROCESS")
    print("="*50)
    
    try:
        
        # Check files
        if not check_files():
            print("❌ Exiting due to missing required files.")
            return
        
        # Step 1: Add permno
        comp_with_permno = step1_add_permno()
        
        # Step 2: Join volatility features
        comp_with_vol = step2_join_volatility(comp_with_permno)
        
        # Step 3: Join options features
        final_df = step3_join_options(comp_with_vol)
        
        # Save results
        output_file = save_results(final_df)
        
        print(f"\n🎉 MERGE PROCESS COMPLETED SUCCESSFULLY!")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during merge process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
