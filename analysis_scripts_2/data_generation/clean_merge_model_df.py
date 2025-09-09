#!/usr/bin/env python3
"""
Clean merge script to update model_df.csv with dispersion_pct_ibes and z_score_momentum
from eps_features_at_analysis_dates.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def main():
    print("=== CLEAN MERGE: MODEL_DF UPDATE ===")
    print("Loading datasets...")
    
    # Load the datasets
    model_df = pd.read_csv('data_files/model_df.csv')
    eps_df = pd.read_csv('data_files/eps_features_at_analysis_dates.csv')
    
    print(f"Model_df shape: {model_df.shape}")
    print(f"EPS_df shape: {eps_df.shape}")
    
    # Convert date columns to datetime for proper comparison
    model_df['earnings_date'] = pd.to_datetime(model_df['earnings_date'])
    eps_df['earnings_date'] = pd.to_datetime(eps_df['earnings_date'])
    
    # Step 1: Drop the columns we want to replace
    columns_to_drop = [
        'momentum_1m', 'momentum_3m', 'momentum_6m', 
        'rolling_momentum_3m', 'z_score_momentum', 
        'VIX_21d_diff', 'dispersion_pct_ibes'
    ]
    
    print(f"\nStep 1: Dropping existing columns...")
    print(f"Columns to drop: {columns_to_drop}")
    
    # Check which columns actually exist
    existing_columns = [col for col in columns_to_drop if col in model_df.columns]
    print(f"Existing columns to drop: {existing_columns}")
    
    # Drop the columns
    model_df_clean = model_df.drop(columns=existing_columns)
    print(f"Model_df shape after dropping: {model_df_clean.shape}")
    
    # Step 2: Select only the columns we want from eps_df
    eps_columns_to_keep = ['ticker', 'earnings_date', 'dispersion_pct_ibes', 'z_score_momentum']
    eps_subset = eps_df[eps_columns_to_keep].copy()
    
    print(f"\nStep 2: Preparing EPS features...")
    print(f"EPS columns to keep: {eps_columns_to_keep}")
    print(f"EPS subset shape: {eps_subset.shape}")
    
    # Step 3: Left join on ticker and earnings_date
    print(f"\nStep 3: Performing left join...")
    
    merged_df = model_df_clean.merge(
        eps_subset, 
        on=['ticker', 'earnings_date'], 
        how='left'
    )
    
    print(f"Merged shape: {merged_df.shape}")
    
    # Step 4: Check merge results
    print(f"\nStep 4: Checking merge results...")
    
    # Count matches
    total_model_records = len(model_df_clean)
    matched_records = merged_df['dispersion_pct_ibes'].notna().sum()
    z_score_matched = merged_df['z_score_momentum'].notna().sum()
    
    print(f"Total model_df records: {total_model_records}")
    print(f"Records with dispersion_pct_ibes: {matched_records} ({matched_records/total_model_records:.1%})")
    print(f"Records with z_score_momentum: {z_score_matched} ({z_score_matched/total_model_records:.1%})")
    
    # Check sample records
    print(f"\nStep 5: Sample verification...")
    sample_tickers = ['EL', 'DHR', 'AFL', 'MSFT']
    
    for ticker in sample_tickers:
        ticker_records = merged_df[merged_df['ticker'] == ticker].head(2)
        if len(ticker_records) > 0:
            print(f"\n{ticker} sample:")
            for _, row in ticker_records.iterrows():
                z_score_str = f"{row['z_score_momentum']:.6f}" if pd.notna(row['z_score_momentum']) else 'NaN'
                print(f"  {row['earnings_date'].strftime('%Y-%m-%d')}: "
                      f"dispersion={row['dispersion_pct_ibes']:.6f}, "
                      f"z_score={z_score_str}")
    
    # Step 6: Save the updated dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'data_files/model_df_backup_{timestamp}.csv'
    output_file = f'data_files/model_df_clean_updated_{timestamp}.csv'
    
    print(f"\nStep 6: Saving updated dataset...")
    print(f"  Backup: {backup_file}")
    print(f"  Updated: {output_file}")
    
    # Create backup of original
    model_df.to_csv(backup_file, index=False)
    
    # Save updated dataset
    merged_df.to_csv(output_file, index=False)
    
    # Also update the main model_df.csv file
    merged_df.to_csv('data_files/model_df.csv', index=False)
    
    print(f"\n✅ Clean merge complete!")
    print(f"Original shape: {model_df.shape}")
    print(f"Final shape: {merged_df.shape}")
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  dispersion_pct_ibes: {merged_df['dispersion_pct_ibes'].notna().sum()} non-null values")
    print(f"  z_score_momentum: {merged_df['z_score_momentum'].notna().sum()} non-null values")
    
    # Check data quality
    if 'dispersion_pct_ibes' in merged_df.columns:
        print(f"  dispersion_pct_ibes range: [{merged_df['dispersion_pct_ibes'].min():.4f}, {merged_df['dispersion_pct_ibes'].max():.4f}]")
    if 'z_score_momentum' in merged_df.columns:
        print(f"  z_score_momentum range: [{merged_df['z_score_momentum'].min():.4f}, {merged_df['z_score_momentum'].max():.4f}]")

if __name__ == "__main__":
    main()
