#!/usr/bin/env python3
"""
Clean remerge of dispersion_pct_ibes and z_score_momentum from eps_features_at_analysis_dates(1).csv
into model_df.csv after dropping existing columns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def main():
    print("🔄 CLEAN REMERGE: EPS FEATURES UPDATE")
    print("="*60)
    print("Dropping existing columns and remerging from eps_features_at_analysis_dates(1).csv")
    print("="*60)
    
    # Load the datasets
    print("📊 Loading datasets...")
    
    # Load model_df.csv
    model_df = pd.read_csv('data_files/model_df.csv')
    print(f"✅ Loaded model_df: {model_df.shape}")
    
    # Load eps_features_at_analysis_dates(1).csv
    eps_df = pd.read_csv('data_files/eps_features_at_analysis_dates (1).csv')
    print(f"✅ Loaded eps_features: {eps_df.shape}")
    
    # Convert date columns to datetime
    model_df['earnings_date'] = pd.to_datetime(model_df['earnings_date'])
    eps_df['earnings_date'] = pd.to_datetime(eps_df['earnings_date'])
    
    # Step 1: Drop existing dispersion and momentum columns
    print(f"\n🗑️ STEP 1: DROPPING EXISTING COLUMNS")
    print("-" * 50)
    
    columns_to_drop = ['dispersion_pct_ibes', 'z_score_momentum']
    existing_columns = [col for col in columns_to_drop if col in model_df.columns]
    
    if existing_columns:
        print(f"Dropping columns: {existing_columns}")
        model_df_clean = model_df.drop(columns=existing_columns)
        print(f"Model_df shape after dropping: {model_df_clean.shape}")
    else:
        print("No existing columns to drop")
        model_df_clean = model_df.copy()
    
    # Step 2: Prepare eps_features data
    print(f"\n📊 STEP 2: PREPARING EPS FEATURES")
    print("-" * 50)
    
    # Select only the columns we want from eps_df
    eps_columns_to_keep = ['ticker', 'earnings_date', 'dispersion_pct_ibes', 'z_score_momentum']
    eps_subset = eps_df[eps_columns_to_keep].copy()
    
    print(f"EPS columns to keep: {eps_columns_to_keep}")
    print(f"EPS subset shape: {eps_subset.shape}")
    
    # Check coverage in eps_features
    print(f"\nEPS features coverage:")
    for feature in ['dispersion_pct_ibes', 'z_score_momentum']:
        valid_count = eps_subset[feature].notna().sum()
        total_count = len(eps_subset)
        coverage = 100.0 * valid_count / total_count
        print(f"  {feature:20s}: {valid_count:6,} ({coverage:5.1f}% coverage)")
    
    # Step 3: Left join on ticker and earnings_date
    print(f"\n🔗 STEP 3: PERFORMING LEFT JOIN")
    print("-" * 50)
    
    merged_df = model_df_clean.merge(
        eps_subset, 
        on=['ticker', 'earnings_date'], 
        how='left'
    )
    
    print(f"Merged shape: {merged_df.shape}")
    
    # Step 4: Check merge results
    print(f"\n📊 STEP 4: CHECKING MERGE RESULTS")
    print("-" * 50)
    
    # Count matches
    total_model_records = len(model_df_clean)
    dispersion_matched = merged_df['dispersion_pct_ibes'].notna().sum()
    z_score_matched = merged_df['z_score_momentum'].notna().sum()
    
    print(f"Total model_df records: {total_model_records}")
    print(f"Records with dispersion_pct_ibes: {dispersion_matched} ({dispersion_matched/total_model_records:.1%})")
    print(f"Records with z_score_momentum: {z_score_matched} ({z_score_matched/total_model_records:.1%})")
    
    # Check sample records
    print(f"\n🔍 SAMPLE VERIFICATION")
    print("-" * 30)
    sample_tickers = ['EL', 'DHR', 'AFL', 'MSFT']
    
    for ticker in sample_tickers:
        ticker_records = merged_df[merged_df['ticker'] == ticker].head(2)
        if len(ticker_records) > 0:
            print(f"\n{ticker} sample:")
            for _, row in ticker_records.iterrows():
                dispersion_val = f"{row['dispersion_pct_ibes']:.6f}" if pd.notna(row['dispersion_pct_ibes']) else 'NaN'
                z_score_val = f"{row['z_score_momentum']:.6f}" if pd.notna(row['z_score_momentum']) else 'NaN'
                print(f"  {row['earnings_date'].strftime('%Y-%m-%d')}: "
                      f"dispersion={dispersion_val}, z_score={z_score_val}")
    
    # Step 5: Save the updated dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'data_files/model_df_backup_{timestamp}.csv'
    output_file = f'data_files/model_df_remereged_{timestamp}.csv'
    
    print(f"\n💾 STEP 5: SAVING UPDATED DATASET")
    print("-" * 50)
    print(f"  Backup: {backup_file}")
    print(f"  Updated: {output_file}")
    
    # Create backup of original
    model_df.to_csv(backup_file, index=False)
    
    # Save updated dataset
    merged_df.to_csv(output_file, index=False)
    
    # Also update the main model_df.csv file
    merged_df.to_csv('data_files/model_df.csv', index=False)
    
    print(f"\n✅ CLEAN REMERGE COMPLETED!")
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
    
    print(f"\n🎉 SUCCESS! model_df.csv has been updated with fresh data from eps_features_at_analysis_dates(1).csv")

if __name__ == "__main__":
    main()

