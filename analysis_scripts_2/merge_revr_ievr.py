#!/usr/bin/env python3
"""
Merge REVR and IEVR datasets by ticker, earnings_date, and analysis_date
"""

import pandas as pd
import numpy as np
from datetime import datetime

def merge_revr_ievr_datasets():
    """
    Merge REVR and IEVR datasets to create a comprehensive volatility analysis dataset.
    """
    print("🔗 MERGING REVR AND IEVR DATASETS")
    print("="*60)
    
    # File paths
    revr_file = 'data_files/bulk_revr_comprehensive_static_cusip_comparison.csv'
    ievr_file = 'data_files/ievr_batch_20250820_042741.csv'
    
    # Check if both files exist
    if not pd.io.common.file_exists(revr_file):
        print(f"❌ REVR file not found: {revr_file}")
        return
    
    if not pd.io.common.file_exists(ievr_file):
        print(f"❌ IEVR file not found: {ievr_file}")
        return
    
    # Load datasets
    print("📊 Loading datasets...")
    df_revr = pd.read_csv(revr_file)
    df_ievr = pd.read_csv(ievr_file)
    
    print(f"✅ REVR dataset: {len(df_revr):,} observations")
    print(f"✅ IEVR dataset: {len(df_ievr):,} observations")
    
    # Display column information
    print(f"\n📋 REVR columns: {list(df_revr.columns)}")
    print(f"📋 IEVR columns: {list(df_ievr.columns)}")
    
    # Convert date columns to datetime for proper merging
    print(f"\n🔄 Converting date columns...")
    
    # REVR dataset date columns
    df_revr['earnings_date'] = pd.to_datetime(df_revr['earnings_date'])
    df_revr['analysis_date'] = pd.to_datetime(df_revr['analysis_date'])
    
    # IEVR dataset date columns
    df_ievr['earnings_date'] = pd.to_datetime(df_ievr['earnings_date'])
    df_ievr['analysis_date'] = pd.to_datetime(df_ievr['analysis_date'])
    
    print(f"✅ Date columns converted to datetime")
    
    # Check for exact matches on ticker and earnings_date only
    print(f"\n🔍 Checking for exact matches on ticker and earnings_date...")
    
    # Create a composite key for matching (ticker + earnings_date only)
    df_revr['merge_key'] = df_revr['ticker'] + '_' + df_revr['earnings_date'].dt.strftime('%Y-%m-%d')
    df_ievr['merge_key'] = df_ievr['ticker'] + '_' + df_ievr['earnings_date'].dt.strftime('%Y-%m-%d')
    
    # Find matches
    revr_keys = set(df_revr['merge_key'])
    ievr_keys = set(df_ievr['merge_key'])
    
    exact_matches = revr_keys.intersection(ievr_keys)
    only_in_revr = revr_keys - ievr_keys
    only_in_ievr = ievr_keys - revr_keys
    
    print(f"📊 Matching Analysis:")
    print(f"  Exact matches: {len(exact_matches):,}")
    print(f"  Only in REVR: {len(only_in_revr):,}")
    print(f"  Only in IEVR: {len(only_in_ievr):,}")
    
    # Perform the merge
    print(f"\n🔄 Performing merge...")
    
    # Merge on ticker and earnings_date only
    merged_df = pd.merge(
        df_revr,
        df_ievr,
        on=['ticker', 'earnings_date'],
        how='inner',  # Only keep observations that exist in both datasets
        suffixes=('_revr', '_ievr')
    )
    
    print(f"✅ Merge completed: {len(merged_df):,} observations")
    
    # Clean up the merge key columns
    merged_df = merged_df.drop(['merge_key_revr', 'merge_key_ievr'], axis=1, errors='ignore')
    
    # Reorder columns for better readability
    print(f"\n📋 Reordering columns...")
    
    # Define column order
    base_cols = ['ticker', 'earnings_date']
    revr_cols = ['analysis_date_revr', 'season', 'year', 'quarter', 'revr']
    ievr_cols = ['analysis_date_ievr', 'closest_quote_date', 'days_to_earnings', 'ievr', 'avg_pre', 'avg_post', 'skew_ratio']
    
    # Check which columns actually exist
    existing_revr_cols = [col for col in revr_cols if col in merged_df.columns]
    existing_ievr_cols = [col for col in ievr_cols if col in merged_df.columns]
    
    column_order = base_cols + existing_revr_cols + existing_ievr_cols
    
    # Add any remaining columns
    remaining_cols = [col for col in merged_df.columns if col not in column_order]
    column_order = column_order + remaining_cols
    
    merged_df = merged_df[column_order]
    
    # Save merged dataset
    output_file = 'data_files/merged_revr_ievr_comprehensive.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"💾 Merged dataset saved to: {output_file}")
    
    # Summary statistics
    print(f"\n📊 MERGE SUMMARY:")
    print(f"  Total observations: {len(merged_df):,}")
    print(f"  Unique stocks: {merged_df['ticker'].nunique():,}")
    print(f"  Date range: {merged_df['earnings_date'].min()} to {merged_df['earnings_date'].max()}")
    
    # Check for missing values
    print(f"\n🔍 Data Quality Check:")
    missing_revr = merged_df['revr'].isna().sum()
    missing_ievr = merged_df['ievr'].isna().sum()
    print(f"  Missing REVR values: {missing_revr}")
    print(f"  Missing IEVR values: {missing_ievr}")
    
    # Sample of merged data
    print(f"\n📋 Sample of merged data:")
    print(merged_df.head(3).to_string(index=False))
    
    # Save summary statistics
    summary_stats = {
        'total_observations': len(merged_df),
        'unique_stocks': merged_df['ticker'].nunique(),
        'date_range_start': merged_df['earnings_date'].min().strftime('%Y-%m-%d'),
        'date_range_end': merged_df['earnings_date'].max().strftime('%Y-%m-%d'),
        'exact_matches_found': len(exact_matches),
        'only_in_revr': len(only_in_revr),
        'only_in_ievr': len(only_in_ievr),
        'missing_revr': missing_revr,
        'missing_ievr': missing_ievr
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = 'data_files/merge_summary_stats.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary statistics saved to: {summary_file}")
    
    return merged_df

def analyze_merged_dataset(df):
    """
    Analyze the merged dataset for insights.
    """
    print(f"\n🔍 ANALYZING MERGED DATASET")
    print("="*60)
    
    # Basic statistics
    print(f"📊 Basic Statistics:")
    print(f"  REVR - Mean: {df['revr'].mean():.3f}, Std: {df['revr'].std():.3f}")
    print(f"  IEVR - Mean: {df['ievr'].mean():.3f}, Std: {df['ievr'].std():.3f}")
    
    # Correlation between REVR and IEVR
    correlation = df['revr'].corr(df['ievr'])
    print(f"  Correlation (REVR vs IEVR): {correlation:.3f}")
    
    # Season distribution
    if 'season' in df.columns:
        print(f"\n📅 Season Distribution:")
        season_counts = df['season'].value_counts().head(10)
        for season, count in season_counts.items():
            print(f"  {season}: {count} observations")
    
    # Stock coverage
    print(f"\n📈 Stock Coverage:")
    print(f"  Top stocks by observations:")
    stock_counts = df['ticker'].value_counts().head(10)
    for ticker, count in stock_counts.items():
        print(f"    {ticker}: {count} observations")

if __name__ == "__main__":
    try:
        # Merge datasets
        merged_df = merge_revr_ievr_datasets()
        
        if merged_df is not None and len(merged_df) > 0:
            # Analyze merged dataset
            analyze_merged_dataset(merged_df)
            
            print(f"\n🎉 Dataset merge completed successfully!")
            print(f"📁 Output files:")
            print(f"  - Merged dataset: data_files/merged_revr_ievr_comprehensive.csv")
            print(f"  - Summary stats: data_files/merge_summary_stats.csv")
        else:
            print("❌ Merge failed - no data generated")
            
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
