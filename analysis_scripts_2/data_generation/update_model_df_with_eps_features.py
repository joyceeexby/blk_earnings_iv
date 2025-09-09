#!/usr/bin/env python3
"""
Script to update model_df.csv with latest dispersion_pct_ibes and z_score_momentum 
from eps_features_at_analysis_dates.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def main():
    print("Loading datasets...")
    
    # Load the datasets
    model_df = pd.read_csv('data_files/model_df.csv')
    eps_df = pd.read_csv('data_files/eps_features_at_analysis_dates.csv')
    
    print(f"Model_df shape: {model_df.shape}")
    print(f"EPS_df shape: {eps_df.shape}")
    
    # Convert date columns to datetime for proper comparison
    model_df['earnings_date'] = pd.to_datetime(model_df['earnings_date'])
    model_df['analysis_date_ievr'] = pd.to_datetime(model_df['analysis_date_ievr'])
    eps_df['earnings_date'] = pd.to_datetime(eps_df['earnings_date'])
    eps_df['analysis_date'] = pd.to_datetime(eps_df['analysis_date'])
    
    # Create merge keys
    model_df['merge_key'] = model_df['ticker'] + '_' + model_df['earnings_date'].dt.strftime('%Y-%m-%d')
    eps_df['merge_key'] = eps_df['ticker'] + '_' + eps_df['earnings_date'].dt.strftime('%Y-%m-%d')
    
    print(f"\nUnique merge keys:")
    print(f"Model_df: {model_df['merge_key'].nunique()}")
    print(f"EPS_df: {eps_df['merge_key'].nunique()}")
    
    # Find common keys
    model_keys = set(model_df['merge_key'])
    eps_keys = set(eps_df['merge_key'])
    common_keys = model_keys & eps_keys
    
    print(f"Common keys: {len(common_keys)}")
    print(f"Match rate: {len(common_keys) / len(model_keys):.2%}")
    
    # Check analysis_date matching for common records
    print("\nChecking analysis_date matching for common records...")
    analysis_date_matches = 0
    total_common = 0
    
    for key in list(common_keys)[:10]:  # Check first 10 for sample
        model_row = model_df[model_df['merge_key'] == key]
        eps_row = eps_df[eps_df['merge_key'] == key]
        
        if len(model_row) > 0 and len(eps_row) > 0:
            total_common += 1
            model_analysis_date = model_row['analysis_date_ievr'].iloc[0]
            eps_analysis_date = eps_row['analysis_date'].iloc[0]
            
            if model_analysis_date == eps_analysis_date:
                analysis_date_matches += 1
            else:
                print(f"  Mismatch for {key}:")
                print(f"    Model: {model_analysis_date}")
                print(f"    EPS: {eps_analysis_date}")
    
    print(f"Analysis date match rate (sample): {analysis_date_matches}/{total_common}")
    
    # Perform the merge
    print("\nPerforming merge...")
    
    # Select only the columns we want to update from eps_df
    eps_update_cols = ['merge_key', 'dispersion_pct_ibes', 'z_score_momentum']
    
    eps_update = eps_df[eps_update_cols].copy()
    
    # Merge on merge_key
    merged_df = model_df.merge(eps_update, on='merge_key', how='left', suffixes=('', '_new'))
    
    print(f"Merged shape: {merged_df.shape}")
    
    # Check how many records got updated
    print("\nUpdate statistics:")
    for col in ['dispersion_pct_ibes', 'z_score_momentum', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'rolling_momentum_3m']:
        if col in merged_df.columns:
            new_col = col + '_new'
            if new_col in merged_df.columns:
                updated_count = merged_df[new_col].notna().sum()
                print(f"  {col}: {updated_count} records updated")
                
                # Replace old values with new ones where available
                merged_df[col] = merged_df[new_col].fillna(merged_df[col])
                merged_df = merged_df.drop(columns=[new_col])
    
    # Clean up merge_key column
    merged_df = merged_df.drop(columns=['merge_key'])
    
    # Save the updated dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data_files/model_df_updated_{timestamp}.csv'
    backup_file = f'data_files/model_df_backup_{timestamp}.csv'
    
    print(f"\nSaving updated dataset...")
    print(f"  Backup: {backup_file}")
    print(f"  Updated: {output_file}")
    
    # Create backup of original
    model_df.to_csv(backup_file, index=False)
    
    # Save updated dataset
    merged_df.to_csv(output_file, index=False)
    
    # Also update the main model_df.csv file
    merged_df.to_csv('data_files/model_df.csv', index=False)
    
    print(f"\nUpdate complete!")
    print(f"Original shape: {model_df.shape}")
    print(f"Updated shape: {merged_df.shape}")
    
    # Summary of changes
    print(f"\nSummary of changes:")
    for col in ['dispersion_pct_ibes', 'z_score_momentum', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'rolling_momentum_3m']:
        if col in merged_df.columns:
            non_null_count = merged_df[col].notna().sum()
            print(f"  {col}: {non_null_count} non-null values")

if __name__ == "__main__":
    main()

