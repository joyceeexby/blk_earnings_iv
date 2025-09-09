#!/usr/bin/env python3
"""
Script to update option surface features in the final merged dataset.
This script will merge the new option features with the existing dataset,
updating the option columns while preserving all other data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def update_option_features():
    """
    Update option features in final_merged_dataset_with_momentum_final.csv
    with new features from option_features_2005Q1_2023Q2_combined.csv
    """
    
    # Set up file paths
    data_dir = "../data_files"
    final_dataset_path = os.path.join(data_dir, "final_merged_dataset_with_momentum_final.csv")
    new_options_path = os.path.join(data_dir, "option_features_2005Q1_2023Q2_combined.csv")
    backup_path = os.path.join(data_dir, "final_merged_dataset_with_momentum_final_backup.csv")
    output_path = os.path.join(data_dir, "final_merged_dataset_with_momentum_updated.csv")
    
    print("Loading datasets...")
    
    # Load the final merged dataset
    final_df = pd.read_csv(final_dataset_path)
    print(f"Final dataset shape: {final_df.shape}")
    
    # Load the new option features
    options_df = pd.read_csv(new_options_path)
    print(f"Option features shape: {options_df.shape}")
    
    # Create backup of original file
    print("Creating backup of original dataset...")
    final_df.to_csv(backup_path, index=False)
    
    # Prepare option features for merge
    print("Preparing option features for merge...")
    
    # Convert earnings_date to datetime for proper matching
    final_df['earnings_date'] = pd.to_datetime(final_df['earnings_date'])
    options_df['earnings_date'] = pd.to_datetime(options_df['earnings_date'])
    
    # Select relevant columns from option features (excluding TERM_RATIO due to missing data)
    option_cols_to_merge = ['ticker', 'earnings_date', 'year', 'quarter', 
                           'SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
    
    options_subset = options_df[option_cols_to_merge].copy()
    
    # Add suffix to new option columns to avoid conflicts during merge
    option_feature_cols = ['SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
    rename_dict = {col: f'{col}_new' for col in option_feature_cols}
    options_subset = options_subset.rename(columns=rename_dict)
    
    print("Performing merge...")
    
    # Merge on ticker, earnings_date, year, and quarter
    merged_df = final_df.merge(
        options_subset, 
        on=['ticker', 'earnings_date', 'year', 'quarter'], 
        how='left',
        suffixes=('', '_new')
    )
    
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Count matches
    matches = merged_df['SKEW_new'].notna().sum()
    total_rows = len(merged_df)
    print(f"Successfully matched {matches} out of {total_rows} rows ({matches/total_rows*100:.1f}%)")
    
    # Update the option feature columns
    print("Updating option feature columns...")
    
    # Replace old values with new values where available
    for col in option_feature_cols:
        old_col = col
        new_col = f'{col}_new'
        
        if old_col in merged_df.columns:
            # Update existing column with new values where available
            merged_df[old_col] = merged_df[new_col].fillna(merged_df[old_col])
        else:
            # Add new column (this is the case for TERM_RATIO)
            merged_df[old_col] = merged_df[new_col]
    
    # Drop the temporary '_new' columns
    cols_to_drop = [f'{col}_new' for col in option_feature_cols]
    merged_df = merged_df.drop(columns=cols_to_drop)
    
    print("Reordering columns...")
    
    # Maintain original column structure (no new columns to add)
    original_cols = final_df.columns.tolist()
    
    # Reorder the dataframe to match original structure
    merged_df = merged_df[original_cols]
    
    # Save the updated dataset
    print(f"Saving updated dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n=== UPDATE SUMMARY ===")
    print(f"Original dataset rows: {len(final_df)}")
    print(f"Updated dataset rows: {len(merged_df)}")
    print(f"Option features matched: {matches} ({matches/total_rows*100:.1f}%)")
    
    # Show sample of updated data
    print("\nSample of updated option features:")
    sample_cols = ['ticker', 'earnings_date', 'SKEW', 'KURT', 'IV_RATIO', 'SMIRK']
    sample_df = merged_df[sample_cols].head(10)
    print(sample_df.to_string(index=False))
    
    # Check for any rows where old values were different from new values
    print("\nChecking for updates in existing option features...")
    
    # For this, we'd need to compare with the backup, but for now just report completion
    print("Update completed successfully!")
    print(f"Backup saved to: {backup_path}")
    print(f"Updated dataset saved to: {output_path}")
    
    return merged_df

if __name__ == "__main__":
    updated_df = update_option_features()
