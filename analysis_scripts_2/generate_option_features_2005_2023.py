#!/usr/bin/env python3
"""
Generate Option Surface Features for 2007-2023
Generate option surface features for every earnings season from 2007-2023
using the top 500 high volume stocks for each season from existing CSV file
Modified to start from 2007 (skipping 2005-2006)
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from option_surface_features import compute_option_surface_features

def load_top_500_stocks_data():
    """
    Load the existing top 500 stocks data from CSV file.
    
    Returns:
    - pandas.DataFrame: DataFrame with top 500 stocks per season
    """
    file_path = 'data_files/top500_liquidity_2005_2023.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Top 500 stocks file not found: {file_path}")
    
    print(f"📊 Loading top 500 stocks data from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert quarter to string format (Q1, Q2, Q3, Q4)
    df['quarter'] = df['quarter'].astype(str).map({
        '1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Q4'
    })
    
    print(f"✅ Loaded {len(df)} records")
    print(f"📅 Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"📊 Unique tickers: {df['ticker'].nunique()}")
    
    return df

def get_top_500_stocks_for_season(year, quarter, top_500_df):
    """
    Get the top 500 high volume stocks for a specific earnings season.
    Uses existing data from the CSV file.
    
    Parameters:
    - year (int): Year
    - quarter (str): Quarter (Q1, Q2, Q3, Q4)
    - top_500_df (pandas.DataFrame): DataFrame with top 500 stocks data
    
    Returns:
    - list: List of ticker symbols
    """
    print(f"📊 Getting top 500 stocks for {quarter} {year}...")
    
    # Filter data for the specific year and quarter
    season_data = top_500_df[
        (top_500_df['year'] == year) & 
        (top_500_df['quarter'] == quarter)
    ]
    
    if season_data.empty:
        print(f"❌ No data found for {quarter} {year}")
        return []
    
    # Get unique tickers for this season
    tickers = season_data['ticker'].unique().tolist()
    
    print(f"✅ Found {len(tickers)} stocks for {quarter} {year}")
    return tickers



def generate_option_features_for_season(year, quarter, db, top_500_df, output_dir='data_files'):
    """
    Generate option surface features for a specific earnings season.
    
    Parameters:
    - year (int): Year
    - quarter (str): Quarter
    - db: WRDS database connection
    - top_500_df (pandas.DataFrame): DataFrame with top 500 stocks data
    - output_dir (str): Output directory
    
    Returns:
    - str: Path to output file
    """
    print(f"\n🎯 GENERATING OPTION FEATURES FOR {quarter} {year}")
    print("="*80)
    
    # Get top 500 stocks for this season
    top_stocks = get_top_500_stocks_for_season(year, quarter, top_500_df)
    
    if not top_stocks:
        print(f"❌ No stocks found for {quarter} {year}")
        return None
    
    print(f"📈 Processing {len(top_stocks)} stocks")
    
    # Generate output filename
    output_filename = f'option_features_{year}_{quarter}.csv'
    output_filepath = os.path.join(output_dir, output_filename)
    
    # Check if file already exists
    if os.path.exists(output_filepath):
        print(f"📋 File already exists: {output_filename}")
        print(f"📊 Loading existing results...")
        try:
            existing_df = pd.read_csv(output_filepath)
            print(f"✅ Loaded {len(existing_df)} existing records")
            return output_filepath
        except Exception as e:
            print(f"⚠️ Error loading existing file: {e}")
    
    # Compute option surface features
    print(f"🔄 Computing option surface features...")
    
    try:
        # Use the existing compute_option_surface_features function
        # Convert quarter string to integer (Q1 -> 1, Q2 -> 2, etc.)
        quarter_int = int(quarter.replace('Q', ''))
        
        results_df = compute_option_surface_features(
            ticker_list=top_stocks,
            earnings_year=year,
            earnings_quarter=quarter_int,
            db=db,
            n_lag=20  # Trading days before earnings
        )
        
        if results_df is not None and not results_df.empty:
            # Add metadata
            results_df['year'] = year
            results_df['quarter'] = quarter
            results_df['computation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to CSV
            results_df.to_csv(output_filepath, index=False)
            print(f"✅ Generated {len(results_df)} records")
            print(f"💾 Saved to: {output_filepath}")
            
            return output_filepath
        else:
            print(f"❌ No results generated for {quarter} {year}")
            return None
            
    except Exception as e:
        print(f"❌ Error computing features for {quarter} {year}: {e}")
        return None

def generate_all_option_features_2005_2023(db):
    """
    Generate option surface features for all earnings seasons from 2007-2023.
    Modified to start from 2007 (skipping 2005-2006).
    
    Parameters:
    - db: WRDS database connection
    
    Returns:
    - list: List of generated file paths
    """
    print("🚀 GENERATING OPTION SURFACE FEATURES FOR 2007-2023")
    print("="*80)
    
    # Load top 500 stocks data
    top_500_df = load_top_500_stocks_data()
    
    # Get unique seasons from the data, but filter out 2005-2006
    all_seasons = top_500_df[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])
    
    # Filter to start from 2007
    seasons = all_seasons[all_seasons['year'] >= 2007].copy()
    
    print(f"📊 Found {len(all_seasons)} total seasons, processing {len(seasons)} seasons from 2007-2023")
    print(f"⏭️ Skipping 2005-2006 seasons (already completed)")
    
    generated_files = []
    start_time = time.time()
    
    for idx, (_, season) in enumerate(seasons.iterrows(), 1):
        year = season['year']
        quarter = season['quarter']
        
        print(f"\n📊 Processing season {idx}/{len(seasons)}: {quarter} {year}")
        
        # Generate features for this season
        result_file = generate_option_features_for_season(
            year=year,
            quarter=quarter,
            db=db,
            top_500_df=top_500_df
        )
        
        if result_file:
            generated_files.append(result_file)
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_season = elapsed_time / idx
        remaining_seasons = len(seasons) - idx
        estimated_remaining_time = remaining_seasons * avg_time_per_season
        
        print(f"⏱️ Progress: {idx}/{len(seasons)} seasons completed")
        print(f"⏱️ Elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"⏱️ Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")
    
    total_time = time.time() - start_time
    print(f"\n🎉 COMPLETED! Total time: {total_time/3600:.2f} hours")
    print(f"📁 Generated {len(generated_files)} files")
    
    return generated_files

def create_summary_report(generated_files, output_dir='data_files'):
    """
    Create a summary report of all generated option surface features.
    
    Parameters:
    - generated_files (list): List of generated file paths
    - output_dir (str): Output directory
    """
    print(f"\n📊 Creating summary report...")
    
    all_data = []
    
    for file_path in generated_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                # Add file info
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined file
        combined_filepath = os.path.join(output_dir, 'option_features_2007_2023_combined.csv')
        combined_df.to_csv(combined_filepath, index=False)
        
        print(f"✅ Combined data saved to: {combined_filepath}")
        print(f"📊 Total records: {len(combined_df)}")
        
                 # Create summary statistics
         summary_stats = {
             'total_records': len(combined_df),
             'unique_tickers': combined_df['ticker'].nunique(),
             'year_range': f"{combined_df['year'].min()}-{combined_df['year'].max()}",
             'quarters': sorted(combined_df['quarter'].unique()),
             'feature_columns': [col for col in combined_df.columns if col not in 
                               ['ticker', 'year', 'quarter', 'computation_date', 'source_file']]
         }
        
        # Save summary
        summary_filepath = os.path.join(output_dir, 'option_features_summary.json')
        import json
        with open(summary_filepath, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"📋 Summary saved to: {summary_filepath}")
        
        return combined_filepath
    else:
        print("❌ No data to combine")
        return None

def main():
    """
    Main function to generate option surface features for 2007-2023.
    """
    print("🚀 OPTION SURFACE FEATURES GENERATION 2007-2023")
    print("="*80)
    
    # You would need to set up your WRDS connection here
    # import wrds
    # db = wrds.Connection()
    
    # For demonstration, we'll show the structure
    print("📋 This script will:")
    print("  1. Load existing top 500 stocks data from CSV file")
    print("  2. Get all earnings seasons from the data")
    print("  3. For each season, compute option surface features for top 500 stocks")
    print("  4. Save results to CSV files")
    print("  5. Create a combined summary report")
    
    # Load and show data structure
    try:
        top_500_df = load_top_500_stocks_data()
        seasons = top_500_df[['year', 'quarter']].drop_duplicates().sort_values(['year', 'quarter'])
        print(f"\n📅 Total seasons in data: {len(seasons)}")
        print(f"📈 Target: Top 500 stocks per season (from existing data)")
        print(f"🎯 Features: TERM_RATIO, SKEW, KURT, IV_RATIO, SMIRK")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    
    print(f"\n⚠️  To run this script:")
    print("  1. Set up your WRDS connection")
    print("  2. Uncomment the database connection code")
    print("  3. Run: python generate_option_features_2005_2023.py")
    
    # Example usage (commented out)
    # db = wrds.Connection()
    # generated_files = generate_all_option_features_2005_2023(db)
    # create_summary_report(generated_files)

if __name__ == "__main__":
    main()
