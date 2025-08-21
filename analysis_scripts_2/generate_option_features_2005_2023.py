#!/usr/bin/env python3
"""
Generate Option Surface Features for Configurable Date Range
Generate option surface features for every earnings season within a configurable date range
using the top 500 high volume stocks for each season from existing CSV file
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

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE TO SET YOUR DATE RANGE
# =============================================================================

# Set your desired start and end dates here
START_YEAR = 2005
START_QUARTER = 1  # 1=Q1, 2=Q2, 3=Q3, 4=Q4
END_YEAR = 2020
END_QUARTER = 2    # 1=Q1, 2=Q2, 3=Q3, 4=Q4

# =============================================================================
# END OF CONFIGURABLE PARAMETERS
# =============================================================================

def get_quarter_string(quarter_int):
    """
    Convert quarter integer to string format.
    
    Parameters:
    - quarter_int (int): Quarter number (1, 2, 3, 4)
    
    Returns:
    - str: Quarter string (Q1, Q2, Q3, Q4)
    """
    quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    return quarter_map.get(quarter_int, 'Q1')

def get_quarter_int(quarter_str):
    """
    Convert quarter string to integer.
    
    Parameters:
    - quarter_str (str): Quarter string (Q1, Q2, Q3, Q4)
    
    Returns:
    - int: Quarter number (1, 2, 3, 4)
    """
    quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    return quarter_map.get(quarter_str, 1)

def generate_date_range():
    """
    Generate list of (year, quarter) tuples for the configured date range.
    
    Returns:
    - list: List of (year, quarter) tuples
    """
    date_range = []
    
    # Convert to quarter strings for consistency
    start_quarter_str = get_quarter_string(START_QUARTER)
    end_quarter_str = get_quarter_string(END_QUARTER)
    
    for year in range(START_YEAR, END_YEAR + 1):
        for quarter_int in range(1, 5):
            quarter_str = get_quarter_string(quarter_int)
            
            # Skip quarters before start
            if year == START_YEAR and quarter_int < START_QUARTER:
                continue
                
            # Skip quarters after end
            if year == END_YEAR and quarter_int > END_QUARTER:
                continue
                
            date_range.append((year, quarter_str))
    
    return date_range

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
    Generate option surface features for all earnings seasons within the configured date range.
    
    Parameters:
    - db: WRDS database connection
    
    Returns:
    - list: List of generated file paths
    """
    start_quarter_str = get_quarter_string(START_QUARTER)
    end_quarter_str = get_quarter_string(END_QUARTER)
    
    print(f"🚀 GENERATING OPTION SURFACE FEATURES FOR {start_quarter_str} {START_YEAR} - {end_quarter_str} {END_YEAR}")
    print("="*80)
    print(f"📅 Configured date range: {start_quarter_str} {START_YEAR} to {end_quarter_str} {END_YEAR}")
    
    # Load top 500 stocks data
    top_500_df = load_top_500_stocks_data()
    
    # Generate the date range to process
    date_range = generate_date_range()
    
    print(f"📊 Total seasons to process: {len(date_range)}")
    print(f"📋 Date range: {date_range[0] if date_range else 'None'} to {date_range[-1] if date_range else 'None'}")
    
    generated_files = []
    skipped_files = []
    start_time = time.time()
    
    for idx, (year, quarter) in enumerate(date_range, 1):
        print(f"\n📊 Processing season {idx}/{len(date_range)}: {quarter} {year}")
        
        # Check if file already exists
        output_filename = f'option_features_{year}_{quarter}.csv'
        output_filepath = os.path.join('data_files', output_filename)
        
        if os.path.exists(output_filepath):
            print(f"⏭️ File already exists: {output_filename}")
            print(f"📋 Skipping {quarter} {year} - file already generated")
            skipped_files.append(output_filepath)
            continue
        
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
        remaining_seasons = len(date_range) - idx
        estimated_remaining_time = remaining_seasons * avg_time_per_season
        
        print(f"⏱️ Progress: {idx}/{len(date_range)} seasons completed")
        print(f"⏱️ Elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"⏱️ Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")
    
    total_time = time.time() - start_time
    print(f"\n🎉 COMPLETED! Total time: {total_time/3600:.2f} hours")
    print(f"📁 Generated {len(generated_files)} new files")
    print(f"⏭️ Skipped {len(skipped_files)} existing files")
    
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
        
        # Save combined file with configurable date range
        start_quarter_str = get_quarter_string(START_QUARTER)
        end_quarter_str = get_quarter_string(END_QUARTER)
        combined_filename = f'option_features_{START_YEAR}{start_quarter_str}_{END_YEAR}{end_quarter_str}_combined.csv'
        combined_filepath = os.path.join(output_dir, combined_filename)
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
    Main function to generate option surface features for configurable date range.
    """
    start_quarter_str = get_quarter_string(START_QUARTER)
    end_quarter_str = get_quarter_string(END_QUARTER)
    print(f"🚀 OPTION SURFACE FEATURES GENERATION {start_quarter_str} {START_YEAR} - {end_quarter_str} {END_YEAR}")
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
