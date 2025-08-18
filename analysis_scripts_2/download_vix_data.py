#!/usr/bin/env python3
"""
Standalone VIX Data Downloader

This script downloads VIX data for existing earnings events without running the full analysis.
It adds VIX level and VIX momentum features to the existing results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import wrds
import warnings
warnings.filterwarnings('ignore')

class VIXDataDownloader:
    """
    Download VIX data for earnings events and add VIX-based features.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def get_vix_data(self, start_date, end_date):
        """
        Fetch VIX data from WRDS.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame or None
            VIX data with columns ['date', 'vix'] or None if not found
        """
        print(f"Fetching VIX data from {start_date} to {end_date}")
        try:
            # Try CBOE VIX data from main cboe table
            vix_queries = [
                # Option 1: Try main cboe table (we know this exists)
                f"""
                SELECT date, vix
                FROM cboe.cboe
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                """,
                
                # Option 2: Check what columns are available in main cboe table
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'cboe' AND table_name = 'cboe'
                ORDER BY ordinal_position
                """
            ]
            
            # Try to get VIX data from CBOE
            vix_data = None
            
            for i, query in enumerate(vix_queries, 1):
                try:
                    if i == 1:  # First query is the actual VIX data
                        print(f"  Fetching VIX data from main CBOE table...")
                        vix_data = self.db.raw_sql(query)
                        
                        if not vix_data.empty:
                            print(f"  ✓ VIX data found in main CBOE table")
                            break
                        else:
                            print(f"  ✗ No VIX data in main CBOE table for date range")
                    else:  # Second query is schema exploration
                        print(f"  Checking main CBOE table schema...")
                        result = self.db.raw_sql(query)
                        
                        if not result.empty:
                            print(f"  ✓ Main CBOE table schema found:")
                            print(result.to_string())
                        else:
                            print(f"  ✗ No schema info available")
                            
                except Exception as e:
                    print(f"  ✗ CBOE query {i} failed: {str(e)[:50]}...")
                    continue
            
            if vix_data is None or vix_data.empty:
                print("No VIX data found in any WRDS source")
                print("Available options:")
                print("1. Check WRDS documentation for VIX table names")
                print("2. Use manual CSV download from Yahoo Finance")
                print("3. Contact WRDS support for VIX data access")
                return None
                
            print(f"✓ Retrieved {len(vix_data)} VIX data points")
            print(f"Date range: {vix_data['date'].min()} to {vix_data['date'].max()}")
            print(f"VIX range: {vix_data['vix'].min():.2f} to {vix_data['vix'].max():.2f}")
            
            return vix_data
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return None
    

    
    def calculate_vix_features(self, earnings_date, vix_data):
        """
        Calculate VIX features for a given earnings date using only data available at analysis date.
        NO LOOK-AHEAD BIAS: Only use VIX data available 30 days before earnings.
        
        Parameters:
        -----------
        earnings_date : datetime
            Earnings announcement date
        vix_data : pd.DataFrame
            VIX daily data
            
        Returns:
        --------
        dict : VIX features
        """
        # Ensure VIX data date column is datetime
        if 'date' in vix_data.columns:
            vix_data = vix_data.copy()
            vix_data['date'] = pd.to_datetime(vix_data['date'])
        
        # Calculate analysis date (30 days before earnings) - same as IEVR
        analysis_date = earnings_date - timedelta(days=30)
        
        # Find VIX values at different dates (ONLY UP TO ANALYSIS DATE)
        vix_features = {}
        
        # VIX at analysis date (30 days before earnings) - current VIX level
        analysis_vix = vix_data[vix_data['date'].dt.date == analysis_date.date()]
        if not analysis_vix.empty:
            vix_features['vix_analysis'] = analysis_vix.iloc[0]['vix']
        else:
            vix_features['vix_analysis'] = np.nan
        
        # VIX 5 days before analysis date (35 days before earnings) - for momentum
        analysis_minus_5 = analysis_date - timedelta(days=5)
        vix_minus_5 = vix_data[vix_data['date'].dt.date == analysis_minus_5.date()]
        if not vix_minus_5.empty:
            vix_features['vix_analysis_minus_5'] = vix_minus_5.iloc[0]['vix']
        else:
            vix_features['vix_analysis_minus_5'] = np.nan
        
        # Calculate VIX momentum (current VIX / 5 days ago VIX) - NO LOOK-AHEAD BIAS
        vix_analysis = vix_features['vix_analysis']  # Current VIX at analysis date
        vix_minus_5 = vix_features['vix_analysis_minus_5']  # VIX 5 days before analysis
        
        if (pd.notna(vix_analysis) and pd.notna(vix_minus_5) and vix_minus_5 > 0):
            vix_features['vix_momentum_5d'] = vix_analysis / vix_minus_5  # Current/5d_ago ratio
        else:
            vix_features['vix_momentum_5d'] = np.nan
        
        # Calculate VIX change (current VIX - 5 days ago VIX)
        if (pd.notna(vix_analysis) and pd.notna(vix_minus_5)):
            vix_features['vix_change_5d'] = vix_analysis - vix_minus_5
        else:
            vix_features['vix_change_5d'] = np.nan
        
        # VIX regime classification (based on analysis date VIX)
        if pd.notna(vix_analysis):
            vix_level = vix_analysis
            if vix_level < 15:
                vix_features['vix_regime'] = 'Low_Stress'
            elif vix_level < 25:
                vix_features['vix_regime'] = 'Normal'
            elif vix_level < 35:
                vix_features['vix_regime'] = 'High_Stress'
            else:
                vix_features['vix_regime'] = 'Crisis'
        else:
            vix_features['vix_regime'] = 'Unknown'
        
        return vix_features
    
    def add_vix_to_results(self, results_file='data_files/expanded_earnings_analysis_results.csv'):
        """
        Add VIX features to existing results file.
        
        Parameters:
        -----------
        results_file : str
            Path to existing results CSV file
        """
        print("="*80)
        print("ADDING VIX FEATURES TO EXISTING RESULTS")
        print("="*80)
        
        # Load existing results
        try:
            results_df = pd.read_csv(results_file)
            print(f"✓ Loaded {len(results_df)} existing results from {results_file}")
        except FileNotFoundError:
            print(f"✗ Results file not found: {results_file}")
            return None
        
        # Convert earnings_date to datetime
        results_df['earnings_date'] = pd.to_datetime(results_df['earnings_date'])
        
        # Determine date range for VIX data
        earliest_date = results_df['earnings_date'].min() - timedelta(days=40)  # 35 days before earliest earnings
        latest_date = results_df['earnings_date'].max() + timedelta(days=5)     # 5 days after latest earnings
        
        print(f"VIX data range needed: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # Fetch VIX data
        vix_data = self.get_vix_data(earliest_date.strftime('%Y-%m-%d'), 
                                    latest_date.strftime('%Y-%m-%d'))
        
        if vix_data is None:
            print("✗ Could not fetch VIX data. Exiting.")
            return None
        
        # Initialize VIX columns
        vix_columns = ['vix_analysis', 'vix_analysis_minus_5', 'vix_momentum_5d', 
                      'vix_change_5d', 'vix_regime']
        
        for col in vix_columns:
            results_df[col] = np.nan
        
        # Calculate VIX features for each earnings event
        print(f"\nCalculating VIX features for {len(results_df)} earnings events...")
        
        # Debug: Show date ranges
        print(f"Earnings date range: {results_df['earnings_date'].min()} to {results_df['earnings_date'].max()}")
        print(f"VIX date range: {vix_data['date'].min()} to {vix_data['date'].max()}")
        
        successful_vix = 0
        for idx, row in results_df.iterrows():
            earnings_date = row['earnings_date']
            
            # Calculate VIX features
            vix_features = self.calculate_vix_features(earnings_date, vix_data)
            
            # Add to results
            for col in vix_columns:
                results_df.loc[idx, col] = vix_features[col]
            
            if not np.isnan(vix_features['vix_analysis']):
                successful_vix += 1
            
            # Progress update every 50 events
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(results_df)} events ({successful_vix} with VIX data)")
        
        print(f"\n✓ VIX calculation complete: {successful_vix}/{len(results_df)} events have VIX data")
        
        # Impute missing VIX values with mean (Option 2)
        print(f"\nImputing missing VIX values with sample means...")
        
        # Calculate means from non-missing values
        vix_analysis_mean = results_df['vix_analysis'].mean()
        vix_momentum_mean = results_df['vix_momentum_5d'].mean()
        vix_change_mean = results_df['vix_change_5d'].mean()
        
        # Fill missing values with means
        results_df['vix_analysis'].fillna(vix_analysis_mean, inplace=True)
        results_df['vix_momentum_5d'].fillna(vix_momentum_mean, inplace=True)
        results_df['vix_change_5d'].fillna(vix_change_mean, inplace=True)
        
        # For VIX regime, fill missing with 'Normal' (most common regime)
        results_df['vix_regime'].fillna('Normal', inplace=True)
        
        print(f"  VIX Analysis mean imputed: {vix_analysis_mean:.2f}")
        print(f"  VIX Momentum 5d mean imputed: {vix_momentum_mean:.3f}")
        print(f"  VIX Change 5d mean imputed: {vix_change_mean:.2f}")
        print(f"  VIX Regime missing filled with: Normal")
        
        # Print final VIX statistics (after imputation)
        print(f"\nVIX Statistics (After Imputation):")
        print(f"  VIX Analysis - Mean: {results_df['vix_analysis'].mean():.2f}, Std: {results_df['vix_analysis'].std():.2f}")
        print(f"  VIX Momentum 5d - Mean: {results_df['vix_momentum_5d'].mean():.3f}, Std: {results_df['vix_momentum_5d'].std():.3f}")
        print(f"  VIX Change 5d - Mean: {results_df['vix_change_5d'].mean():.2f}, Std: {results_df['vix_change_5d'].std():.2f}")
        
        print(f"\nVIX Regime Distribution (After Imputation):")
        regime_counts = results_df['vix_regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} events ({count/len(results_df)*100:.1f}%)")
        
        # Save updated results
        output_file = results_file.replace('.csv', '_with_vix.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Updated results saved to: {output_file}")
        
        return results_df

def main():
    """
    Main function to download VIX data and add to existing results.
    """
    print("VIX DATA DOWNLOADER")
    print("="*80)
    
    try:
        # Connect to WRDS
        db = wrds.Connection(wrds_username="sami_sellami",
                             password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize VIX downloader
        vix_downloader = VIXDataDownloader(db)
        
        # Add VIX features to existing results
        updated_results = vix_downloader.add_vix_to_results()
        
        if updated_results is not None:
            print(f"\n{'='*80}")
            print(f"VIX DATA INTEGRATION COMPLETE")
            print(f"{'='*80}")
            print(f"✓ VIX features added to {len(updated_results)} earnings events")
            print(f"✓ Results saved with '_with_vix' suffix")
            print(f"✓ Ready for regression analysis with VIX controls")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
