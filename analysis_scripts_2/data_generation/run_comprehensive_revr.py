#!/usr/bin/env python3
"""
Runner script for Comprehensive REVR Analysis (Static Stock List - ST/MT Methodology)
"""

import sys
import os
from data_generation.comprehensive_revr_analysis import ComprehensiveREVRAnalysis
from data_generation.comprehensive_revr_config import *
import wrds

def main():
    """
    Main function to run comprehensive REVR analysis with static stock list using ST/MT methodology.
    """
    print("🚀 COMPREHENSIVE REVR ANALYSIS RUNNER (STATIC STOCK LIST - ST/MT METHODOLOGY)")
    print("="*60)
    print("📊 ST/MT Methodology: Short-term vs Medium-term volatility ratio")
    print("   - ST: 5-day half-life expanding EWM volatility")
    print("   - MT: 21-day half-life expanding EWM volatility")
    print("   - REVR = ST_volatility / MT_volatility at T+3 (3rd trading day after earnings)")
    print("   - Uses excess returns (stock returns - SPX returns) for market adjustment")
    print("="*60)
    
    # Check configuration
    if not WRDS_CONFIG['username'] or not WRDS_CONFIG['password']:
        print("❌ Error: Please update WRDS credentials in comprehensive_revr_config.py")
        print("   - Set your WRDS username and password")
        return
    
    # Check if static stock list file exists
    static_stocks_file = 'data_files/top500_liquidity_2005_2023.csv'
    if not os.path.exists(static_stocks_file):
        print(f"❌ Error: Static stock list file '{static_stocks_file}' not found")
        print("   - Please ensure the file exists in the current directory")
        print("   - The file should contain either 'cusip' or 'ticker' column")
        print("   - CUSIP column is preferred for more accurate data matching")
        return
    
    try:
        # Connect to WRDS
        print("🔌 Connecting to WRDS...")
        db = wrds.Connection(
            wrds_username=WRDS_CONFIG['username'],
            password=WRDS_CONFIG['password']
        )
        print("✅ Connected to WRDS successfully")
        
        # Initialize comprehensive analyzer
        print("🔧 Initializing analyzer...")
        analyzer = ComprehensiveREVRAnalysis(db)
        print("✅ Analyzer initialized")
        
        # Check what type of stock list we have
        print("📊 Analyzing stock list file...")
        import pandas as pd
        try:
            stocks_df = pd.read_csv(static_stocks_file)
            has_cusip = 'cusip' in stocks_df.columns or 'CUSIP' in stocks_df.columns
            has_ticker = 'ticker' in stocks_df.columns or 'Ticker' in stocks_df.columns
            
            if has_cusip:
                print("✅ Found CUSIP column - will use CUSIP for accurate data queries")
                if has_ticker:
                    print("✅ Found ticker column - will create CUSIP->ticker mapping")
                else:
                    print("⚠️  No ticker column - will use CUSIP only")
            else:
                print("⚠️  No CUSIP column found - will fall back to ticker-based queries")
                print("   Note: Ticker-based queries may be less accurate due to ticker changes")
            
            # Count stocks PER QUARTER (not total unique across all time)
            if 'year' in stocks_df.columns and 'quarter' in stocks_df.columns:
                # Calculate stocks per quarter
                stocks_per_quarter = stocks_df.groupby(['year', 'quarter'])['ticker'].nunique()
                avg_stocks_per_quarter = stocks_per_quarter.mean()
                min_stocks_per_quarter = stocks_per_quarter.min()
                max_stocks_per_quarter = stocks_per_quarter.max()
                
                print(f"📊 Stock count analysis:")
                print(f"  Total rows in file: {len(stocks_df):,}")
                print(f"  Total quarters: {len(stocks_per_quarter)}")
                print(f"  Average stocks per quarter: {avg_stocks_per_quarter:.0f}")
                print(f"  Range: {min_stocks_per_quarter} to {max_stocks_per_quarter} stocks per quarter")
                
                stock_count = int(avg_stocks_per_quarter)
                
            elif has_cusip:
                # Fallback: count unique CUSIPs (this might be total unique across time)
                cusip_col = 'cusip' if 'cusip' in stocks_df.columns else 'CUSIP'
                total_unique_cusips = len(stocks_df[cusip_col].dropna().unique())
                print(f"⚠️  No year/quarter columns found")
                print(f"  Total unique CUSIPs: {total_unique_cusips:,}")
                print(f"  Assuming this is stocks per quarter (may be incorrect)")
                stock_count = total_unique_cusips
                
            else:
                # Fallback: count unique tickers (this might be total unique across time)
                ticker_col = 'ticker' if 'ticker' in stocks_df.columns else 'Ticker'
                total_unique_tickers = len(stocks_df[ticker_col].dropna().unique())
                print(f"⚠️  No year/quarter columns found")
                print(f"  Total unique tickers: {total_unique_tickers:,}")
                print(f"  Assuming this is stocks per quarter (may be incorrect)")
                stock_count = total_unique_tickers
                
        except Exception as e:
            print(f"⚠️  Could not analyze stock list file: {e}")
            stock_count = "Unknown"
        
        # Display analysis parameters
        print(f"\n📋 Analysis Parameters:")
        print(f"  Period: {ANALYSIS_CONFIG['start_quarter']} {ANALYSIS_CONFIG['start_year']} to {ANALYSIS_CONFIG['end_quarter']} {ANALYSIS_CONFIG['end_year']}")
        print(f"  Analysis window: {ANALYSIS_CONFIG['analysis_days_before']} days before earnings")
        print(f"  Stock list source: {static_stocks_file}")
        print(f"  Data matching: {'CUSIP-based (recommended)' if has_cusip else 'Ticker-based (fallback)'}")
        
        # Calculate expected results
        total_quarters = 0
        current_year = ANALYSIS_CONFIG['start_year']
        current_quarter = ANALYSIS_CONFIG['start_quarter']
        
        while (current_year < ANALYSIS_CONFIG['end_year']) or (current_year == ANALYSIS_CONFIG['end_year'] and current_quarter <= ANALYSIS_CONFIG['end_quarter']):
            total_quarters += 1
            if current_quarter == 'Q1':
                current_quarter = 'Q2'
            elif current_quarter == 'Q2':
                current_quarter = 'Q3'
            elif current_quarter == 'Q3':
                current_quarter = 'Q4'
            elif current_quarter == 'Q4':
                current_quarter = 'Q1'
                current_year += 1
        
        expected_observations = total_quarters * (stock_count if isinstance(stock_count, int) else 500)
        print(f"  Total quarters: {total_quarters}")
        print(f"  Stocks per quarter: {stock_count}")
        print(f"  Expected observations: ~{expected_observations:,}")
        print(f"  Estimated runtime: {total_quarters * 1.5:.1f} hours (assuming 1.5 hours per quarter)")
        
        # Confirm execution
        print(f"\n⚠️  This analysis will:")
        print(f"  - Process {total_quarters} earnings seasons")
        print(f"  - Use static stock list from '{static_stocks_file}'")
        print(f"  - Use {'CUSIP' if has_cusip else 'ticker'} for data matching")
        print(f"  - Analyze ~{expected_observations:,} stock-earnings combinations")
        print(f"  - Take several hours to complete")
        print(f"  - Save intermediate results every 10 seasons")
        print(f"  - Generate comparison file: bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv")
        
        if has_cusip:
            print(f"  ✅ CUSIP-based analysis will provide more accurate results")
        else:
            print(f"  ⚠️  Ticker-based analysis may have some inaccuracies")
        
        response = input(f"\n🤔 Proceed with analysis? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Analysis cancelled by user")
            return
        
        # Run comprehensive analysis
        print(f"\n🎯 Starting comprehensive REVR analysis with static stock list (ST/MT methodology)...")
        results = analyzer.run_comprehensive_analysis(
            start_quarter=ANALYSIS_CONFIG['start_quarter'],
            start_year=ANALYSIS_CONFIG['start_year'],
            end_quarter=ANALYSIS_CONFIG['end_quarter'],
            end_year=ANALYSIS_CONFIG['end_year'],
            analysis_days_before=ANALYSIS_CONFIG['analysis_days_before'],
            static_stocks_csv=static_stocks_file
        )
        
        if results is not None:
            print(f"\n🎉 COMPREHENSIVE REVR ANALYSIS (ST/MT METHODOLOGY) COMPLETED SUCCESSFULLY!")
            print(f"📊 Generated {len(results):,} observations across all seasons")
            print(f"💾 Final results saved to: bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv")
            
            # Show final summary
            print(f"\n📈 Final Summary:")
            print(f"  Total observations: {len(results):,}")
            print(f"  Unique stocks: {len(set(r['ticker'] for r in results)):,}")
            print(f"  Seasons covered: {len(set(r['season'] for r in results))}")
            
            # Comparison instructions
            print(f"\n🔍 Comparison Instructions:")
            print(f"  Original file: bulk_revr_comprehensive.csv")
            print(f"  New file: bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv")
            print(f"  Use pandas to compare the two datasets:")
            print(f"    df_orig = pd.read_csv('bulk_revr_comprehensive.csv')")
            print(f"    df_new = pd.read_csv('bulk_revr_comprehensive_st_mt_static_cusip_comparison.csv')")
            
            if has_cusip:
                print(f"  ✅ New dataset includes CUSIP column for better traceability")
            print(f"  ✅ New dataset uses ST/MT methodology (same as original system)")
            
        else:
            print("❌ Analysis failed - no results generated")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
        print("💾 Any intermediate results have been saved")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💾 Check intermediate results for partial data")
        
    finally:
        # Close WRDS connection
        try:
            if 'db' in locals():
                db.close()
                print("🔌 WRDS connection closed")
        except:
            pass

if __name__ == "__main__":
    main()
