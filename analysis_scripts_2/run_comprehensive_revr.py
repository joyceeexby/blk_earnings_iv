#!/usr/bin/env python3
"""
Runner script for Comprehensive REVR Analysis
"""

import sys
import os
from comprehensive_revr_analysis import ComprehensiveREVRAnalysis
from comprehensive_revr_config import *
import wrds

def main():
    """
    Main function to run comprehensive REVR analysis.
    """
    print("🚀 COMPREHENSIVE REVR ANALYSIS RUNNER")
    print("="*60)
    
    # Check configuration
    if not WRDS_CONFIG['username'] or not WRDS_CONFIG['password']:
        print("❌ Error: Please update WRDS credentials in comprehensive_revr_config.py")
        print("   - Set your WRDS username and password")
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
        
        # Display analysis parameters
        print(f"\n📋 Analysis Parameters:")
        print(f"  Period: {ANALYSIS_CONFIG['start_quarter']} {ANALYSIS_CONFIG['start_year']} to {ANALYSIS_CONFIG['end_quarter']} {ANALYSIS_CONFIG['end_year']}")
        print(f"  Analysis window: {ANALYSIS_CONFIG['analysis_days_before']} days before earnings")
        print(f"  Top stocks per quarter: {ANALYSIS_CONFIG['num_top_stocks']}")
        
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
        
        expected_observations = total_quarters * ANALYSIS_CONFIG['num_top_stocks']
        print(f"  Total quarters: {total_quarters}")
        print(f"  Expected observations: ~{expected_observations:,}")
        print(f"  Estimated runtime: {total_quarters * 2:.1f} hours (assuming 2 hours per quarter)")
        
        # Confirm execution
        print(f"\n⚠️  This analysis will:")
        print(f"  - Process {total_quarters} earnings seasons")
        print(f"  - Analyze ~{expected_observations:,} stock-earnings combinations")
        print(f"  - Take several hours to complete")
        print(f"  - Save intermediate results every 10 seasons")
        
        response = input(f"\n🤔 Proceed with analysis? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Analysis cancelled by user")
            return
        
        # Run comprehensive analysis
        print(f"\n🎯 Starting comprehensive REVR analysis...")
        results = analyzer.run_comprehensive_analysis(
            start_quarter=ANALYSIS_CONFIG['start_quarter'],
            start_year=ANALYSIS_CONFIG['start_year'],
            end_quarter=ANALYSIS_CONFIG['end_quarter'],
            end_year=ANALYSIS_CONFIG['end_year'],
            analysis_days_before=ANALYSIS_CONFIG['analysis_days_before']
        )
        
        if results is not None:
            print(f"\n🎉 COMPREHENSIVE REVR ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📊 Generated {len(results):,} observations across all seasons")
            print(f"💾 Final results saved to: {OUTPUT_CONFIG['final_filename']}")
            
            # Show final summary
            print(f"\n📈 Final Summary:")
            print(f"  Total observations: {len(results):,}")
            print(f"  Unique stocks: {len(set(r['ticker'] for r in results)):,}")
            print(f"  Seasons covered: {len(set(r['season'] for r in results))}")
            
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
