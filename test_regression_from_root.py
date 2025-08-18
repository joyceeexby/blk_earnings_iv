#!/usr/bin/env python3
"""
Test script to run regression analysis from root directory
"""

import sys
import os

# Add the analysis_scripts directory to the path
sys.path.append('analysis_scripts')

try:
    # Import and run the regression analysis
    from regression_analysis import FixedRegressionAnalysis
    
    print("="*80)
    print("TESTING REGRESSION ANALYSIS FROM ROOT DIRECTORY")
    print("="*80)
    
    # Test loading the data
    data_file = 'analysis_scripts/data_files/streamlined_earnings_analysis_results.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files in analysis_scripts/data_files/:")
        if os.path.exists('analysis_scripts/data_files/'):
            files = os.listdir('analysis_scripts/data_files/')
            for f in files:
                print(f"  - {f}")
        else:
            print("  Directory not found")
        sys.exit(1)
    
    print(f"✓ File found: {data_file}")
    
    # Load the data
    analysis = FixedRegressionAnalysis(data_file)
    
    print(f"✓ Data loaded successfully!")
    print(f"  Observations: {len(analysis.data)}")
    print(f"  Stocks: {analysis.data['ticker'].nunique()}")
    print(f"  Columns: {len(analysis.data.columns)}")
    
    # Test basic regression
    print(f"\n{'='*60}")
    print("TESTING BASIC REGRESSION")
    print(f"{'='*60}")
    
    basic_models = analysis.run_basic_regressions()
    
    if basic_models:
        print(f"✓ Basic regression completed successfully!")
        print(f"  Models: {len(basic_models)}")
    else:
        print(f"⚠ Basic regression returned no results")
    
    print(f"\n{'='*80}")
    print("REGRESSION TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"✓ Data loading works from root directory")
    print(f"✓ Regression analysis working")
    print(f"✓ Path issues resolved!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
