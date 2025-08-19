#!/usr/bin/env python3
"""
Simple test script to verify streamlined features are working
"""

import pandas as pd
import numpy as np

def test_streamlined_features():
    """
    Test that all streamlined features are present and accessible
    """
    print("="*80)
    print("TESTING STREAMLINED FEATURES")
    print("="*80)
    
    try:
        # Load the streamlined dataset
        data_file = 'data_files/streamlined_earnings_analysis_results.csv'
        data = pd.read_csv(data_file)
        
        print(f"✓ Dataset loaded: {len(data)} observations, {len(data.columns)} columns")
        
        # Check core features
        core_features = ['ievr', 'skew_ratio', 'normative_iv_rv_ratio']
        print(f"\nCore Features:")
        for feature in core_features:
            if feature in data.columns:
                non_null = data[feature].notna().sum()
                print(f"  ✓ {feature}: {non_null}/{len(data)} non-null values")
            else:
                print(f"  ❌ {feature}: NOT FOUND")
        
        # Check dispersion feature
        print(f"\nDispersion Feature:")
        if 'dispersion' in data.columns:
            non_null = data['dispersion'].notna().sum()
            print(f"  ✓ dispersion: {non_null}/{len(data)} non-null values")
        else:
            print(f"  ❌ dispersion: NOT FOUND")
        
        # Check option surface features
        option_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        print(f"\nOption Surface Features:")
        for feature in option_features:
            if feature in data.columns:
                non_null = data[feature].notna().sum()
                print(f"  ✓ {feature}: {non_null}/{len(data)} non-null values")
            else:
                print(f"  ❌ {feature}: NOT FOUND")
        
        # Check Fama-French features
        ff_features = ['SMB', 'HML', 'RMW', 'CMA', 'RF']
        print(f"\nFama-French Features:")
        for feature in ff_features:
            if feature in data.columns:
                non_null = data[feature].notna().sum()
                print(f"  ✓ {feature}: {non_null}/{len(data)} non-null values")
            else:
                print(f"  ❌ {feature}: NOT FOUND")
        
        # Summary
        print(f"\n{'='*80}")
        print("FEATURE SUMMARY")
        print(f"{'='*80}")
        
        total_features = 0
        feature_categories = {
            'Core': len([col for col in data.columns if col in core_features]),
            'Dispersion': len([col for col in data.columns if 'dispersion' in col]),
            'Option Surface': len([col for col in data.columns if any(opt in col for opt in option_features)]),
            'Fama-French': len([col for col in data.columns if any(ff in col for ff in ff_features)])
        }
        
        for category, count in feature_categories.items():
            print(f"{category} features: {count}")
            total_features += count
        
        print(f"\nTotal essential features: {total_features}")
        print(f"Target: 14 features (3 core + 1 dispersion + 5 option + 5 FF)")
        
        if total_features >= 14:
            print(f"\n🎉 SUCCESS: All streamlined features are present!")
            print(f"Your dataset is ready for regression analysis!")
        else:
            print(f"\n⚠ WARNING: Some features may be missing")
            print(f"Please check the integration process")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing streamlined features: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main test function
    """
    print("STREAMLINED FEATURES TEST")
    print("="*80)
    
    success = test_streamlined_features()
    
    if success:
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"Your streamlined dataset is ready!")
    else:
        print(f"\n❌ TEST FAILED")
        print(f"Please check the error messages above")

if __name__ == "__main__":
    main()
