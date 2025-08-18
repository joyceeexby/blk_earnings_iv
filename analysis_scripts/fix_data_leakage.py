#!/usr/bin/env python3
"""
Fix Data Leakage in Streamlined Dataset
Remove features that are too closely related to the target variable (REVR)
"""

import pandas as pd
import numpy as np

def fix_data_leakage():
    """
    Fix data leakage by removing problematic features
    """
    print("="*80)
    print("FIXING DATA LEAKAGE IN STREAMLINED DATASET")
    print("="*80)
    
    try:
        # Load the original streamlined dataset
        data_file = 'data_files/streamlined_earnings_analysis_results.csv'
        data = pd.read_csv(data_file)
        
        print(f"✓ Original dataset loaded: {len(data)} observations, {len(data.columns)} columns")
        
        # Identify problematic features that cause data leakage
        problematic_features = [
            'ratio',                    # This is essentially REVR (vol_st/vol_mt)
            'normative_iv_rv_ratio',   # Too closely related to REVR
            'vol_st',                   # Component of REVR
            'vol_mt',                   # Component of REVR
            'avg_pre',                  # Component of REVR calculation
            'avg_post',                 # Component of REVR calculation
            'normative_implied_vol',    # Component of problematic ratio
            'normative_realized_vol'    # Component of problematic ratio
        ]
        
        # Check which problematic features exist
        existing_problematic = [f for f in problematic_features if f in data.columns]
        print(f"⚠ Problematic features found: {existing_problematic}")
        
        # Remove problematic features
        data_clean = data.drop(columns=existing_problematic, errors='ignore')
        
        print(f"✓ Removed {len(existing_problematic)} problematic features")
        print(f"✓ Clean dataset: {len(data_clean)} observations, {len(data_clean.columns)} columns")
        
        # Verify target variable is still present
        if 'revr' not in data_clean.columns:
            print("❌ ERROR: Target variable 'revr' was accidentally removed!")
            return None
        
        # Check remaining features
        remaining_features = [col for col in data_clean.columns if col not in ['earnings_date', 'ticker', 'methodology', 'underlying_price']]
        print(f"✓ Remaining analysis features: {len(remaining_features)}")
        print(f"Features: {remaining_features}")
        
        # Verify no remaining data leakage
        print(f"\n{'='*60}")
        print("DATA LEAKAGE CHECK")
        print(f"{'='*60}")
        
        # Check correlations with REVR
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        revr_correlations = data_clean[numeric_cols].corr()['revr'].sort_values(key=abs, ascending=False)
        
        print("Correlations with REVR:")
        for feature, corr in revr_correlations.head(10).items():
            if feature != 'revr':
                print(f"  {feature}: {corr:.4f}")
        
        # Check for suspiciously high correlations
        high_corr_features = revr_correlations[abs(revr_correlations) > 0.8]
        if len(high_corr_features) > 1:  # More than just REVR itself
            print(f"\n⚠ WARNING: High correlations found (potential remaining leakage):")
            for feature, corr in high_corr_features.items():
                if feature != 'revr':
                    print(f"  {feature}: {corr:.4f}")
        else:
            print(f"\n✓ No suspiciously high correlations found")
        
        # Save clean dataset
        output_file = 'data_files/clean_streamlined_earnings_analysis_results.csv'
        data_clean.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("DATA LEAKAGE FIXED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"✓ Problematic features removed")
        print(f"✓ Clean dataset saved to: {output_file}")
        print(f"✓ Ready for legitimate analysis")
        
        return data_clean
        
    except Exception as e:
        print(f"❌ Error fixing data leakage: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_clean_dataset():
    """
    Analyze the clean dataset to verify it's ready for analysis
    """
    print(f"\n{'='*60}")
    print("ANALYZING CLEAN DATASET")
    print(f"{'='*60}")
    
    try:
        # Load clean dataset
        clean_file = 'data_files/clean_streamlined_earnings_analysis_results.csv'
        clean_data = pd.read_csv(clean_file)
        
        print(f"✓ Clean dataset loaded: {len(clean_data)} observations, {len(clean_data.columns)} columns")
        
        # Check for required variables
        required_vars = ['revr', 'ievr']
        missing_vars = [var for var in required_vars if var not in clean_data.columns]
        
        if missing_vars:
            print(f"❌ Missing required variables: {missing_vars}")
            return False
        
        print(f"✓ All required variables present")
        
        # Check for streamlined features (excluding problematic ones)
        expected_features = [
            'ievr', 'skew_ratio',  # Core (2 - removed normative_iv_rv_ratio)
            'dispersion',  # Dispersion (1)
            'term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk',  # Option surface (5)
            'SMB', 'HML', 'RMW', 'CMA', 'RF'  # Fama-French (5)
        ]
        
        missing_features = [f for f in expected_features if f not in clean_data.columns]
        if missing_features:
            print(f"⚠ Missing streamlined features: {missing_features}")
        else:
            print(f"✓ All {len(expected_features)} clean streamlined features available")
        
        # Check data quality
        clean_data_numeric = clean_data.dropna(subset=['revr', 'ievr'])
        clean_data_numeric = clean_data_numeric[np.isfinite(clean_data_numeric['revr']) & np.isfinite(clean_data_numeric['ievr'])]
        
        print(f"✓ Clean numeric data: {len(clean_data_numeric)} observations after removing NaN")
        
        # Check correlations again
        numeric_cols = clean_data_numeric.select_dtypes(include=[np.number]).columns
        revr_correlations = clean_data_numeric[numeric_cols].corr()['revr'].sort_values(key=abs, ascending=False)
        
        print(f"\nFinal correlations with REVR (top 5):")
        for feature, corr in revr_correlations.head(6).items():
            if feature != 'revr':
                print(f"  {feature}: {corr:.4f}")
        
        # Check for reasonable correlations
        reasonable_correlations = revr_correlations[abs(revr_correlations) < 0.7]
        if len(reasonable_correlations) > 1:
            print(f"\n✓ Correlations look reasonable (no suspiciously high values)")
        else:
            print(f"\n⚠ Some correlations still seem suspicious")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing clean dataset: {e}")
        return False

def main():
    """
    Main function to fix data leakage
    """
    print("DATA LEAKAGE FIX")
    print("="*80)
    
    # Fix data leakage
    clean_data = fix_data_leakage()
    
    if clean_data is not None:
        # Analyze clean dataset
        success = analyze_clean_dataset()
        
        if success:
            print(f"\n🎉 DATA LEAKAGE SUCCESSFULLY FIXED!")
            print(f"Your dataset is now ready for legitimate analysis!")
            print(f"\nNext steps:")
            print(f"1. Use the clean dataset: clean_streamlined_earnings_analysis_results.csv")
            print(f"2. R² values should now be more reasonable (< 0.7)")
            print(f"3. Run your analysis again with the clean data")
        else:
            print(f"\n⚠ Clean dataset analysis had issues")
    else:
        print(f"\n❌ Failed to fix data leakage")

if __name__ == "__main__":
    main()
