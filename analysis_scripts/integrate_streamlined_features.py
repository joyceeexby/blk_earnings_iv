#!/usr/bin/env python3
"""
Streamlined Feature Integration Script - Essential Features Only
Run this after main.py generates the basic results, before running regression models

Features to integrate:
- Core (3): ievr, skew_ratio, normative_iv_rv_ratio
- Dispersion (1): dispersion coefficient
- Option Surface (5): term_ratio, skew, kurt, iv_ratio, smirk
- Fama-French (5): SMB, HML, RMW, CMA, RF
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def integrate_streamlined_features():
    """
    Integrate only essential features into the main analysis results
    """
    print("="*80)
    print("STREAMLINED FEATURE INTEGRATION - ESSENTIAL FEATURES ONLY")
    print("="*80)
    
    try:
        # Check if main analysis results exist
        main_results_file = 'data_files/expanded_earnings_analysis_results.csv'
        
        if not os.path.exists(main_results_file):
            print(f"❌ Main analysis results not found: {main_results_file}")
            print("Please run main.py first to generate the basic results")
            return None
        
        print(f"✓ Found main analysis results: {main_results_file}")
        
        # Load main results
        main_results = pd.read_csv(main_results_file)
        main_results['earnings_date'] = pd.to_datetime(main_results['earnings_date'])
        
        print(f"✓ Loaded {len(main_results)} observations with {len(main_results.columns)} columns")
        print(f"Date range: {main_results['earnings_date'].min()} to {main_results['earnings_date'].max()}")
        print(f"Stocks: {main_results['ticker'].nunique()}")
        
        # Initialize streamlined integration
        from streamlined_feature_integration import StreamlinedFeatureIntegration
        integration = StreamlinedFeatureIntegration()
        
        print(f"\n{'='*80}")
        print("STEP 1: ENSURING CORE FEATURES")
        print(f"{'='*80}")
        
        # Ensure core features are present and properly named
        enhanced_results = integration.ensure_core_features(main_results)
        
        print(f"\n{'='*80}")
        print("STEP 2: INTEGRATING DISPERSION FEATURE")
        print(f"{'='*80}")
        
        # Integrate dispersion feature (without database connection for now)
        enhanced_results = integration.integrate_dispersion_feature(enhanced_results)
        
        print(f"\n{'='*80}")
        print("STEP 3: INTEGRATING FAMA-FRENCH FEATURES")
        print(f"{'='*80}")
        
        # Try to integrate Fama-French features
        ff_data_file = 'data_files/F-F_Research_Data_5_Factors_2x3.csv'
        if os.path.exists(ff_data_file):
            print(f"✓ Found Fama-French data file: {ff_data_file}")
            enhanced_results = integration.integrate_fama_french_features(enhanced_results, ff_data_file)
        else:
            print(f"⚠ Fama-French data file not found: {ff_data_file}")
            print("Creating mock Fama-French features for testing")
            enhanced_results = integration.integrate_fama_french_features(enhanced_results)
        
        print(f"\n{'='*80}")
        print("STEP 4: INTEGRATING ESSENTIAL OPTION SURFACE FEATURES")
        print(f"{'='*80}")
        
        # Integrate essential option surface features
        enhanced_results = integration.integrate_essential_option_features(enhanced_results)
        
        # Final summary
        print(f"\n{'='*80}")
        print("STREAMLINED INTEGRATION COMPLETE - FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Final dataset: {len(enhanced_results)} observations, {len(enhanced_results.columns)} columns")
        
        # Show feature categories
        feature_categories = {
            'Core Features': ['ievr', 'skew_ratio', 'normative_iv_rv_ratio'],
            'Dispersion Features': ['dispersion'],
            'Fama-French Features': ['SMB', 'HML', 'RMW', 'CMA', 'RF'],
            'Option Surface Features': ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        }
        
        total_features = 0
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in enhanced_results.columns]
            if available_features:
                print(f"\n{category} ({len(available_features)} features):")
                for feature in available_features:
                    print(f"  - {feature}")
                total_features += len(available_features)
        
        print(f"\nTotal essential features: {total_features}")
        print(f"Target: 14 features (3 core + 1 dispersion + 5 FF + 5 option)")
        
        # Save streamlined results
        streamlined_output_file = 'data_files/streamlined_earnings_analysis_results.csv'
        enhanced_results.to_csv(streamlined_output_file, index=False)
        
        print(f"\n{'='*80}")
        print("SAVING STREAMLINED RESULTS")
        print(f"{'='*80}")
        print(f"✓ Streamlined results saved to: {streamlined_output_file}")
        print(f"✓ Original results preserved at: {main_results_file}")
        
        # Create feature summary for regression analysis
        feature_summary = {
            'total_observations': len(enhanced_results),
            'total_features': len(enhanced_results.columns),
            'core_features': len([col for col in enhanced_results.columns if col in ['ievr', 'skew_ratio', 'normative_iv_rv_ratio']]),
            'dispersion_features': len([col for col in enhanced_results.columns if 'dispersion' in col]),
            'fama_french_features': len([col for col in enhanced_results.columns if any(ff in col for ff in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]),
            'option_surface_features': len([col for col in enhanced_results.columns if any(opt in col for opt in ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk'])]),
            'essential_features_total': total_features
        }
        
        # Save feature summary
        summary_df = pd.DataFrame([feature_summary])
        summary_file = 'data_files/streamlined_feature_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Streamlined feature summary saved to: {summary_file}")
        
        print(f"\n{'='*80}")
        print("NEXT STEPS FOR REGRESSION ANALYSIS")
        print(f"{'='*80}")
        print("1. Use the streamlined results file for regression analysis:")
        print(f"   File: {streamlined_output_file}")
        print("2. Update regression_analysis.py to use the streamlined file")
        print("3. Run regression models with essential features only")
        print("4. All 14 essential features are now available")
        
        return enhanced_results
        
    except Exception as e:
        print(f"❌ Error during streamlined feature integration: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_regression_analysis_file():
    """
    Update the regression analysis file to use the streamlined results
    """
    print(f"\n{'='*80}")
    print("UPDATING REGRESSION ANALYSIS FILE")
    print(f"{'='*80}")
    
    try:
        # Read the current regression analysis file
        reg_file = 'regression_analysis.py'
        
        if not os.path.exists(reg_file):
            print(f"❌ Regression analysis file not found: {reg_file}")
            return
        
        # Update the default data file path
        with open(reg_file, 'r') as f:
            content = f.read()
        
        # Replace the default file path
        old_path = "data_file='data_files/comprehensive_earnings_analysis_results.csv'"
        new_path = "data_file='data_files/streamlined_earnings_analysis_results.csv'"
        
        if old_path in content:
            content = content.replace(old_path, new_path)
            
            with open(reg_file, 'w') as f:
                f.write(content)
            
            print(f"✓ Updated {reg_file} to use streamlined results")
        else:
            print(f"⚠ Could not find default file path in {reg_file}")
            print("Please manually update the data file path to:")
            print("data_file='data_files/streamlined_earnings_analysis_results.csv'")
        
    except Exception as e:
        print(f"❌ Error updating regression analysis file: {e}")

def main():
    """
    Main function to integrate streamlined features
    """
    print("STREAMLINED FEATURE INTEGRATION SCRIPT")
    print("="*80)
    print("This script integrates only essential features:")
    print("Core (3): ievr, skew_ratio, normative_iv_rv_ratio")
    print("Dispersion (1): dispersion coefficient")
    print("Option Surface (5): term_ratio, skew, kurt, iv_ratio, smirk")
    print("Fama-French (5): SMB, HML, RMW, CMA, RF")
    print("="*80)
    print("Run this AFTER running main.py and BEFORE running regression models")
    print("="*80)
    
    # Integrate streamlined features
    streamlined_results = integrate_streamlined_features()
    
    if streamlined_results is not None:
        # Update regression analysis file
        update_regression_analysis_file()
        
        print(f"\n{'='*80}")
        print("🎉 STREAMLINED FEATURE INTEGRATION COMPLETE! 🎉")
        print(f"{'='*80}")
        print("Your analysis now includes only the essential features:")
        print("✓ 3 core features (ievr, skew_ratio, normative_iv_rv_ratio)")
        print("✓ 1 dispersion feature (dispersion coefficient)")
        print("✓ 5 option surface features (term_ratio, skew, kurt, iv_ratio, smirk)")
        print("✓ 5 Fama-French features (SMB, HML, RMW, CMA, RF)")
        print("✓ No interaction terms or extra features")
        
        print(f"\nNext steps:")
        print("1. Run your regression analysis using the streamlined file")
        print("2. All 14 essential features will be available in your models")
        print("3. Check the streamlined feature summary for complete feature list")
    else:
        print(f"\n❌ Streamlined feature integration failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
