#!/usr/bin/env python3
"""
Simple Regression Test - Verify streamlined features work in regression models
Uses only basic libraries (pandas, numpy) to avoid dependency issues
"""

import pandas as pd
import numpy as np

def simple_regression_test():
    """
    Test basic regression functionality with streamlined features
    """
    print("="*80)
    print("SIMPLE REGRESSION TEST WITH STREAMLINED FEATURES")
    print("="*80)
    
    try:
        # Load the streamlined dataset
        data_file = 'data_files/streamlined_earnings_analysis_results.csv'
        data = pd.read_csv(data_file)
        
        print(f"✓ Dataset loaded: {len(data)} observations, {len(data.columns)} columns")
        
        # Check for required variables
        required_vars = ['revr', 'ievr', 'dispersion']
        missing_vars = [var for var in required_vars if var not in data.columns]
        
        if missing_vars:
            print(f"❌ Missing required variables: {missing_vars}")
            return False
        
        print(f"✓ All required variables present")
        
        # Clean data - remove NaN values
        clean_data = data.dropna(subset=['revr', 'ievr', 'dispersion'])
        print(f"✓ Clean data: {len(clean_data)} observations after removing NaN")
        
        if len(clean_data) < 10:
            print(f"❌ Insufficient clean data for regression")
            return False
        
        # Simple linear regression: REVR = α + β × IEVR
        print(f"\n{'='*60}")
        print("MODEL 1: REVR = α + β × IEVR")
        print(f"{'='*60}")
        
        X = clean_data['ievr'].values
        y = clean_data['revr'].values
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS estimation: β = (X'X)^(-1) X'y
        try:
            beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
            alpha, beta_ievr = beta
            
            print(f"✓ Regression completed successfully!")
            print(f"  α (intercept): {alpha:.4f}")
            print(f"  β (IEVR coefficient): {beta_ievr:.4f}")
            
            # Calculate R-squared
            y_pred = alpha + beta_ievr * X
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"  R-squared: {r_squared:.4f}")
            
        except Exception as e:
            print(f"❌ Regression failed: {e}")
            return False
        
        # Model 2: REVR = α + β₁×IEVR + β₂×Dispersion
        print(f"\n{'='*60}")
        print("MODEL 2: REVR = α + β₁×IEVR + β₂×Dispersion")
        print(f"{'='*60}")
        
        X2 = clean_data[['ievr', 'dispersion']].values
        X2_with_const = np.column_stack([np.ones(len(X2)), X2])
        
        try:
            beta2 = np.linalg.inv(X2_with_const.T @ X2_with_const) @ X2_with_const.T @ y
            alpha2, beta_ievr2, beta_dispersion = beta2
            
            print(f"✓ Regression completed successfully!")
            print(f"  α (intercept): {alpha2:.4f}")
            print(f"  β₁ (IEVR coefficient): {beta_ievr2:.4f}")
            print(f"  β₂ (Dispersion coefficient): {beta_dispersion:.4f}")
            
            # Calculate R-squared
            y_pred2 = alpha2 + beta_ievr2 * X2[:, 0] + beta_dispersion * X2[:, 1]
            ss_res2 = np.sum((y - y_pred2) ** 2)
            ss_tot2 = np.sum((y - np.mean(y)) ** 2)
            r_squared2 = 1 - (ss_res2 / ss_tot2)
            
            print(f"  R-squared: {r_squared2:.4f}")
            
        except Exception as e:
            print(f"❌ Regression failed: {e}")
            return False
        
        # Model 3: REVR = α + β₁×IEVR + β₂×Dispersion + β₃×Skew
        print(f"\n{'='*60}")
        print("MODEL 3: REVR = α + β₁×IEVR + β₂×Dispersion + β₃×Skew")
        print(f"{'='*60}")
        
        if 'skew' in clean_data.columns:
            # Clean data for all variables
            clean_data3 = clean_data.dropna(subset=['revr', 'ievr', 'dispersion', 'skew'])
            
            if len(clean_data3) >= 10:
                X3 = clean_data3[['ievr', 'dispersion', 'skew']].values
                y3 = clean_data3['revr'].values
                X3_with_const = np.column_stack([np.ones(len(X3)), X3])
                
                try:
                    beta3 = np.linalg.inv(X3_with_const.T @ X3_with_const) @ X3_with_const.T @ y3
                    alpha3, beta_ievr3, beta_dispersion3, beta_skew = beta3
                    
                    print(f"✓ Regression completed successfully!")
                    print(f"  α (intercept): {alpha3:.4f}")
                    print(f"  β₁ (IEVR coefficient): {beta_ievr3:.4f}")
                    print(f"  β₂ (Dispersion coefficient): {beta_dispersion3:.4f}")
                    print(f"  β₃ (Skew coefficient): {beta_skew:.4f}")
                    
                    # Calculate R-squared
                    y_pred3 = alpha3 + beta_ievr3 * X3[:, 0] + beta_dispersion3 * X3[:, 1] + beta_skew * X3[:, 2]
                    ss_res3 = np.sum((y3 - y_pred3) ** 2)
                    ss_tot3 = np.sum((y3 - np.mean(y3)) ** 2)
                    r_squared3 = 1 - (ss_res3 / ss_tot3)
                    
                    print(f"  R-squared: {r_squared3:.4f}")
                    
                except Exception as e:
                    print(f"❌ Regression failed: {e}")
            else:
                print(f"⚠ Insufficient clean data for 3-variable regression")
        else:
            print(f"⚠ Skew variable not found")
        
        print(f"\n{'='*80}")
        print("REGRESSION TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"✓ Basic regression models working")
        print(f"✓ Streamlined features integrated")
        print(f"✓ Your dataset is ready for full regression analysis!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in regression test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main test function
    """
    print("SIMPLE REGRESSION TEST")
    print("="*80)
    
    success = simple_regression_test()
    
    if success:
        print(f"\n🎉 REGRESSION TEST SUCCESSFUL!")
        print(f"Your streamlined features are working in regression models!")
        print(f"\nNext steps:")
        print(f"1. Install required libraries: matplotlib, seaborn, scipy, statsmodels")
        print(f"2. Run the full regression_analysis.py")
        print(f"3. Or continue with this simple approach for basic analysis")
    else:
        print(f"\n❌ REGRESSION TEST FAILED")
        print(f"Please check the error messages above")

if __name__ == "__main__":
    main()
