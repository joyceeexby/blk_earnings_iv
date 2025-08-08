#!/usr/bin/env python3
"""
Test script for the new multiple linear regression functionality
in NonlinearModelAnalysis class.
"""

import pandas as pd
import numpy as np
from nonlinear_models import NonlinearModelAnalysis

def test_linear_regression():
    """
    Test the multiple linear regression functionality.
    """
    print("="*80)
    print("TESTING MULTIPLE LINEAR REGRESSION")
    print("="*80)
    
    try:
        # Initialize the analysis
        analysis = NonlinearModelAnalysis()
        
        # Run the complete analysis (includes linear regression)
        analysis.run_complete_analysis(optimize_hyperparameters=False)
        
        # Access linear regression results
        if hasattr(analysis, 'linear_results'):
            print("\n" + "="*60)
            print("LINEAR REGRESSION RESULTS SUMMARY")
            print("="*60)
            
            results = analysis.linear_results
            print(f"Test R²: {results['test_r2']:.4f}")
            print(f"Test RMSE: {results['test_rmse']:.4f}")
            print(f"Test MAE: {results['test_mae']:.4f}")
            
            # Show coefficient details
            model = results['model']
            print(f"\nCoefficients:")
            print(f"Intercept: {model.params['const']:.4f}")
            
            for feature in analysis.X_train.columns:
                if feature in model.params.index:
                    coef = model.params[feature]
                    pval = model.pvalues[feature]
                    significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    print(f"  {feature}: {coef:.4f} (p={pval:.4f}) {significance}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_linear_regression() 