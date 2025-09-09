#!/usr/bin/env python3
"""
Simple Linear Regression: IEVR vs REVR
Just the essential regression analysis with clean output
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Run simple linear regression and print results.
    """
    print("🔬 SIMPLE LINEAR REGRESSION: IEVR vs REVR")
    print("="*60)
    
    # Load data
    file_path = 'data_files/merged_revr_ievr_comprehensive.csv'
    df = pd.read_csv(file_path)
    print(f"📊 Loaded dataset: {len(df):,} observations")
    
    # Remove extreme outliers (z-score > 3)
    revr_zscore = np.abs((df['revr'] - df['revr'].mean()) / df['revr'].std())
    ievr_zscore = np.abs((df['ievr'] - df['ievr'].mean()) / df['ievr'].std())
    
    df_clean = df[(revr_zscore <= 3) & (ievr_zscore <= 3)].copy()
    print(f"🧹 After outlier removal: {len(df_clean):,} observations")
    
    # Prepare variables
    X = df_clean['ievr'].values.reshape(-1, 1)  # Independent: IEVR
    y = df_clean['revr'].values                  # Dependent: REVR
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    correlation = df_clean['revr'].corr(df_clean['ievr'])
    
    # Print results
    print(f"\n📊 REGRESSION RESULTS")
    print("="*60)
    print(f"Model: REVR = α + β × IEVR + ε")
    print(f"")
    print(f"Intercept (α): {intercept:.4f}")
    print(f"Slope (β):     {slope:.4f}")
    print(f"")
    print(f"R-squared:     {r2:.4f}")
    print(f"Correlation:   {correlation:.4f}")
    print(f"RMSE:          {rmse:.4f}")
    print(f"")
    print(f"📝 INTERPRETATION:")
    print(f"• For every 1 unit increase in IEVR, REVR changes by {slope:.4f} units")
    print(f"• {r2*100:.1f}% of variance in REVR is explained by IEVR")
    print(f"• Correlation indicates a {'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'} relationship")
    
    if slope > 0:
        print(f"• Positive relationship: Higher implied volatility predicts higher realized volatility")
    else:
        print(f"• Negative relationship: Higher implied volatility predicts lower realized volatility")

if __name__ == "__main__":
    main()
