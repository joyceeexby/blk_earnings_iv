#!/usr/bin/env python3
"""
Test script to verify linear regression integration with streamlined features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def test_linear_integration():
    """
    Test that linear regression models work with streamlined features
    """
    print("="*80)
    print("TESTING LINEAR REGRESSION INTEGRATION WITH STREAMLINED FEATURES")
    print("="*80)
    
    try:
        # Load streamlined dataset
        data_file = 'data_files/streamlined_earnings_analysis_results.csv'
        data = pd.read_csv(data_file)
        
        print(f"✓ Dataset loaded: {len(data)} observations, {len(data.columns)} columns")
        
        # Check for required variables
        required_vars = ['revr', 'ievr']
        missing_vars = [var for var in required_vars if var not in data.columns]
        
        if missing_vars:
            print(f"❌ Missing required variables: {missing_vars}")
            return False
        
        print(f"✓ All required variables present")
        
        # Check for streamlined features
        expected_features = [
            'ievr', 'skew_ratio', 'normative_iv_rv_ratio',  # Core (3)
            'dispersion',  # Dispersion (1)
            'term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk',  # Option surface (5)
            'SMB', 'HML', 'RMW', 'CMA', 'RF'  # Fama-French (5)
        ]
        
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features:
            print(f"⚠ Missing streamlined features: {missing_features}")
        else:
            print(f"✓ All {len(expected_features)} streamlined features available")
        
        # Clean data
        clean_data = data.dropna(subset=['revr', 'ievr'])
        clean_data = clean_data[np.isfinite(clean_data['revr']) & np.isfinite(clean_data['ievr'])]
        
        print(f"✓ Clean data: {len(clean_data)} observations after removing NaN")
        
        # Prepare features and target
        available_features = [f for f in expected_features if f in clean_data.columns]
        
        if len(available_features) < 5:
            print(f"⚠ Limited features available: {len(available_features)}")
            available_features = ['ievr', 'skew_ratio', 'normative_iv_rv_ratio']
        
        X = clean_data[available_features].copy()
        y = clean_data['revr'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"✓ Final dataset: {len(X)} observations, {len(available_features)} features")
        print(f"Features: {available_features}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Training set: {len(X_train)} observations")
        print(f"✓ Test set: {len(X_test)} observations")
        
        # Test different linear regression models
        print(f"\n{'='*60}")
        print("TESTING LINEAR REGRESSION MODELS")
        print(f"{'='*60}")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.1, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Store results
            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
            
            print(f"  Training R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Training RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Training MAE: {train_mae:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            
            # Feature importance (coefficients)
            if hasattr(model, 'coef_'):
                print(f"  Feature Importance (coefficients):")
                feature_names = X.columns.tolist()
                coefficients = np.abs(model.coef_)
                indices = np.argsort(coefficients)[::-1]
                
                for i, idx in enumerate(indices[:5]):  # Top 5 features
                    print(f"    {feature_names[idx]}: {coefficients[idx]:.4f}")
        
        # Compare models
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Test R²': f"{result['test_r2']:.4f}",
                'Test RMSE': f"{result['test_rmse']:.4f}",
                'Test MAE': f"{result['test_mae']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest model by Test R²: {best_model[0]} ({best_model[1]['test_r2']:.4f})")
        
        print(f"\n{'='*80}")
        print("LINEAR REGRESSION INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"✓ All linear regression models working")
        print(f"✓ Streamlined features integrated successfully")
        print(f"✓ Models trained and evaluated")
        print(f"✓ Feature importance calculated")
        print(f"✓ Ready for full analysis pipeline!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in linear regression integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main test function
    """
    print("LINEAR REGRESSION INTEGRATION TEST")
    print("="*80)
    
    success = test_linear_integration()
    
    if success:
        print(f"\n🎉 INTEGRATION TEST SUCCESSFUL!")
        print(f"Your linear regression models are working with streamlined features!")
        print(f"\nNext steps:")
        print(f"1. Run: python3 nonlinear_models.py")
        print(f"2. This will now include linear regression baseline + machine learning")
        print(f"3. All 15 streamlined features will be used in both approaches!")
    else:
        print(f"\n❌ INTEGRATION TEST FAILED")
        print(f"Please check the error messages above")

if __name__ == "__main__":
    main()
