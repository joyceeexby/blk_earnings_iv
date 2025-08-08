#!/usr/bin/env python3
"""
Nonlinear Models with Fama-French 5-Factor Model
Updated version that includes Fama-French 5 factors as additional features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class NonlinearModelAnalysisWithFF:
    """
    Nonlinear model analysis with Fama-French 5 factors.
    """
    
    def __init__(self, data_file='data_files/earnings_with_fama_french_5factor_monthly_match.csv'):
        """
        Initialize with data file that includes Fama-French 5 factors.
        """
        self.data_file = data_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.linear_results = None
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features for modeling including Fama-French 5 factors.
        """
        print("Loading and preparing data for non-linear modeling with Fama-French 5 factors...")
        
        # Load data
        self.data = pd.read_csv(self.data_file)
        
        # Clean data - remove NaN and infinite values
        self.data = self.data.dropna(subset=['revr', 'ievr'])
        self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
        
        # Create additional features
        self.create_features()
        
        # Prepare features and target - include Fama-French 5 factors
        base_features = ['ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr']
        ff_features = ['SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return', 'Mkt_Volatility', 'Factor_Volatility']
        
        # Combine all features
        feature_columns = base_features + ff_features
        
        # Remove columns that might not exist
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if len(available_features) < 4:
            # Fallback to basic features
            available_features = ['ievr']
            print("Warning: Limited features available, using only IEVR")
        
        X = self.data[available_features].copy()
        y = self.data['revr'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print("Final dataset: {} observations, {} features".format(len(X), len(available_features)))
        print("Base features: {}".format([f for f in base_features if f in available_features]))
        print("Fama-French 5 factors: {}".format([f for f in ff_features if f in available_features]))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} observations")
        print(f"Test set: {len(self.X_test)} observations")
        
        # Add training/test period information if earnings_date column exists
        if 'earnings_date' in self.data.columns:
            # Get the original indices to map back to the data
            train_indices = self.X_train.index
            test_indices = self.X_test.index
            
            # Get date ranges
            train_dates = self.data.loc[train_indices, 'earnings_date']
            test_dates = self.data.loc[test_indices, 'earnings_date']
            
            print(f"Training period: {train_dates.min()} to {train_dates.max()}")
            print(f"Test period: {test_dates.min()} to {test_dates.max()}")
            
            # Show number of unique stocks in each set
            if 'ticker' in self.data.columns:
                train_tickers = self.data.loc[train_indices, 'ticker'].nunique()
                test_tickers = self.data.loc[test_indices, 'ticker'].nunique()
                print(f"Training stocks: {train_tickers} unique tickers")
                print(f"Test stocks: {test_tickers} unique tickers")
    
    def create_features(self):
        """
        Create additional features for modeling.
        """
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
        
        # Create S&P 500 IEVR feature
        self.create_spx_ievr_feature()
        
        # Create Fama-French 5-factor interaction features
        self.create_ff_interaction_features()
        
        # Log transformations (for positive values)
        mask_positive = (self.data['revr'] > 0) & (self.data['ievr'] > 0)
        if mask_positive.sum() > 0:
            self.data.loc[mask_positive, 'log_revr'] = np.log(self.data.loc[mask_positive, 'revr'])
            self.data.loc[mask_positive, 'log_ievr'] = np.log(self.data.loc[mask_positive, 'ievr'])
        
        # Squared terms
        self.data['ievr_squared'] = self.data['ievr'] ** 2
    
    def create_ff_interaction_features(self):
        """
        Create interaction features between IEVR and Fama-French 5 factors.
        """
        print("Creating Fama-French 5-factor interaction features...")
        
        # Check if Fama-French factors exist
        ff_factors = ['SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mkt_Return', 'Mkt_Volatility']
        available_ff = [f for f in ff_factors if f in self.data.columns]
        
        if len(available_ff) > 0:
            # Create IEVR interactions with each factor
            for factor in available_ff:
                interaction_name = f'IEVR_{factor}_Interaction'
                self.data[interaction_name] = self.data['ievr'] * self.data[factor]
                print(f"  Created {interaction_name}")
            
            # Create market regime features
            if 'Mkt_Volatility' in self.data.columns:
                # High volatility regime
                self.data['High_Mkt_Vol_Regime'] = (self.data['Mkt_Volatility'] > 
                                                   self.data['Mkt_Volatility'].quantile(0.75)).astype(int)
                
                # IEVR in high volatility regime
                self.data['IEVR_High_Vol_Regime'] = self.data['ievr'] * self.data['High_Mkt_Vol_Regime']
                print("  Created market volatility regime features")
            
            # Create factor momentum features
            for factor in ['SMB', 'HML', 'RMW', 'CMA']:
                if factor in self.data.columns:
                    # 6-month momentum
                    momentum_name = f'{factor}_Momentum_6m'
                    self.data[momentum_name] = self.data[factor].rolling(window=6).mean()
                    
                    # IEVR interaction with momentum
                    interaction_name = f'IEVR_{momentum_name}_Interaction'
                    self.data[interaction_name] = self.data['ievr'] * self.data[momentum_name]
                    print(f"  Created {interaction_name}")
        else:
            print("  No Fama-French 5 factors found in data")
    
    def create_spx_ievr_feature(self):
        """
        Create S&P 500 IEVR feature.
        """
        print("Creating S&P 500 IEVR feature...")
        
        if 'spx_ievr' not in self.data.columns:
            print("Warning: 'spx_ievr' not found in data. Creating placeholder.")
            self.data['spx_ievr'] = 1.0
        else:
            if self.data['spx_ievr'].isna().all():
                print("Warning: 'spx_ievr' column exists but is empty. Creating placeholder.")
                self.data['spx_ievr'] = 1.0
        
        self.data['spx_ievr'] = self.data['spx_ievr'].replace([np.inf, -np.inf], np.nan)
        print(f"Created spx_ievr feature. Non-null values: {self.data['spx_ievr'].notna().sum()}")
    
    def create_normative_iv_rv_ratio(self):
        """
        Create normative IV/RV ratio feature.
        """
        print("Creating normative IV/RV ratio feature...")
        
        if 'normative_implied_vol' not in self.data.columns:
            print("Warning: 'normative_implied_vol' not found in data. Creating placeholder.")
            self.data['normative_implied_vol'] = self.data['ievr'] * 1.0
        
        if 'normative_realized_vol' not in self.data.columns:
            print("Warning: 'normative_realized_vol' not found in data. Creating placeholder.")
            self.data['normative_realized_vol'] = 1.0
        
        mask = (self.data['normative_implied_vol'] > 0) & (self.data['normative_realized_vol'] > 0)
        self.data.loc[mask, 'normative_iv_rv_ratio'] = (
            self.data.loc[mask, 'normative_implied_vol'] / 
            self.data.loc[mask, 'normative_realized_vol']
        )
        
        self.data['normative_iv_rv_ratio'] = self.data['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
        print(f"Created normative_iv_rv_ratio feature. Non-null values: {self.data['normative_iv_rv_ratio'].notna().sum()}")
    
    def create_skew_ratio(self):
        """
        Create skew ratio feature.
        """
        print("Creating skew ratio feature...")
        
        if 'skew_ratio' not in self.data.columns:
            print("Warning: 'skew_ratio' not found in data. Creating placeholder.")
            self.data['skew_ratio'] = 1.0
        else:
            if self.data['skew_ratio'].isna().all():
                print("Warning: 'skew_ratio' column exists but is empty. Creating placeholder.")
                self.data['skew_ratio'] = 1.0
        
        self.data['skew_ratio'] = self.data['skew_ratio'].replace([np.inf, -np.inf], np.nan)
        print(f"Created skew_ratio feature. Non-null values: {self.data['skew_ratio'].notna().sum()}")
    
    def train_multiple_linear_regression(self):
        """
        Train Multiple Linear Regression model with Fama-French 5 factors.
        """
        print("\n" + "="*60)
        print("TRAINING MULTIPLE LINEAR REGRESSION MODEL WITH FF 5-FACTORS")
        print("="*60)
        
        try:
            import statsmodels.api as sm
            
            # Prepare data for statsmodels (add constant)
            X_train_with_constant = sm.add_constant(self.X_train)
            X_test_with_constant = sm.add_constant(self.X_test)
            
            # Fit the model
            model = sm.OLS(self.y_train, X_train_with_constant).fit()
            
            # Store the model
            self.linear_model = model
            
            # Print detailed summary
            print("\nMultiple Linear Regression Results (with FF 5-factors):")
            print("="*50)
            print(model.summary())
            
            # Make predictions
            y_pred_train = model.predict(X_train_with_constant)
            y_pred_test = model.predict(X_test_with_constant)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # Store results
            self.linear_results = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
            
            # Print performance metrics
            print(f"\nModel Performance:")
            print(f"  Training R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Training RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Training MAE: {train_mae:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            
            # Print coefficient interpretation
            print(f"\nCoefficient Interpretation:")
            print(f"  Intercept: {model.params['const']:.4f}")
            for feature in self.X_train.columns:
                if feature in model.params.index:
                    coef = model.params[feature]
                    pval = model.pvalues[feature]
                    tstat = model.tvalues[feature]
                    significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    print(f"  {feature}: {coef:.4f} (t={tstat:.3f}, p={pval:.4f}) {significance}")
            
            # Check for overfitting
            if train_r2 - test_r2 > 0.1:
                print(f"  ⚠ Warning: Potential overfitting (train R² - test R² = {train_r2 - test_r2:.3f})")
            else:
                print(f"  ✓ Model shows good generalization (train R² - test R² = {train_r2 - test_r2:.3f})")
            
            return model
            
        except Exception as e:
            print(f"Error training multiple linear regression: {e}")
            return None
    
    def train_random_forest(self, optimize_hyperparameters=True):
        """
        Train Random Forest model with Fama-French 5 factors.
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL WITH FF 5-FACTORS")
        print("="*60)
        
        if optimize_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            best_rf = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_rf = RandomForestRegressor(
                n_estimators=100, 
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            best_rf.fit(self.X_train_scaled, self.y_train)
        
        # Store model
        self.models['random_forest'] = best_rf
        
        # Evaluate model
        self.evaluate_model('random_forest', 'Random Forest')
        
        return best_rf
    
    def train_xgboost(self, optimize_hyperparameters=True):
        """
        Train XGBoost model with Fama-French 5 factors.
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL WITH FF 5-FACTORS")
        print("="*60)
        
        if optimize_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            best_xgb = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_xgb = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                random_state=42
            )
            best_xgb.fit(self.X_train_scaled, self.y_train)
        
        # Store model
        self.models['xgboost'] = best_xgb
        
        # Evaluate model
        self.evaluate_model('xgboost', 'XGBoost')
        
        return best_xgb
    
    def evaluate_model(self, model_name, model_display_name):
        """
        Evaluate a trained model.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        # Make predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        self.results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Print results
        print(f"\n{model_display_name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance for models that support it.
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS (FF 5-FACTORS)")
        print("="*60)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.X_train.columns.tolist()
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n{model_name.replace('_', ' ').title()} Feature Importance:")
                print("-" * 40)
                
                for i, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
                
                # Categorize features
                ff_features = [f for f in feature_names if any(ff in f for ff in ['SMB', 'HML', 'RMW', 'CMA', 'RF'])]
                base_features = [f for f in feature_names if f not in ff_features]
                
                ff_importance = importance_df[importance_df['feature'].isin(ff_features)]['importance'].sum()
                base_importance = importance_df[importance_df['feature'].isin(base_features)]['importance'].sum()
                
                print(f"\n  Fama-French 5 factors total importance: {ff_importance:.4f}")
                print(f"  Base features total importance: {base_importance:.4f}")
    
    def run_complete_analysis(self, optimize_hyperparameters=True):
        """
        Run complete analysis including linear and non-linear models with FF 5 factors.
        """
        print("="*80)
        print("COMPREHENSIVE MACHINE LEARNING ANALYSIS WITH FAMA-FRENCH 5-FACTORS")
        print("="*80)
        
        # Train linear regression first (baseline)
        print("\n" + "="*60)
        print("LINEAR BASELINE MODEL")
        print("="*60)
        self.train_multiple_linear_regression()
        
        # Train non-linear models
        print("\n" + "="*60)
        print("NON-LINEAR MODELS")
        print("="*60)
        self.train_random_forest(optimize_hyperparameters)
        self.train_xgboost(optimize_hyperparameters)
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Print summary
        self.print_summary_table()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def print_summary_table(self):
        """
        Print a summary table comparing all models including linear regression.
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY (WITH FAMA-FRENCH 5-FACTORS)")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        
        # Add linear regression results if available
        if hasattr(self, 'linear_results'):
            summary_data.append({
                'Model': 'Multiple Linear Regression',
                'Test R²': f"{self.linear_results['test_r2']:.4f}",
                'CV R²': 'N/A',
                'Test RMSE': f"{self.linear_results['test_rmse']:.4f}",
                'Test MAE': f"{self.linear_results['test_mae']:.4f}"
            })
        
        # Add non-linear model results
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test R²': f"{results['test_r2']:.4f}",
                'CV R²': f"{results['cv_mean']:.4f} (±{results['cv_std']:.4f})",
                'Test RMSE': f"{results['test_rmse']:.4f}",
                'Test MAE': f"{results['test_mae']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv('data_files/nonlinear_model_summary_with_ff_5factor.csv', index=False)
        print("\n✓ Model summary saved to data_files/nonlinear_model_summary_with_ff_5factor.csv")
        
        # Print linear vs non-linear comparison
        if hasattr(self, 'linear_results') and self.results:
            print(f"\nLinear vs Non-linear Comparison:")
            linear_r2 = self.linear_results['test_r2']
            best_nonlinear_r2 = max([results['test_r2'] for results in self.results.values()])
            improvement = best_nonlinear_r2 - linear_r2
            print(f"  Linear Regression R²: {linear_r2:.4f}")
            print(f"  Best Non-linear R²: {best_nonlinear_r2:.4f}")
            print(f"  Improvement: {improvement:.4f} ({improvement/linear_r2*100:.1f}%)")
            
            if improvement > 0.05:
                print(f"  ✓ Non-linear models provide significant improvement")
            elif improvement > 0.01:
                print(f"  ⚠ Non-linear models provide modest improvement")
            else:
                print(f"  ⚠ Linear model performs similarly to non-linear models")

def main():
    """
    Main function to run the non-linear analysis with Fama-French 5 factors.
    """
    # Run analysis with hyperparameter optimization
    analysis = NonlinearModelAnalysisWithFF()
    analysis.run_complete_analysis(optimize_hyperparameters=True)

if __name__ == "__main__":
    main()
