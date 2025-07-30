"""
Non-linear Machine Learning Models for IEVR-REVR Analysis

This module implements various non-linear models to explore the relationship between
Implied Earnings Volatility Ratio (IEVR) and Realized Earnings Volatility Ratio (REVR).

Models included:
- Random Forest Regression
- XGBoost Regression
- Model comparison and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class NonlinearModelAnalysis:
    """
    Class for implementing and comparing non-linear models for IEVR-REVR analysis.
    """
    
    def __init__(self, data_file='data_files/expanded_earnings_analysis_results.csv'):
        """
        Initialize the analysis with data.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        """
        self.data_file = data_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features for modeling.
        """
        print("Loading and preparing data for non-linear modeling...")
        
        # Load data
        self.data = pd.read_csv(self.data_file)
        
        # Clean data - remove NaN and infinite values
        self.data = self.data.dropna(subset=['revr', 'ievr'])
        self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
        
        # Create additional features
        self.create_features()
        
        # Prepare features and target - only use truly independent features
        feature_columns = ['ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr']  # Independent features including S&P 500 IEVR
        # Remove columns that might not exist
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if len(available_features) < 2:
            # Fallback to basic features
            available_features = ['ievr']
            print("Warning: Limited features available, using only IEVR")
        
        X = self.data[available_features].copy()
        y = self.data['revr'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset: {len(X)} observations, {len(available_features)} features")
        print(f"Features: {available_features}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} observations")
        print(f"Test set: {len(self.X_test)} observations")
    
    def create_features(self):
        """
        Create additional features for modeling.
        """
        # Note: Removed vol_st, vol_mt, and volatility_spread features to avoid circular dependency
        # These are components of the REVR target variable (REVR = vol_st / vol_mt)
        # Only IEVR is truly independent of the target
        
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
        
        # Create S&P 500 IEVR feature
        self.create_spx_ievr_feature()
        
        # Log transformations (for positive values)
        mask_positive = (self.data['revr'] > 0) & (self.data['ievr'] > 0)
        if mask_positive.sum() > 0:
            self.data.loc[mask_positive, 'log_revr'] = np.log(self.data.loc[mask_positive, 'revr'])
            self.data.loc[mask_positive, 'log_ievr'] = np.log(self.data.loc[mask_positive, 'ievr'])
        
        # Squared terms
        self.data['ievr_squared'] = self.data['ievr'] ** 2
    
    def create_spx_ievr_feature(self):
        """
        Create S&P 500 IEVR feature.
        This captures market-level volatility expectations for comparison with individual stock IEVR.
        """
        print("Creating S&P 500 IEVR feature...")
        
        # Check if we have the necessary data
        if 'spx_ievr' not in self.data.columns:
            print("Warning: 'spx_ievr' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
        else:
            # Check if the column exists but is empty
            if self.data['spx_ievr'].isna().all():
                print("Warning: 'spx_ievr' column exists but is empty. Creating placeholder.")
                self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
        
        # Handle infinite values
        self.data['spx_ievr'] = self.data['spx_ievr'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created spx_ievr feature. Non-null values: {self.data['spx_ievr'].notna().sum()}")
        
        # Print summary statistics
        if self.data['spx_ievr'].notna().sum() > 0:
            print(f"  Mean: {self.data['spx_ievr'].mean():.4f}")
            print(f"  Std: {self.data['spx_ievr'].std():.4f}")
            print(f"  Min: {self.data['spx_ievr'].min():.4f}")
            print(f"  Max: {self.data['spx_ievr'].max():.4f}")
            
            # Check for reasonable values
            if 0.5 <= self.data['spx_ievr'].mean() <= 2.0:
                print(f"  ✓ S&P 500 IEVR is in reasonable range")
            else:
                print(f"  ⚠ S&P 500 IEVR mean ({self.data['spx_ievr'].mean():.3f}) seems unusual")
            
            # Note: Removed relative_ievr feature to avoid multicollinearity with individual ievr and spx_ievr
            print(f"  ✓ Using individual ievr and spx_ievr features (no ratio to avoid redundancy)")
        else:
            print("  ⚠ No valid S&P 500 IEVR data available")
    
    def create_normative_iv_rv_ratio(self):
        """
        Create normative IV/RV ratio feature.
        This calculates the ratio of medium-term implied vol to medium-term realized vol
        at 30 days before earnings (same time point as normative implied vol in IEVR).
        """
        print("Creating normative IV/RV ratio feature...")
        
        # Check if we have the necessary data
        if 'normative_implied_vol' not in self.data.columns:
            print("Warning: 'normative_implied_vol' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['normative_implied_vol'] = self.data['ievr'] * 1.0  # Placeholder
        
        if 'normative_realized_vol' not in self.data.columns:
            print("Warning: 'normative_realized_vol' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your REVR calculation
            self.data['normative_realized_vol'] = 1.0  # Placeholder
        
        # Calculate the ratio
        mask = (self.data['normative_implied_vol'] > 0) & (self.data['normative_realized_vol'] > 0)
        self.data.loc[mask, 'normative_iv_rv_ratio'] = (
            self.data.loc[mask, 'normative_implied_vol'] / 
            self.data.loc[mask, 'normative_realized_vol']
        )
        
        # Handle infinite values
        self.data['normative_iv_rv_ratio'] = self.data['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created normative_iv_rv_ratio feature. Non-null values: {self.data['normative_iv_rv_ratio'].notna().sum()}")
        
        # Print summary statistics
        if self.data['normative_iv_rv_ratio'].notna().sum() > 0:
            print(f"  Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
            print(f"  Std: {self.data['normative_iv_rv_ratio'].std():.4f}")
            print(f"  Min: {self.data['normative_iv_rv_ratio'].min():.4f}")
            print(f"  Max: {self.data['normative_iv_rv_ratio'].max():.4f}")
            
            # Print additional diagnostics
            print(f"\nNormative Values Summary:")
            print(f"  Normative Implied Vol - Mean: {self.data['normative_implied_vol'].mean():.4f}")
            print(f"  Normative Realized Vol - Mean: {self.data['normative_realized_vol'].mean():.4f}")
            print(f"  IV/RV Ratio - Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
            
            # Check for reasonable values
            if self.data['normative_iv_rv_ratio'].mean() > 1.0:
                print(f"  ✓ IV > RV on average (typical volatility risk premium)")
            else:
                print(f"  ⚠ RV > IV on average (unusual)")
    

    
    def create_skew_ratio(self):
        """
        Create skew ratio feature (95Put IV / 105Call IV).
        This captures the directional bias in volatility expectations.
        """
        print("Creating skew ratio feature (90Put / 110Call)...")
        
        # Check if we have the necessary data
        if 'skew_ratio' not in self.data.columns:
            print("Warning: 'skew_ratio' not found in data. Creating placeholder.")
            # Create a placeholder - in practice, this should come from your IEVR calculation
            self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
        else:
            # Check if the column exists but is empty
            if self.data['skew_ratio'].isna().all():
                print("Warning: 'skew_ratio' column exists but is empty. Creating placeholder.")
                self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
        
        # Handle infinite values
        self.data['skew_ratio'] = self.data['skew_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"Created skew_ratio feature. Non-null values: {self.data['skew_ratio'].notna().sum()}")
        
        # Print summary statistics
        if self.data['skew_ratio'].notna().sum() > 0:
            print(f"  Mean: {self.data['skew_ratio'].mean():.4f}")
            print(f"  Std: {self.data['skew_ratio'].std():.4f}")
            print(f"  Min: {self.data['skew_ratio'].min():.4f}")
            print(f"  Max: {self.data['skew_ratio'].max():.4f}")
            
            # Check for reasonable values
            if self.data['skew_ratio'].mean() > 1.0:
                print(f"  ✓ Put skew > Call skew on average (typical for earnings)")
            else:
                print(f"  ⚠ Call skew > Put skew on average (unusual)")
            
            # Check correlation with REVR
            correlation = self.data['revr'].corr(self.data['skew_ratio'])
            print(f"  Correlation with REVR: {correlation:.4f}")
            
            if abs(correlation) > 0.1:
                print(f"  ✓ Skew ratio shows meaningful correlation with REVR")
            else:
                print(f"  ⚠ Skew ratio shows weak correlation with REVR")
    
    def train_random_forest(self, optimize_hyperparameters=True):
        """
        Train Random Forest model.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        if optimize_hyperparameters:
            # Hyperparameter grid for optimization
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
            # Use default parameters
            best_rf = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5, 
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
        Train XGBoost model.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        if optimize_hyperparameters:
            # Hyperparameter grid for optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            best_xgb = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            best_xgb = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
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
        
        Parameters:
        -----------
        model_name : str
            Key name of the model in self.models
        model_display_name : str
            Display name for output
        """
        model = self.models[model_name]
        
        # Predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        
        # Store results
        self.results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        # Print results
        print(f"\n{model_display_name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance for all models.
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        feature_names = self.X_train.columns.tolist()
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()} Feature Importance:")
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models have feature_importances_
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                for i, idx in enumerate(indices):
                    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
            
            # Permutation importance (more robust)
            try:
                perm_importance = permutation_importance(
                    model, self.X_test_scaled, self.y_test, 
                    n_repeats=10, random_state=42
                )
                
                print(f"\n{model_name.upper()} Permutation Importance:")
                perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
                
                for i, idx in enumerate(perm_indices):
                    print(f"  {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} "
                          f"(±{perm_importance.importances_std[idx]:.4f})")
            except Exception as e:
                print(f"  Permutation importance failed: {e}")
    
    def plot_model_comparison(self):
        """
        Create comparison plots for all models.
        """
        print("\n" + "="*60)
        print("CREATING MODEL COMPARISON PLOTS")
        print("="*60)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Non-linear Model Comparison: IEVR vs REVR', fontsize=16)
        
        # Plot 1: Actual vs Predicted (Test Set)
        ax1 = axes[0, 0]
        colors = ['blue', 'red', 'green']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            ax1.scatter(self.y_test, results['y_test_pred'], 
                       alpha=0.6, label=model_name.replace('_', ' ').title(), 
                       color=colors[i % len(colors)])
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), min(r['y_test_pred'].min() for r in self.results.values()))
        max_val = max(self.y_test.max(), max(r['y_test_pred'].max() for r in self.results.values()))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax1.set_xlabel('Actual REVR')
        ax1.set_ylabel('Predicted REVR')
        ax1.set_title('Actual vs Predicted (Test Set)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        ax2 = axes[0, 1]
        for i, (model_name, results) in enumerate(self.results.items()):
            residuals = self.y_test - results['y_test_pred']
            ax2.scatter(results['y_test_pred'], residuals, 
                       alpha=0.6, label=model_name.replace('_', ' ').title(),
                       color=colors[i % len(colors)])
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Predicted REVR')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Performance Comparison
        ax3 = axes[1, 0]
        model_names = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_r2'] for name in model_names]
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        ax3.bar(x + width/2, cv_scores, width, label='CV R²', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature Importance (Random Forest)
        ax4 = axes[1, 1]
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = self.X_train.columns.tolist()
            
            ax4.bar(range(len(importances)), importances[indices])
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Importance')
            ax4.set_title('Random Forest Feature Importance')
            ax4.set_xticks(range(len(importances)))
            ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_files/nonlinear_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison plots saved to output_files/nonlinear_model_comparison.png")
        plt.show()
    
    def print_summary_table(self):
        """
        Print a summary table comparing all models.
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
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
        summary_df.to_csv('data_files/nonlinear_model_summary.csv', index=False)
        print("\n✓ Model summary saved to data_files/nonlinear_model_summary.csv")
    
    def run_complete_analysis(self, optimize_hyperparameters=True):
        """
        Run complete non-linear analysis.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        """
        print("="*80)
        print("NON-LINEAR MACHINE LEARNING ANALYSIS")
        print("="*80)
        
        # Train models
        self.train_random_forest(optimize_hyperparameters)
        self.train_xgboost(optimize_hyperparameters)
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Create plots
        self.plot_model_comparison()
        
        # Print summary
        self.print_summary_table()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def main():
    """
    Main function to run the non-linear analysis.
    """
    # Run analysis with hyperparameter optimization
    analysis = NonlinearModelAnalysis()
    analysis.run_complete_analysis(optimize_hyperparameters=True)


if __name__ == "__main__":
    main() 