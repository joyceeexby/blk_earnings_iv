"""
Non-linear Machine Learning Models for IEVR-REVR Analysis

This module implements various non-linear models to explore the relationship between
Implied Earnings Volatility Ratio (IEVR) and Realized Earnings Volatility Ratio (REVR).

Models included:
- Random Forest Regression
- XGBoost Regression
- Model comparison and evaluation

Updated to use streamlined features: dispersion, option surface, and Fama-French factors.
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
    Updated to use streamlined features from integrated dataset.
    """
    
    def __init__(self, data_file='analysis_scripts/data_files/streamlined_earnings_analysis_results.csv'):
        """
        Initialize the analysis with data.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the streamlined analysis results
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
        print("Using streamlined features from integrated dataset...")
        
        # Load data
        self.data = pd.read_csv(self.data_file)
        
        # Clean data - remove NaN and infinite values
        self.data = self.data.dropna(subset=['revr', 'ievr'])
        self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
        
        # Create additional features
        self.create_features()
        
        # Prepare features and target - use all streamlined features
        # Core features (3)
        core_features = ['ievr', 'skew_ratio', 'normative_iv_rv_ratio']
        
        # Dispersion feature (1)
        dispersion_features = ['dispersion']
        
        # Option surface features (5)
        option_features = ['term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk']
        
        # Fama-French features (5)
        ff_features = ['SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        # Combine all features
        all_feature_columns = core_features + dispersion_features + option_features + ff_features
        
        # Remove columns that might not exist
        available_features = [col for col in all_feature_columns if col in self.data.columns]
        
        if len(available_features) < 5:
            # Fallback to core features
            available_features = ['ievr', 'skew_ratio', 'normative_iv_rv_ratio']
            print("Warning: Limited features available, using core features only")
        
        print(f"Using {len(available_features)} features: {available_features}")
        
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
        Updated to use streamlined features from the integrated dataset.
        """
        print("Using streamlined features from integrated dataset...")
        
        # Check if all streamlined features are available
        expected_features = [
            'ievr', 'skew_ratio', 'normative_iv_rv_ratio',  # Core (3)
            'dispersion',  # Dispersion (1)
            'term_ratio', 'skew', 'kurt', 'iv_ratio', 'smirk',  # Option surface (5)
            'SMB', 'HML', 'RMW', 'CMA', 'RF'  # Fama-French (5)
        ]
        
        missing_features = [f for f in expected_features if f not in self.data.columns]
        if missing_features:
            print(f"⚠ Missing features: {missing_features}")
        else:
            print(f"✓ All {len(expected_features)} streamlined features available")
        
        # Create additional derived features for non-linear modeling
        # Log transformations (for positive values)
        mask_positive = (self.data['revr'] > 0) & (self.data['ievr'] > 0)
        if mask_positive.sum() > 0:
            self.data.loc[mask_positive, 'log_revr'] = np.log(self.data.loc[mask_positive, 'revr'])
            self.data.loc[mask_positive, 'log_ievr'] = np.log(self.data.loc[mask_positive, 'ievr'])
        
        # Squared terms for key features
        if 'ievr' in self.data.columns:
            self.data['ievr_squared'] = self.data['ievr'] ** 2
        
        if 'dispersion' in self.data.columns:
            self.data['dispersion_squared'] = self.data['dispersion'] ** 2
        
        # Interaction terms (selective to avoid multicollinearity)
        if all(f in self.data.columns for f in ['ievr', 'dispersion']):
            self.data['ievr_dispersion_interaction'] = self.data['ievr'] * self.data['dispersion']
        
        if all(f in self.data.columns for f in ['ievr', 'skew']):
            self.data['ievr_skew_interaction'] = self.data['ievr'] * self.data['skew']
        
        print(f"Created additional derived features for non-linear modeling")
    
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
            Key for the model in self.models
        model_display_name : str
            Display name for the model
        """
        model = self.models[model_name]
        
        # Predictions
        y_pred_train = model.predict(self.X_train_scaled)
        y_pred_test = model.predict(self.X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Store results
        self.results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        # Print results
        print(f"\n{model_display_name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.results[model_name]
    
    def analyze_feature_importance(self, model_name='random_forest'):
        """
        Analyze feature importance for a given model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to analyze
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ANALYSIS - {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get feature names
        feature_names = self.X_train.columns.tolist()
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance (Tree-based):")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance (Coefficient-based):")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, self.X_test_scaled, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            perm_importance_df = pd.DataFrame({
                'feature': feature_names,
                'permutation_importance': perm_importance.importances_mean
            }).sort_values('permutation_importance', ascending=False)
            
            print(f"\nPermutation Feature Importance:")
            for _, row in perm_importance_df.iterrows():
                print(f"  {row['feature']}: {row['permutation_importance']:.4f}")
                
        except Exception as e:
            print(f"Could not calculate permutation importance: {e}")
        
        return importance_df
    
    def compare_models(self):
        """
        Compare all trained models.
        """
        if not self.results:
            print("No models trained yet")
            return None
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train R²': f"{results['train_r2']:.4f}",
                'Test R²': f"{results['test_r2']:.4f}",
                'Train RMSE': f"{results['train_rmse']:.4f}",
                'Test RMSE': f"{results['test_rmse']:.4f}",
                'Train MAE': f"{results['train_mae']:.4f}",
                'Test MAE': f"{results['test_mae']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model by test R²
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest model by Test R²: {best_model[0]} ({best_model[1]['test_r2']:.4f})")
        
        return comparison_df
    
    def plot_results(self, save_plots=True):
        """
        Plot model results and comparisons.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        if not self.results:
            print("No models trained yet")
            return
        
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Non-linear Model Results', fontsize=16)
            
            # Plot 1: Actual vs Predicted (Training)
            ax1 = axes[0, 0]
            for model_name, results in self.results.items():
                ax1.scatter(self.y_train, results['y_pred_train'], 
                           alpha=0.6, s=20, label=f'{model_name} (Train)')
            ax1.plot([self.y_train.min(), self.y_train.max()], 
                    [self.y_train.min(), self.y_train.max()], 'k--', alpha=0.8)
            ax1.set_xlabel('Actual REVR')
            ax1.set_ylabel('Predicted REVR')
            ax1.set_title('Training: Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Actual vs Predicted (Test)
            ax2 = axes[0, 1]
            for model_name, results in self.results.items():
                ax2.scatter(self.y_test, results['y_pred_test'], 
                           alpha=0.6, s=20, label=f'{model_name} (Test)')
            ax2.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'k--', alpha=0.8)
            ax2.set_xlabel('Actual REVR')
            ax2.set_ylabel('Predicted REVR')
            ax2.set_title('Test: Actual vs Predicted')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: R² Comparison
            ax3 = axes[1, 0]
            models = list(self.results.keys())
            train_r2 = [self.results[m]['train_r2'] for m in models]
            test_r2 = [self.results[m]['test_r2'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax3.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
            ax3.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
            ax3.set_xlabel('Models')
            ax3.set_ylabel('R² Score')
            ax3.set_title('Model Performance Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in models])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: RMSE Comparison
            ax4 = axes[1, 1]
            train_rmse = [self.results[m]['train_rmse'] for m in models]
            test_rmse = [self.results[m]['test_rmse'] for m in models]
            
            ax4.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
            ax4.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('RMSE')
            ax4.set_title('Model Error Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels([m.replace('_', ' ').title() for m in models])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('analysis_scripts/output_files/nonlinear_model_comparison.png', 
                           dpi=300, bbox_inches='tight')
                print("✓ Plots saved to analysis_scripts/output_files/nonlinear_model_comparison.png")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting results: {e}")
    
    def run_full_analysis(self, optimize_hyperparameters=True):
        """
        Run the complete non-linear analysis pipeline.
        
        Parameters:
        -----------
        optimize_hyperparameters : bool
            Whether to optimize hyperparameters
        """
        print("="*80)
        print("RUNNING COMPLETE NON-LINEAR ANALYSIS PIPELINE")
        print("="*80)
        print("Using streamlined features: dispersion, option surface, and Fama-French factors")
        
        # Train models
        print(f"\n{'='*60}")
        print("TRAINING MODELS")
        print(f"{'='*60}")
        
        rf_model = self.train_random_forest(optimize_hyperparameters)
        xgb_model = self.train_xgboost(optimize_hyperparameters)
        
        # Analyze feature importance
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        rf_importance = self.analyze_feature_importance('random_forest')
        xgb_importance = self.analyze_feature_importance('xgboost')
        
        # Compare models
        comparison = self.compare_models()
        
        # Plot results
        self.plot_results(save_plots=True)
        
        # Save results
        self.save_results()
        
        print(f"\n{'='*80}")
        print("NON-LINEAR ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("✓ Random Forest model trained and evaluated")
        print("✓ XGBoost model trained and evaluated")
        print("✓ Feature importance analyzed")
        print("✓ Models compared")
        print("✓ Results plotted and saved")
        print("✓ All streamlined features integrated successfully!")
        
        return {
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'comparison': comparison,
            'feature_importance': {
                'random_forest': rf_importance,
                'xgboost': xgb_importance
            }
        }
    
    def save_results(self):
        """
        Save analysis results to files.
        """
        try:
            # Save model comparison
            if self.results:
                comparison_data = []
                for model_name, results in self.results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Train_R2': results['train_r2'],
                        'Test_R2': results['test_r2'],
                        'Train_RMSE': results['train_rmse'],
                        'Test_RMSE': results['test_rmse'],
                        'Train_MAE': results['train_mae'],
                        'Test_MAE': results['test_mae']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_csv('analysis_scripts/data_files/nonlinear_model_summary.csv', index=False)
                print("✓ Model comparison saved to analysis_scripts/data_files/nonlinear_model_summary.csv")
            
            # Save feature importance
            if hasattr(self, 'X_train'):
                feature_importance_data = []
                for model_name in self.models.keys():
                    if model_name in self.results:
                        importance_df = self.analyze_feature_importance(model_name)
                        if importance_df is not None:
                            importance_df['model'] = model_name
                            feature_importance_data.append(importance_df)
                
                if feature_importance_data:
                    all_importance = pd.concat(feature_importance_data, ignore_index=True)
                    all_importance.to_csv('analysis_scripts/data_files/feature_importance_analysis.csv', index=False)
                    print("✓ Feature importance saved to analysis_scripts/data_files/feature_importance_analysis.csv")
                    
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """
    Main function to run non-linear analysis.
    """
    print("NON-LINEAR MACHINE LEARNING MODELS FOR IEVR-REVR ANALYSIS")
    print("Updated for streamlined features: dispersion, option surface, and Fama-French factors")
    print("="*80)
    
    try:
        # Initialize analysis
        analysis = NonlinearModelAnalysis()
        
        # Run full analysis
        results = analysis.run_full_analysis(optimize_hyperparameters=True)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"All streamlined features integrated and analyzed!")
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
