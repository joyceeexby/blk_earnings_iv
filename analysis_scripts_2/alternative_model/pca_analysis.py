#!/usr/bin/env python3
"""
PCA Analysis and Prediction for IEVR-REVR Relationship

This module implements Principal Component Analysis (PCA) to:
1. Reduce dimensionality of features
2. Understand feature relationships
3. Predict REVR using PCA components
4. Compare PCA-based predictions with original features

Key Features:
- PCA dimensionality reduction
- Explained variance analysis
- Feature importance in principal components
- Prediction using PCA components
- Comparison with original feature models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.inspection import permutation_importance
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class PCAAnalysis:
    """
    Class for implementing PCA analysis and prediction for IEVR-REVR relationship.
    """
    
    def __init__(self, data_file='data_files/expanded_earnings_analysis_results_with_vix.csv'):
        """
        Initialize the PCA analysis.
        
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
        self.pca = None
        self.pca_components = None
        self.feature_columns = []
        
        # Create output directory
        os.makedirs('output_files', exist_ok=True)
        os.makedirs('data_files', exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """
        Load data and prepare features for PCA analysis.
        """
        print("Loading and preparing data for PCA analysis...")
        
        try:
            # Load data
            self.data = pd.read_csv(self.data_file)
            print(f"✓ Loaded {len(self.data)} rows from {self.data_file}")
            
            # Convert earnings_date to datetime
            self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
            
            # Clean data - remove NaN and infinite values
            initial_size = len(self.data)
            self.data = self.data.dropna(subset=['revr', 'ievr'])
            self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
            print(f"✓ Cleaned data: {len(self.data)} observations (removed {initial_size - len(self.data)} rows)")
            
            # Create additional features (same as nonlinear models)
            self.create_features()
            
            # Prepare features and target - same as nonlinear models
            feature_columns = ['ievr', 'normative_iv_rv_ratio', 'skew_ratio', 'spx_ievr', 'sector_leader_revr', 
                              'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d']
            
            # Remove columns that might not exist
            available_features = [col for col in feature_columns if col in self.data.columns]
            
            if len(available_features) < 2:
                available_features = ['ievr']
                print("Warning: Limited features available, using only IEVR")
            
            X = self.data[available_features].copy()
            y = self.data['revr'].copy()
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            # Update data to match the cleaned version
            self.data = self.data[mask].reset_index(drop=True)
            
            # Set feature columns
            self.feature_columns = available_features
            
            print(f"✓ Final dataset: {len(self.data)} observations, {len(available_features)} features")
            print(f"✓ Features: {available_features}")
            
            # Sort by date for temporal splitting
            self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
            
            # Create temporal split (80% train, 20% test)
            split_idx = int(len(self.data) * 0.8)
            
            # Split data temporally
            train_data = self.data.iloc[:split_idx]
            test_data = self.data.iloc[split_idx:]
            
            self.X_train = train_data[available_features].copy()
            self.y_train = train_data['revr'].copy()
            self.X_test = test_data[available_features].copy()
            self.y_test = test_data['revr'].copy()
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print(f"✓ Training set: {len(self.X_train)} observations")
            print(f"✓ Test set: {len(self.X_test)} observations")
            print(f"✓ Training period: {train_data['earnings_date'].min().strftime('%Y-%m-%d')} to {train_data['earnings_date'].max().strftime('%Y-%m-%d')}")
            print(f"✓ Test period: {test_data['earnings_date'].min().strftime('%Y-%m-%d')} to {test_data['earnings_date'].max().strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_features(self):
        """
        Create additional features for the analysis.
        Same methodology as nonlinear models.
        """
        print("Creating additional features (same as nonlinear models)...")
        
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
        
        # Create S&P 500 IEVR feature
        self.create_spx_ievr_feature()
        
        # Create sector leader REVR feature
        self.create_sector_leader_revr_feature(min_days_gap=30)
    
    def create_normative_iv_rv_ratio(self):
        """
        Create normative IV/RV ratio feature.
        """
        print("Creating normative IV/RV ratio feature...")
        
        # Calculate normative implied vol (using median of IEVR)
        normative_iv = self.data['ievr'].median()
        
        # Calculate normative realized vol (using median of REVR)
        normative_rv = self.data['revr'].median()
        
        # Create ratio
        self.data['normative_iv_rv_ratio'] = normative_iv / normative_rv if normative_rv > 0 else 1.0
        
        # Add normative values as features
        self.data['normative_implied_vol'] = normative_iv
        self.data['normative_realized_vol'] = normative_rv
        
        print(f"  Normative IV/RV ratio: {self.data['normative_iv_rv_ratio'].iloc[0]:.4f}")
        print(f"  Normative Implied Vol: {normative_iv:.4f}")
        print(f"  Normative Realized Vol: {normative_rv:.4f}")
    
    def create_skew_ratio(self):
        """
        Create skew ratio feature (90Put / 110Call).
        """
        print("Creating skew ratio feature...")
        
        # For now, create a placeholder based on typical put skew
        # In a real implementation, this would use actual option skew data
        self.data['skew_ratio'] = 1.3  # Typical put skew ratio
        
        # Calculate correlation with REVR
        correlation = self.data['skew_ratio'].corr(self.data['revr'])
        print(f"  Skew ratio correlation with REVR: {correlation:.4f}")
    
    def create_spx_ievr_feature(self):
        """
        Create S&P 500 IEVR feature.
        """
        print("Creating S&P 500 IEVR feature...")
        
        # For now, create a placeholder
        # In a real implementation, this would use actual S&P 500 IEVR data
        self.data['spx_ievr'] = 1.0  # Placeholder value
        
        print(f"  S&P 500 IEVR placeholder: {self.data['spx_ievr'].iloc[0]:.4f}")
    
    def create_sector_leader_revr_feature(self, min_days_gap=30):
        """
        Create sector leader REVR feature.
        """
        print("Creating sector leader REVR feature...")
        
        # For now, create a placeholder
        # In a real implementation, this would use actual sector peer data
        self.data['sector_leader_revr'] = np.nan  # Placeholder
        
        print(f"  Sector leader REVR: placeholder (NaN values)")
    
    def perform_pca_analysis_on_target(self, n_components=None):
        """
        Perform PCA analysis on target variable (REVR) using training data only.
        FEATURE INDEPENDENT ANALYSIS.
        
        Parameters:
        -----------
        n_components : int or None
            Number of components to keep. If None, keep all components.
        """
        print("\n" + "="*80)
        print("PCA ANALYSIS ON TARGET VARIABLE (REVR) - FEATURE INDEPENDENT")
        print("="*80)
        
        # Prepare target data for PCA (time series windows of REVR)
        target_data_train = self.prepare_target_for_pca_training()
        target_data_test = self.prepare_target_for_pca_test()
        
        if target_data_train is None or target_data_test is None:
            print("Insufficient data for target PCA analysis")
            return None
        
        # Perform PCA on training target data only
        if n_components is None:
            n_components = min(target_data_train.shape[1], target_data_train.shape[0])
        
        self.pca_target = PCA(n_components=n_components, random_state=42)
        self.target_components_train = self.pca_target.fit_transform(target_data_train)
        
        # Transform test target data using fitted PCA (no refitting)
        self.target_components_test = self.pca_target.transform(target_data_test)
        
        # Analyze explained variance
        explained_variance_ratio = self.pca_target.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"✓ PCA fitted on training REVR data with {n_components} components")
        print(f"✓ Test REVR data transformed using fitted PCA")
        print(f"✓ Total explained variance: {cumulative_variance[-1]:.4f}")
        
        # Print component details
        print(f"\nREVR Component Analysis:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"  PC{i+1}: {var_ratio:.4f} ({cum_var:.4f} cumulative)")
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"\n✓ {n_components_95} components explain 95% of REVR variance")
        
        # Analyze REVR structure
        self.analyze_revr_structure()
        
        # Plot explained variance
        self.plot_target_explained_variance(explained_variance_ratio, cumulative_variance)
        
        return n_components_95
    
    def prepare_target_for_pca_training(self, window_size=10):
        """
        Prepare training target variable (REVR) for PCA analysis.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for REVR analysis
            
        Returns:
        --------
        np.array : Matrix where each row is a window of REVR values from training data
        """
        print(f"Preparing training REVR data for PCA analysis (window size: {window_size})...")
        
        # Get training REVR values in chronological order
        train_data = self.data.iloc[:int(len(self.data) * 0.8)]  # Training data
        revr_series = train_data.sort_values('earnings_date')['revr'].values
        
        if len(revr_series) < window_size:
            print(f"Insufficient training data: {len(revr_series)} observations < {window_size} window size")
            return None
        
        # Create rolling windows
        windows = []
        for i in range(len(revr_series) - window_size + 1):
            window = revr_series[i:i+window_size]
            windows.append(window)
        
        target_matrix = np.array(windows)
        
        print(f"✓ Created {len(windows)} training REVR windows of size {window_size}")
        print(f"✓ Training target matrix shape: {target_matrix.shape}")
        
        return target_matrix
    
    def prepare_target_for_pca_test(self, window_size=10):
        """
        Prepare test target variable (REVR) for PCA analysis.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for REVR analysis
            
        Returns:
        --------
        np.array : Matrix where each row is a window of REVR values from test data
        """
        print(f"Preparing test REVR data for PCA analysis (window size: {window_size})...")
        
        # Get test REVR values in chronological order
        test_data = self.data.iloc[int(len(self.data) * 0.8):]  # Test data
        revr_series = test_data.sort_values('earnings_date')['revr'].values
        
        if len(revr_series) < window_size:
            print(f"Insufficient test data: {len(revr_series)} observations < {window_size} window size")
            return None
        
        # Create rolling windows
        windows = []
        for i in range(len(revr_series) - window_size + 1):
            window = revr_series[i:i+window_size]
            windows.append(window)
        
        target_matrix = np.array(windows)
        
        print(f"✓ Created {len(windows)} test REVR windows of size {window_size}")
        print(f"✓ Test target matrix shape: {target_matrix.shape}")
        
        return target_matrix
    
    def prepare_target_for_pca(self, window_size=10):
        """
        Prepare target variable (REVR) for PCA analysis by creating time series windows.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for REVR analysis
            
        Returns:
        --------
        np.array : Matrix where each row is a window of REVR values
        """
        print(f"Preparing REVR data for PCA analysis (window size: {window_size})...")
        
        # Get REVR values in chronological order
        revr_series = self.data.sort_values('earnings_date')['revr'].values
        
        if len(revr_series) < window_size:
            print(f"Insufficient data: {len(revr_series)} observations < {window_size} window size")
            return None
        
        # Create rolling windows
        windows = []
        for i in range(len(revr_series) - window_size + 1):
            window = revr_series[i:i+window_size]
            windows.append(window)
        
        target_matrix = np.array(windows)
        
        print(f"✓ Created {len(windows)} REVR windows of size {window_size}")
        print(f"✓ Target matrix shape: {target_matrix.shape}")
        
        return target_matrix
    
    def analyze_revr_structure(self):
        """
        Analyze the structure of REVR based on PCA components.
        """
        print(f"\nREVR Structure Analysis:")
        
        # Analyze the first few principal components
        for i in range(min(3, len(self.pca_target.components_))):
            component = self.pca_target.components_[i]
            print(f"\nPC{i+1} REVR Pattern:")
            print(f"  Component weights: {component}")
            print(f"  Explained variance: {self.pca_target.explained_variance_ratio_[i]:.4f}")
            
            # Interpret the pattern
            if i == 0:
                print(f"  Interpretation: Primary REVR pattern (trend/level)")
            elif i == 1:
                print(f"  Interpretation: Secondary REVR pattern (volatility/oscillation)")
            elif i == 2:
                print(f"  Interpretation: Tertiary REVR pattern (higher frequency)")
        
        # Analyze REVR characteristics
        revr_stats = self.data['revr'].describe()
        print(f"\nREVR Statistics:")
        print(f"  Mean: {revr_stats['mean']:.4f}")
        print(f"  Std: {revr_stats['std']:.4f}")
        print(f"  Min: {revr_stats['min']:.4f}")
        print(f"  Max: {revr_stats['max']:.4f}")
        print(f"  Skewness: {self.data['revr'].skew():.4f}")
        print(f"  Kurtosis: {self.data['revr'].kurtosis():.4f}")
    
    def plot_target_explained_variance(self, explained_variance_ratio, cumulative_variance):
        """
        Plot explained variance for target variable PCA.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot individual explained variance
        ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('REVR Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
        ax2.axhline(y=0.95, color='b', linestyle='--', label='95% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('REVR Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_files/revr_pca_explained_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ REVR explained variance plot saved to output_files/revr_pca_explained_variance.png")
    
    def analyze_feature_contributions(self):
        """
        Analyze how each feature contributes to principal components.
        """
        print(f"\nFeature Contributions to Principal Components:")
        
        for i in range(min(5, len(self.pca.components_))):  # Show first 5 components
            print(f"\nPC{i+1} Feature Weights:")
            component_weights = self.pca.components_[i]
            
            # Sort features by absolute weight
            feature_weights = list(zip(self.feature_columns, component_weights))
            feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, weight in feature_weights:
                print(f"  {feature}: {weight:.4f}")
    
    def plot_explained_variance(self, explained_variance_ratio, cumulative_variance):
        """
        Plot explained variance and cumulative variance.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot individual explained variance
        ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative explained variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output_files/pca_explained_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Explained variance plot saved to output_files/pca_explained_variance.png")
    
    def predict_with_pca_components(self, n_components):
        """
        Predict REVR using PCA components (proper train/test split).
        
        Parameters:
        -----------
        n_components : int
            Number of PCA components to use for prediction
        """
        print(f"\n" + "="*80)
        print(f"PREDICTION WITH PCA COMPONENTS ({n_components} components)")
        print("="*80)
        
        # Use pre-computed PCA components (fitted on training data)
        pca_train = self.pca_components_train[:, :n_components]
        pca_test = self.pca_components_test[:, :n_components]
        
        # Train models on PCA components
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        pca_results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name} on PCA components...")
            
            # Train model
            model.fit(pca_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(pca_train)
            y_pred_test = model.predict(pca_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, pca_train, self.y_train, cv=5, scoring='r2')
            
            pca_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"  {model_name} Results:")
            print(f"    Training R²: {train_r2:.4f}")
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Training RMSE: {train_rmse:.4f}")
            print(f"    Test RMSE: {test_rmse:.4f}")
            print(f"    CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return pca_results
    
    def compare_with_original_features(self):
        """
        Compare PCA-based predictions with original feature predictions.
        """
        print(f"\n" + "="*80)
        print("COMPARISON: PCA vs ORIGINAL FEATURES")
        print("="*80)
        
        # Train models on original features
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        original_results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name} on original features...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
            
            original_results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"  {model_name} Results:")
            print(f"    Training R²: {train_r2:.4f}")
            print(f"    Test R²: {test_r2:.4f}")
            print(f"    Training RMSE: {train_rmse:.4f}")
            print(f"    Test RMSE: {test_rmse:.4f}")
            print(f"    CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return original_results
    
    def create_comparison_summary(self, pca_results, original_results, n_components):
        """
        Create a summary comparison between PCA and original features.
        """
        print(f"\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        comparison_data = []
        
        for model_name in pca_results.keys():
            pca_result = pca_results[model_name]
            original_result = original_results[model_name]
            
            comparison_data.append({
                'Model': model_name,
                'PCA_Test_R2': pca_result['test_r2'],
                'Original_Test_R2': original_result['test_r2'],
                'PCA_Test_RMSE': pca_result['test_rmse'],
                'Original_Test_RMSE': original_result['test_rmse'],
                'PCA_CV_R2': pca_result['cv_r2_mean'],
                'Original_CV_R2': original_result['cv_r2_mean'],
                'R2_Improvement': pca_result['test_r2'] - original_result['test_r2'],
                'RMSE_Improvement': original_result['test_rmse'] - pca_result['test_rmse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"Comparison using {n_components} PCA components:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison results
        comparison_df.to_csv('data_files/pca_comparison_results.csv', index=False)
        print(f"\n✓ Comparison results saved to data_files/pca_comparison_results.csv")
        
        return comparison_df
    
    def run_analysis(self):
        """
        Run the complete PCA analysis on target variable (REVR) - FEATURE INDEPENDENT.
        """
        print("PCA ANALYSIS ON TARGET VARIABLE (REVR) - FEATURE INDEPENDENT")
        print("="*80)
        
        # Perform PCA analysis on REVR (training data only)
        n_components_95 = self.perform_pca_analysis_on_target()
        
        if n_components_95 is None:
            print("Could not perform PCA analysis on target variable")
            return None
        
        print(f"\n" + "="*80)
        print("PCA ANALYSIS COMPLETE")
        print("="*80)
        print(f"✓ PCA fitted on training REVR data with {n_components_95} components")
        print(f"✓ Test REVR data transformed using fitted PCA")
        print(f"✓ REVR structure analyzed independently of features")
        print(f"✓ Results saved to output_files/revr_pca_explained_variance.png")
        
        return n_components_95

def main():
    """
    Main function to run PCA analysis.
    """
    try:
        # Initialize PCA analysis
        pca_analyzer = PCAAnalysis()
        
        # Run analysis
        results = pca_analyzer.run_analysis()
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        raise

if __name__ == "__main__":
    main()
