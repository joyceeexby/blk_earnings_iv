"""
Rolling Walk-Forward Analysis for IEVR-REVR Relationship

This module implements a robust rolling walk-forward analysis to evaluate how the relationship
between Implied Earnings Volatility Ratio (IEVR) and Realized Earnings Volatility Ratio (REVR)
evolves over time and how model performance changes in different market conditions.

Key Features:
- Expanding window approach with configurable time periods
- Proper temporal validation (no future data leakage)
- Performance tracking over time
- Model stability analysis
- Robust error handling and data validation
- Configurable parameters and file paths
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import os
from typing import Dict, List, Tuple, Optional, Any
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RollingWalkForwardAnalysis:
    """
    Class for implementing rolling walk-forward analysis for IEVR-REVR relationship.
    """
    
    def __init__(self, 
                 data_file: str = 'sami_model/expanded_earnings_analysis_results.csv',
                 output_dir: str = 'output_files',
                 data_dir: str = 'data_files'):
        """
        Initialize the rolling walk-forward analysis.
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing the analysis results
        output_dir : str
            Directory for output files
        data_dir : str
            Directory for data files
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.data = None
        self.results = []
        self.feature_columns = []
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self) -> None:
        """
        Load data and prepare features for rolling analysis.
        """
        logger.info("Loading and preparing data for rolling walk-forward analysis...")
        
        try:
            # Load data
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(self.data)} rows from {self.data_file}")
            
            # Validate required columns
            required_columns = ['earnings_date', 'revr', 'ievr']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert earnings_date to datetime
            self.data['earnings_date'] = pd.to_datetime(self.data['earnings_date'])
            
            # Sort by date to ensure temporal order
            self.data = self.data.sort_values('earnings_date').reset_index(drop=True)
            
            # Clean data - remove NaN and infinite values for target and main predictor
            initial_size = len(self.data)
            self.data = self.data.dropna(subset=['revr', 'ievr'])
            self.data = self.data[np.isfinite(self.data['revr']) & np.isfinite(self.data['ievr'])]
            
            logger.info(f"Removed {initial_size - len(self.data)} rows with missing/infinite values")
            
            # If sector missing, derive from ticker using same mapping as regression analysis
            if 'sector' not in self.data.columns and 'ticker' in self.data.columns:
                try:
                    from regression_analysis import FixedRegressionAnalysis
                    self.data['sector'] = self.data['ticker'].map(FixedRegressionAnalysis.ticker_to_sector())
                except Exception:
                    pass
            
            # Create additional features (exact same as nonlinear models)
            self.create_features()

            # Prepare features and target - handle optional features properly
            # Core features (required)
            core_features = ['ievr', 'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d', 'normative_iv_rv_ratio']
            core_available = [col for col in core_features if col in self.data.columns]
            
            # Optional features (exclude 'afd' from regressions but keep available in data)
            optional_features = ['sector_leader_revr', 'afd']
            optional_available = [col for col in optional_features if col in self.data.columns]
            
            # Combine features
            feature_columns = core_available + optional_available
            
            if len(core_available) < 2:
                feature_columns = ['ievr']
                logger.warning("Limited core features available, using only IEVR")
            
            X = self.data[feature_columns].copy()
            y = self.data['revr'].copy()
            
            # Handle missing values in optional features with mean imputation (preserve row count)
            for feature in optional_available:
                if X[feature].isna().sum() > 0:
                    feature_mean = X[feature].mean()
                    X[feature] = X[feature].fillna(feature_mean)
                    logger.info(f"Imputed {X[feature].isna().sum()} missing values in {feature} with mean: {feature_mean:.4f}")
            
            # Only require core features and target to be non-NaN
            core_mask = ~(X[core_available].isna().any(axis=1) | y.isna())
            X = X[core_mask]
            y = y[core_mask]
            
            # Update data to match the cleaned version
            self.data = self.data[core_mask].reset_index(drop=True)
            
            # Set feature columns
            self.feature_columns = feature_columns
            
            logger.info(f"Final dataset: {len(self.data)} observations, {len(feature_columns)} features")
            logger.info(f"Features: {feature_columns}")
            logger.info(f"Date range: {self.data['earnings_date'].min().strftime('%Y-%m-%d')} to {self.data['earnings_date'].max().strftime('%Y-%m-%d')}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_features(self) -> None:
        """
        Create additional features for the analysis.
        Exact same methodology as nonlinear models.
        """
        logger.info("Creating additional features (same as nonlinear models)...")
        
        # Create normative IV/RV ratio feature
        self.create_normative_iv_rv_ratio()
        
        # Create skew ratio feature
        self.create_skew_ratio()
        
        # Create S&P 500 IEVR feature
        self.create_spx_ievr_feature()
        
        # Log transformations (for positive values) - same as nonlinear models
        mask_positive = (self.data['revr'] > 0) & (self.data['ievr'] > 0)
        if mask_positive.sum() > 0:
            self.data.loc[mask_positive, 'log_revr'] = np.log(self.data.loc[mask_positive, 'revr'])
            self.data.loc[mask_positive, 'log_ievr'] = np.log(self.data.loc[mask_positive, 'ievr'])
        
        # Squared terms - same as nonlinear models
        self.data['ievr_squared'] = self.data['ievr'] ** 2

        # Sector leader REVR feature from earlier peer within same quarter
        self.create_sector_leader_revr_feature(min_days_gap=30)

    def create_sector_leader_revr_feature(self, min_days_gap: int = 30) -> None:
        """
        For each row, attach `sector_leader_revr` as the average of all sector peers'
        REVR within the same year-quarter whose earnings are at least `min_days_gap`
        days earlier than the current event.
        """
        try:
            required = {'sector', 'earnings_date', 'revr'}
            if not required.issubset(set(self.data.columns)):
                logger.warning("Missing columns for sector leader feature; skipping")
                return

            df = self.data.copy()
            df['earnings_date'] = pd.to_datetime(df['earnings_date'])
            df['_year'] = df['earnings_date'].dt.year
            df['_quarter'] = df['earnings_date'].dt.quarter

            # For each event, find all qualifying peers and take their average REVR
            sector_leader_revr_list = []
            
            for idx, row in df.iterrows():
                # Find all peers in same sector, year, quarter
                peer_mask = (
                    (df['sector'] == row['sector']) &
                    (df['_year'] == row['_year']) &
                    (df['_quarter'] == row['_quarter']) &
                    (df['earnings_date'] <= row['earnings_date'] - pd.Timedelta(days=min_days_gap)) &
                    (df['ticker'] != row['ticker'])  # Exclude self
                )
                
                qualifying_peers = df[peer_mask]
                
                if len(qualifying_peers) > 0:
                    # Take average of all qualifying peers' REVR
                    avg_peer_revr = qualifying_peers['revr'].mean()
                    sector_leader_revr_list.append(avg_peer_revr)
                else:
                    sector_leader_revr_list.append(np.nan)
            
            # Assign back to main dataframe
            self.data['sector_leader_revr'] = sector_leader_revr_list
            self.data['sector_leader_revr'] = self.data['sector_leader_revr'].replace([np.inf, -np.inf], np.nan)

            # Cleanup temporary columns
            self.data.drop(columns=[c for c in ['_year', '_quarter'] if c in self.data.columns], inplace=True)
            logger.info("Created feature: sector_leader_revr (average peer REVR from same quarter, >=30 days earlier)")
            logger.info(f"  Non-null values: {self.data['sector_leader_revr'].notna().sum()}")
        except Exception as e:
            logger.error(f"Error creating sector_leader_revr feature: {str(e)}")
    
    def create_normative_iv_rv_ratio(self) -> None:
        """
        Create normative IV/RV ratio feature.
        """
        logger.info("Creating normative IV/RV ratio feature...")
        
        # Check if we have the necessary data
        if 'normative_implied_vol' not in self.data.columns:
            logger.warning("'normative_implied_vol' not found in data. Creating placeholder.")
            self.data['normative_implied_vol'] = self.data['ievr'] * 1.0  # Placeholder
        
        if 'normative_realized_vol' not in self.data.columns:
            logger.warning("'normative_realized_vol' not found in data. Creating placeholder.")
            self.data['normative_realized_vol'] = 1.0  # Placeholder
        
        # Calculate the ratio
        mask = (self.data['normative_implied_vol'] > 0) & (self.data['normative_realized_vol'] > 0)
        self.data.loc[mask, 'normative_iv_rv_ratio'] = (
            self.data.loc[mask, 'normative_implied_vol'] / 
            self.data.loc[mask, 'normative_realized_vol']
        )
        
        # Handle infinite values
        self.data['normative_iv_rv_ratio'] = self.data['normative_iv_rv_ratio'].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created normative_iv_rv_ratio feature. Non-null values: {self.data['normative_iv_rv_ratio'].notna().sum()}")
        
        # Print summary statistics
        if self.data['normative_iv_rv_ratio'].notna().sum() > 0:
            logger.info(f"  Mean: {self.data['normative_iv_rv_ratio'].mean():.4f}")
            logger.info(f"  Std: {self.data['normative_iv_rv_ratio'].std():.4f}")
            logger.info(f"  Min: {self.data['normative_iv_rv_ratio'].min():.4f}")
            logger.info(f"  Max: {self.data['normative_iv_rv_ratio'].max():.4f}")
            
            # Check for reasonable values
            if self.data['normative_iv_rv_ratio'].mean() > 1.0:
                logger.info(f"  ✓ IV > RV on average (typical volatility risk premium)")
            else:
                logger.warning(f"  ⚠ RV > IV on average (unusual)")
    
    def create_skew_ratio(self) -> None:
        """
        Create skew ratio feature (95Put IV / 105Call IV).
        This captures the directional bias in volatility expectations.
        """
        logger.info("Creating skew ratio feature...")
        
        # Check if we have the necessary data
        if 'skew_ratio' not in self.data.columns:
            logger.warning("'skew_ratio' not found in data. Creating placeholder.")
            self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
        else:
            # Check if the column exists but is empty
            if self.data['skew_ratio'].isna().all():
                logger.warning("'skew_ratio' column exists but is empty. Creating placeholder.")
                self.data['skew_ratio'] = 1.0  # Placeholder (no skew)
        
        # Handle infinite values
        self.data['skew_ratio'] = self.data['skew_ratio'].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created skew_ratio feature. Non-null values: {self.data['skew_ratio'].notna().sum()}")
        
        # Print summary statistics
        if self.data['skew_ratio'].notna().sum() > 0:
            logger.info(f"  Mean: {self.data['skew_ratio'].mean():.4f}")
            logger.info(f"  Std: {self.data['skew_ratio'].std():.4f}")
            logger.info(f"  Min: {self.data['skew_ratio'].min():.4f}")
            logger.info(f"  Max: {self.data['skew_ratio'].max():.4f}")
            
            # Check for reasonable values
            if self.data['skew_ratio'].mean() > 1.0:
                logger.info(f"  ✓ Put skew > Call skew on average (typical for earnings)")
            else:
                logger.warning(f"  ⚠ Call skew > Put skew on average (unusual)")
    
    def create_spx_ievr_feature(self) -> None:
        """
        Create S&P 500 IEVR feature.
        This captures market-level volatility expectations for comparison with individual stock IEVR.
        """
        logger.info("Creating S&P 500 IEVR feature...")
        
        # Check if we have the necessary data
        if 'spx_ievr' not in self.data.columns:
            logger.warning("'spx_ievr' not found in data. Creating placeholder.")
            self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
        else:
            # Check if the column exists but is empty
            if self.data['spx_ievr'].isna().all():
                logger.warning("'spx_ievr' column exists but is empty. Creating placeholder.")
                self.data['spx_ievr'] = 1.0  # Placeholder (no market effect)
        
        # Handle infinite values
        self.data['spx_ievr'] = self.data['spx_ievr'].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created spx_ievr feature. Non-null values: {self.data['spx_ievr'].notna().sum()}")
        
        # Print summary statistics
        if self.data['spx_ievr'].notna().sum() > 0:
            logger.info(f"  Mean: {self.data['spx_ievr'].mean():.4f}")
            logger.info(f"  Std: {self.data['spx_ievr'].std():.4f}")
            logger.info(f"  Min: {self.data['spx_ievr'].min():.4f}")
            logger.info(f"  Max: {self.data['spx_ievr'].max():.4f}")
            
            # Check for reasonable values
            if 0.5 <= self.data['spx_ievr'].mean() <= 2.0:
                logger.info(f"  ✓ S&P 500 IEVR is in reasonable range")
            else:
                logger.warning(f"  ⚠ S&P 500 IEVR mean ({self.data['spx_ievr'].mean():.3f}) seems unusual")
    

    
    def run_rolling_analysis(self, 
                           initial_months: int = 24,
                           step_months: int = 6,
                           min_train_size: int = 50,
                           min_test_size: int = 10,
                           optimize_hyperparameters: bool = True) -> None:
        """
        Run rolling walk-forward analysis using months instead of years for better granularity.
        
        Parameters:
        -----------
        initial_months : int
            Number of months to use for initial training set
        step_months : int
            Number of months to step forward in each iteration
        min_train_size : int
            Minimum number of observations required for training
        min_test_size : int
            Minimum number of observations required for testing
        """
        logger.info("="*80)
        logger.info("ROLLING WALK-FORWARD ANALYSIS")
        logger.info("="*80)
        
        # Get date range
        start_date = self.data['earnings_date'].min()
        end_date = self.data['earnings_date'].max()
        
        logger.info(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Rolling training window: {initial_months} months")
        logger.info(f"Step size: {step_months} months")
        logger.info(f"Training window rolls forward (not expanding)")
        
        # Initialize results storage
        self.results = []
        
        # Calculate initial training window (rolling window approach)
        current_train_start = start_date
        current_train_end = start_date + timedelta(days=initial_months * 30)
        
        iteration = 0
        while current_train_end < end_date:
            iteration += 1
            
            # Define test period
            test_start = current_train_end
            test_end = min(current_train_end + timedelta(days=step_months * 30), end_date)
            
            # Create masks for training and test data (rolling window)
            train_mask = (self.data['earnings_date'] >= current_train_start) & (self.data['earnings_date'] <= current_train_end)
            test_mask = (self.data['earnings_date'] > test_start) & (self.data['earnings_date'] <= test_end)
            
            train_data = self.data[train_mask]
            test_data = self.data[test_mask]
            
            # Check if we have enough data
            if len(train_data) < min_train_size:
                logger.warning(f"Iteration {iteration}: Insufficient training data ({len(train_data)} < {min_train_size})")
                current_train_end += timedelta(days=step_months * 30)
                continue
            
            if len(test_data) < min_test_size:
                logger.warning(f"Iteration {iteration}: Insufficient test data ({len(test_data)} < {min_test_size})")
                current_train_end += timedelta(days=step_months * 30)
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"Training period: {current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')}")
            logger.info(f"Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            logger.info(f"{'='*60}")
            logger.info(f"Training observations: {len(train_data)}")
            logger.info(f"Test observations: {len(test_data)}")
            
            # Prepare features and target
            X_train = train_data[self.feature_columns].copy()
            y_train = train_data['revr'].copy()
            X_test = test_data[self.feature_columns].copy()
            y_test = test_data['revr'].copy()
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate models with hyperparameter optimization (same as nonlinear models)
            try:
                # Linear Regression (no hyperparameter optimization needed)
                lr_results = self.train_and_evaluate_linear_model(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                rf_results = self.train_and_evaluate_model(
                    'Random Forest', 
                    'Random Forest',
                    X_train_scaled, y_train, X_test_scaled, y_test,
                    optimize_hyperparameters=optimize_hyperparameters
                )
                
                xgb_results = self.train_and_evaluate_model(
                    'XGBoost', 
                    'XGBoost',
                    X_train_scaled, y_train, X_test_scaled, y_test,
                    optimize_hyperparameters=optimize_hyperparameters
                )
                
                # Store results
                period_results = {
                    'iteration': iteration,
                    'train_start': current_train_start,
                    'train_end': current_train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_observations': len(train_data),
                    'test_observations': len(test_data),
                    'lr_results': lr_results,
                    'rf_results': rf_results,
                    'xgb_results': xgb_results,
                    'feature_columns': self.feature_columns.copy()
                }
                
                self.results.append(period_results)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                continue
            
            # Move to next period (rolling window)
            current_train_start += timedelta(days=step_months * 30)
            current_train_end += timedelta(days=step_months * 30)
        
        logger.info(f"\n{'='*80}")
        logger.info("ROLLING ANALYSIS COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total periods analyzed: {len(self.results)}")
    
    def train_and_evaluate_linear_model(self,
                                      X_train: np.ndarray, 
                                      y_train: np.ndarray, 
                                      X_test: np.ndarray, 
                                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate Linear Regression model using only core features.
        Linear Regression cannot handle NaN values, so we use only core features.
        
        Parameters:
        -----------
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        
        Returns:
        --------
        dict : Linear Regression results
                """
        try:
            # For Linear Regression, use only core features (no sector_leader_revr, no afd)
            core_features = ['ievr', 'normative_iv_rv_ratio', 'beta_market', 'beta_smb', 'beta_hml', 'vix_momentum_5d']
            core_available = [col for col in core_features if col in self.data.columns]
            
            # Get indices of core features in the feature matrix
            feature_indices = []
            for feature in core_available:
                if feature in self.feature_columns:
                    feature_indices.append(self.feature_columns.index(feature))
            
            # Use only core features for Linear Regression
            X_train_core = X_train[:, feature_indices]
            X_test_core = X_test[:, feature_indices]
            
            # Train Linear Regression
            lr = LinearRegression()
            lr.fit(X_train_core, y_train)
            
            # Make predictions
            y_pred_train = lr.predict(X_train_core)
            y_pred_test = lr.predict(X_test_core)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(lr, X_train_core, y_train, cv=tscv, scoring='r2')
            
            # Feature importance (coefficients) - only for core features
            feature_importance = dict(zip(core_available, lr.coef_))
            
            results = {
                'model_name': 'Linear Regression',
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'model': lr
            }
            
            logger.info(f"Linear Regression Results:")
            logger.info(f"  Training R²: {train_r2:.4f}")
            logger.info(f"  Test R²: {test_r2:.4f}")
            logger.info(f"  Training RMSE: {train_rmse:.4f}")
            logger.info(f"  Test RMSE: {test_rmse:.4f}")
            logger.info(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training Linear Regression: {str(e)}")
            return None
    
    def train_and_evaluate_model(self, 
                               model_name: str, 
                               model_display_name: str,
                               X_train: np.ndarray, 
                               y_train: np.ndarray, 
                               X_test: np.ndarray, 
                               y_test: np.ndarray,
                               optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train and evaluate a single model with hyperparameter optimization.
        Same methodology as nonlinear models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('Random Forest' or 'XGBoost')
        model_display_name : str
            Display name for logging
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
        optimize_hyperparameters : bool
            Whether to perform hyperparameter optimization
        
        Returns:
        --------
        dict : Model results
        """
        try:
            if optimize_hyperparameters:
                if model_name == 'Random Forest':
                    # Hyperparameter grid for Random Forest (same as nonlinear models)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    
                    rf = RandomForestRegressor(random_state=42)
                    # Use TimeSeriesSplit for temporal cross-validation
                    from sklearn.model_selection import TimeSeriesSplit
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    grid_search = GridSearchCV(
                        rf, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")
                    
                elif model_name == 'XGBoost':
                    # Hyperparameter grid for XGBoost (same as nonlinear models)
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                    
                    xgb_model = xgb.XGBRegressor(random_state=42)
                    # Use TimeSeriesSplit for temporal cross-validation
                    from sklearn.model_selection import TimeSeriesSplit
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    grid_search = GridSearchCV(
                        xgb_model, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
                    
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            else:
                # Use default parameters (same as nonlinear models)
                if model_name == 'Random Forest':
                    best_model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=5, 
                        random_state=42
                    )
                elif model_name == 'XGBoost':
                    best_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42,
                        verbosity=0
                    )
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            
            # Train the best model
            best_model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Get feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, best_model.feature_importances_))
            else:
                feature_importance = {col: 0.0 for col in self.feature_columns}
            
            # Print results
            logger.info(f"\n{model_display_name} Results:")
            logger.info(f"  Training R²: {train_r2:.4f}")
            logger.info(f"  Test R²: {test_r2:.4f}")
            logger.info(f"  Training RMSE: {train_rmse:.4f}")
            logger.info(f"  Test RMSE: {test_rmse:.4f}")
            logger.info(f"  Training MAE: {train_mae:.4f}")
            logger.info(f"  Test MAE: {test_mae:.4f}")
            
            # Show top feature importances
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"  Top features: {', '.join([f'{feat}: {imp:.3f}' for feat, imp in top_features])}")
            
            return {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': feature_importance,
                'predictions_summary': {
                    'mean_pred': np.mean(y_test_pred),
                    'std_pred': np.std(y_test_pred),
                    'mean_actual': np.mean(y_test),
                    'std_actual': np.std(y_test)
                }
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            # Return default results
            return {
                'train_r2': np.nan,
                'test_r2': np.nan,
                'train_rmse': np.nan,
                'test_rmse': np.nan,
                'train_mae': np.nan,
                'test_mae': np.nan,
                'feature_importance': {col: 0.0 for col in self.feature_columns},
                'predictions_summary': {
                    'mean_pred': np.nan,
                    'std_pred': np.nan,
                    'mean_actual': np.mean(y_test),
                    'std_actual': np.std(y_test)
                }
            }
    
    def create_performance_plots(self) -> plt.Figure:
        """
        Create plots showing performance over time.
        """
        if not self.results:
            logger.error("No results to plot. Run rolling analysis first.")
            return None
        
        logger.info("Creating performance plots...")
        
        # Prepare data for plotting
        periods = []
        lr_test_r2 = []
        rf_test_r2 = []
        xgb_test_r2 = []
        lr_test_rmse = []
        rf_test_rmse = []
        xgb_test_rmse = []
        train_sizes = []
        test_sizes = []
        
        for result in self.results:
            periods.append(result['test_start'].strftime('%Y-%m'))
            
            # Handle cases where models might have failed
            lr_test_r2.append(result['lr_results']['test_r2'] if result['lr_results'] else np.nan)
            rf_test_r2.append(result['rf_results']['test_r2'] if result.get('rf_results') else np.nan)
            xgb_test_r2.append(result['xgb_results']['test_r2'] if result.get('xgb_results') else np.nan)
            
            lr_test_rmse.append(result['lr_results']['test_rmse'] if result['lr_results'] else np.nan)
            rf_test_rmse.append(result['rf_results']['test_rmse'] if result.get('rf_results') else np.nan)
            xgb_test_rmse.append(result['xgb_results']['test_rmse'] if result.get('xgb_results') else np.nan)
            
            train_sizes.append(result['train_observations'])
            test_sizes.append(result['test_observations'])
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Rolling Walk-Forward Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: R² over time
        # Only plot Linear Regression if we have valid data
        if any(not np.isnan(x) for x in lr_test_r2):
            axes[0, 0].plot(range(len(periods)), lr_test_r2, '^-', label='Linear Regression', 
                           linewidth=2, markersize=4, alpha=0.8, color='red')
        axes[0, 0].plot(range(len(periods)), rf_test_r2, 'o-', label='Random Forest', 
                       linewidth=2, markersize=4, alpha=0.8, color='blue')
        axes[0, 0].plot(range(len(periods)), xgb_test_r2, 's-', label='XGBoost', 
                       linewidth=2, markersize=4, alpha=0.8, color='green')
        axes[0, 0].set_title('Test R² Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Period')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set x-axis labels (show every 3rd period to avoid crowding)
        step = max(1, len(periods) // 10)
        axes[0, 0].set_xticks(range(0, len(periods), step))
        axes[0, 0].set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
        
        # Plot 2: RMSE over time
        # Only plot Linear Regression if we have valid data
        if any(not np.isnan(x) for x in lr_test_rmse):
            axes[0, 1].plot(range(len(periods)), lr_test_rmse, '^-', label='Linear Regression', 
                           linewidth=2, markersize=4, alpha=0.8, color='red')
        axes[0, 1].plot(range(len(periods)), rf_test_rmse, 'o-', label='Random Forest', 
                       linewidth=2, markersize=4, alpha=0.8, color='blue')
        axes[0, 1].plot(range(len(periods)), xgb_test_rmse, 's-', label='XGBoost', 
                       linewidth=2, markersize=4, alpha=0.8, color='green')
        axes[0, 1].set_title('Test RMSE Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time Period')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(0, len(periods), step))
        axes[0, 1].set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
        
        # Plot 3: Training set size over time
        axes[1, 0].plot(range(len(periods)), train_sizes, 'o-', color='green', 
                       linewidth=2, markersize=4, alpha=0.8, label='Training')
        axes[1, 0].plot(range(len(periods)), test_sizes, 's-', color='orange', 
                       linewidth=2, markersize=4, alpha=0.8, label='Test')
        axes[1, 0].set_title('Sample Sizes Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time Period')
        axes[1, 0].set_ylabel('Number of Observations')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(0, len(periods), step))
        axes[1, 0].set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
        
        # Plot 4: Feature importance over time (Random Forest)
        self.plot_feature_importance_over_time(axes[0, 2], periods, step)
        
        # Plot 5: Model comparison (average performance)
        # Filter out NaN values for proper comparison
        valid_lr_r2 = [x for x in lr_test_r2 if not np.isnan(x)]
        valid_rf_r2 = [x for x in rf_test_r2 if not np.isnan(x)]
        valid_xgb_r2 = [x for x in xgb_test_r2 if not np.isnan(x)]
        valid_lr_rmse = [x for x in lr_test_rmse if not np.isnan(x)]
        valid_rf_rmse = [x for x in rf_test_rmse if not np.isnan(x)]
        valid_xgb_rmse = [x for x in xgb_test_rmse if not np.isnan(x)]
        
        # Handle model comparison based on available data
        models_to_plot = []
        avg_r2_values = []
        avg_rmse_values = []
        colors_r2 = []
        colors_rmse = []
        labels = []
        
        if valid_lr_r2:
            models_to_plot.append('Linear Regression')
            avg_r2_values.append(np.mean(valid_lr_r2))
            avg_rmse_values.append(np.mean(valid_lr_rmse))
            colors_r2.append('red')
            colors_rmse.append('lightcoral')
            labels.append('Linear Regression')
        
        if valid_rf_r2:
            models_to_plot.append('Random Forest')
            avg_r2_values.append(np.mean(valid_rf_r2))
            avg_rmse_values.append(np.mean(valid_rf_rmse))
            colors_r2.append('blue')
            colors_rmse.append('lightblue')
            labels.append('Random Forest')
        
        if valid_xgb_r2:
            models_to_plot.append('XGBoost')
            avg_r2_values.append(np.mean(valid_xgb_r2))
            avg_rmse_values.append(np.mean(valid_xgb_rmse))
            colors_r2.append('green')
            colors_rmse.append('lightgreen')
            labels.append('XGBoost')
        
        if models_to_plot:
            x = np.arange(len(models_to_plot))
            width = 0.35
            
            # R² comparison
            axes[1, 1].bar(x - width/2, avg_r2_values, width, 
                          label='R²', color=colors_r2, alpha=0.8)
            axes[1, 1].bar(x + width/2, [-rmse for rmse in avg_rmse_values], width, 
                          label='-RMSE', color=colors_rmse, alpha=0.8)
            
            axes[1, 1].set_title('Average Model Performance', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Models')
            axes[1, 1].set_ylabel('Performance Metric')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(labels)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 6: Average feature importance across all periods
        self.plot_average_feature_importance(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'rolling_walk_forward_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Performance plots saved to {plot_path}")
        
        return fig
    
    def plot_feature_importance_over_time(self, ax, periods, step):
        """
        Plot feature importance over time for Random Forest model.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        periods : list
            List of time periods
        step : int
            Step size for x-axis labels
        """
        # Extract feature importance data for Random Forest
        feature_importance_data = {}
        
        for result in self.results:
            if result['rf_results'] and 'feature_importance' in result['rf_results']:
                period = result['test_start'].strftime('%Y-%m')
                importance = result['rf_results']['feature_importance']
                
                for feature, importance_value in importance.items():
                    if feature not in feature_importance_data:
                        feature_importance_data[feature] = []
                    feature_importance_data[feature].append(importance_value)
        
        # Plot feature importance over time
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_importance_data)))
        
        for i, (feature, importance_values) in enumerate(feature_importance_data.items()):
            if len(importance_values) == len(periods):
                ax.plot(range(len(periods)), importance_values, 
                       marker='o', linewidth=2, markersize=4, alpha=0.8, 
                       color=colors[i], label=feature)
        
        ax.set_title('Feature Importance Over Time (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Feature Importance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, len(periods), step))
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
    
    def plot_average_feature_importance(self, ax):
        """
        Plot average feature importance across all periods.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        """
        # Calculate average feature importance across all periods
        feature_avg_importance = {}
        feature_std_importance = {}
        
        for feature in self.feature_columns:
            importance_values = []
            
            for result in self.results:
                if result['rf_results'] and 'feature_importance' in result['rf_results']:
                    importance = result['rf_results']['feature_importance']
                    if feature in importance:
                        importance_values.append(importance[feature])
            
            if importance_values:
                feature_avg_importance[feature] = np.mean(importance_values)
                feature_std_importance[feature] = np.std(importance_values)
        
        if feature_avg_importance:
            # Sort features by average importance
            sorted_features = sorted(feature_avg_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            features = [item[0] for item in sorted_features]
            avg_importance = [item[1] for item in sorted_features]
            std_importance = [feature_std_importance[feature] for feature in features]
            
            # Create bar plot with error bars
            x = np.arange(len(features))
            bars = ax.bar(x, avg_importance, yerr=std_importance, 
                         capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
            
            ax.set_title('Average Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Average Importance')
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_importance):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of the rolling analysis results.
        """
        if not self.results:
            logger.error("No results to summarize. Run rolling analysis first.")
            return pd.DataFrame()
        
        logger.info("Creating summary table...")
        
        # Prepare summary data
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Iteration': result['iteration'],
                'Train_Start': result['train_start'].strftime('%Y-%m-%d'),
                'Train_End': result['train_end'].strftime('%Y-%m-%d'),
                'Test_Start': result['test_start'].strftime('%Y-%m-%d'),
                'Test_End': result['test_end'].strftime('%Y-%m-%d'),
                'Train_Obs': result['train_observations'],
                'Test_Obs': result['test_observations'],
                'LR_R2': result['lr_results']['test_r2'] if result['lr_results'] else np.nan,
                'RF_R2': result['rf_results']['test_r2'] if result['rf_results'] else np.nan,
                'XGB_R2': result['xgb_results']['test_r2'] if result['xgb_results'] else np.nan,
                'LR_RMSE': result['lr_results']['test_rmse'] if result['lr_results'] else np.nan,
                'RF_RMSE': result['rf_results']['test_rmse'] if result['rf_results'] else np.nan,
                'XGB_RMSE': result['xgb_results']['test_rmse'] if result['xgb_results'] else np.nan,
                'LR_MAE': result['lr_results']['test_mae'] if result['lr_results'] else np.nan,
                'RF_MAE': result['rf_results']['test_mae'] if result['rf_results'] else np.nan,
                'XGB_MAE': result['xgb_results']['test_mae'] if result['xgb_results'] else np.nan
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_path = os.path.join(self.data_dir, 'rolling_walk_forward_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Summary table saved to {summary_path}")
        
        # Calculate and print summary statistics
        self.print_summary_statistics(summary_df)
        
        return summary_df
    
    def print_summary_statistics(self, summary_df: pd.DataFrame) -> None:
        """
        Print summary statistics from the rolling analysis.
        """
        logger.info(f"\n{'='*80}")
        logger.info("ROLLING WALK-FORWARD SUMMARY STATISTICS")
        logger.info(f"{'='*80}")
        
        # Filter out NaN values for statistics
        valid_rf = summary_df.dropna(subset=['RF_R2', 'RF_RMSE', 'RF_MAE'])
        valid_xgb = summary_df.dropna(subset=['XGB_R2', 'XGB_RMSE', 'XGB_MAE'])
        
        if len(valid_rf) > 0:
            logger.info(f"\nRandom Forest Performance ({len(valid_rf)} valid periods):")
            logger.info(f"  Average Test R²: {valid_rf['RF_R2'].mean():.4f}")
            logger.info(f"  Average Test RMSE: {valid_rf['RF_RMSE'].mean():.4f}")
            logger.info(f"  Average Test MAE: {valid_rf['RF_MAE'].mean():.4f}")
            logger.info(f"  Best Test R²: {valid_rf['RF_R2'].max():.4f}")
            logger.info(f"  Worst Test R²: {valid_rf['RF_R2'].min():.4f}")
            logger.info(f"  R² Std Dev: {valid_rf['RF_R2'].std():.4f}")
        
        if len(valid_xgb) > 0:
            logger.info(f"\nXGBoost Performance ({len(valid_xgb)} valid periods):")
            logger.info(f"  Average Test R²: {valid_xgb['XGB_R2'].mean():.4f}")
            logger.info(f"  Average Test RMSE: {valid_xgb['XGB_RMSE'].mean():.4f}")
            logger.info(f"  Average Test MAE: {valid_xgb['XGB_MAE'].mean():.4f}")
            logger.info(f"  Best Test R²: {valid_xgb['XGB_R2'].max():.4f}")
            logger.info(f"  Worst Test R²: {valid_xgb['XGB_R2'].min():.4f}")
            logger.info(f"  R² Std Dev: {valid_xgb['XGB_R2'].std():.4f}")
        
        # Overall statistics
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total periods analyzed: {len(summary_df)}")
        logger.info(f"  Average training set size: {summary_df['Train_Obs'].mean():.1f}")
        logger.info(f"  Average test set size: {summary_df['Test_Obs'].mean():.1f}")
        
        # Model comparison
        if len(valid_rf) > 0 and len(valid_xgb) > 0:
            rf_avg_r2 = valid_rf['RF_R2'].mean()
            xgb_avg_r2 = valid_xgb['XGB_R2'].mean()
            
            logger.info(f"\nModel Comparison:")
            logger.info(f"  Random Forest avg R²: {rf_avg_r2:.4f}")
            logger.info(f"  XGBoost avg R²: {xgb_avg_r2:.4f}")
            
            if rf_avg_r2 > xgb_avg_r2:
                logger.info(f"  Random Forest performs better by {rf_avg_r2 - xgb_avg_r2:.4f}")
            elif xgb_avg_r2 > rf_avg_r2:
                logger.info(f"  XGBoost performs better by {xgb_avg_r2 - rf_avg_r2:.4f}")
            else:
                logger.info(f"  Both models perform similarly")
    
    def run_complete_analysis(self, 
                            initial_months: int = 36,
                            step_months: int = 3,
                            min_train_size: int = 50,
                            min_test_size: int = 10,
                            optimize_hyperparameters: bool = True) -> None:
        """
        Run the complete rolling walk-forward analysis including plots and summary.
        
        Parameters:
        -----------
        initial_months : int
            Number of months to use for initial training set
        step_months : int
            Number of months to step forward in each iteration
        min_train_size : int
            Minimum number of observations required for training
        min_test_size : int
            Minimum number of observations required for testing
        """
        logger.info("="*80)
        logger.info("COMPLETE ROLLING WALK-FORWARD ANALYSIS")
        logger.info("="*80)
        
        # Run rolling analysis
        self.run_rolling_analysis(
            initial_months=initial_months,
            step_months=step_months,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            optimize_hyperparameters=optimize_hyperparameters
        )
        
        # Create plots
        if self.results:
            self.create_performance_plots()
            self.create_summary_table()
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)


def main():
    """
    Main function to run the rolling walk-forward analysis.
    """
    try:
        # Initialize analysis
        analysis = RollingWalkForwardAnalysis()
        
        # Run complete analysis
        analysis.run_complete_analysis(
            initial_months=48,  # 3 years initial training
            step_months=12,      # 6 months step size
            min_train_size=50,  # Minimum 50 observations for training
            min_test_size=10    # Minimum 10 observations for testing
        )
        
        print("\n✓ Rolling walk-forward analysis completed successfully!")
        print("✓ Check the output_files directory for plots")
        print("✓ Check the data_files directory for summary tables")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()